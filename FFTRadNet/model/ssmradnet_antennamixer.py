import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# -----------------------------
# Ensure GPU-safe attention kernels (disable fused SDPA)
# -----------------------------
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False 


NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna


from mamba_ssm import Mamba

class MambaSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, bidirectional: bool = True,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.ssm = Mamba(d_model, d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ssm(x)
        return F.silu(self.out_proj(y))

# -----------------------------
# LayerNorm over (N, C, H, W)
# -----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

# --------------------------------------------------------
# AntennaAttentionMixer
#   Input : (B, 512, 256, 32)  where 32 = 16 Rx * 2 (I/Q)
#   Output: (B, 512, 256, 384) = 16 Rx * 12 Tx * 2
#   NOTE: internally micro-batches the (fast,slow) windows to avoid CUDA launch limits.
# --------------------------------------------------------
class AntennaAttentionMixer(nn.Module):
    def __init__(self,
                 n_rx=NbRxAntenna,
                 n_tx=NbTxAntenna,
                 d_model=64,
                 n_heads=4,
                 ff_mult=2,
                 dropout=0.0,
                 attn_chunk: int = None):
        super().__init__()
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.d_model = d_model

        # micro-batch size for attention over windows (B*512*256)
        # override with env var RAD_ATTENTION_CHUNK if desired
        env_chunk = os.getenv("RAD_ATTENTION_CHUNK")
        self.attn_chunk = attn_chunk if attn_chunk is not None else int(env_chunk) if env_chunk else 8192

        # Map (I,Q) -> d_model per Rx token
        self.in_proj = nn.Linear(2, d_model)

        # Rx & Tx embeddings
        self.rx_embed = nn.Embedding(n_rx, d_model)
        self.tx_embed = nn.Embedding(n_tx, d_model)

        # Self-attention across Rx tokens
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        # Post-attn MLP
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.SiLU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.attn_ln = nn.LayerNorm(d_model)
        self.ff_ln = nn.LayerNorm(d_model)

        # Per-Rx projection to (Tx × I/Q) = (12 × 2) = 24
        self.out_per_rx = nn.Linear(d_model, n_tx * 2)

        # Final norm over channel dimension (384)
        self.out_ln = nn.LayerNorm(n_rx * n_tx * 2)

    @torch.no_grad()
    def _make_indices(self, B, Ffast, Fslow, device):
        # cache-friendly index builders
        tx_idx = torch.arange(Fslow, device=device, dtype=torch.long) % self.n_tx
        tx_idx = tx_idx.view(1, 1, Fslow, 1).expand(B, Ffast, Fslow, self.n_rx)  # (B,512,256,16)
        rx_idx = torch.arange(self.n_rx, device=device, dtype=torch.long)
        rx_idx = rx_idx.view(1, 1, 1, self.n_rx).expand(B, Ffast, Fslow, self.n_rx)  # (B,512,256,16)
        return tx_idx, rx_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 512, 256, 32) with I/Q stacked along last dim (16*2)
        B, Ffast, Fslow, C = x.shape
        assert C == self.n_rx * 2, f"Expected channels {self.n_rx*2}, got {C}"

        x = x.reshape(B, Ffast, Fslow, self.n_rx, 2).contiguous()

        # indices & embeddings
        tx_idx, rx_idx = self._make_indices(B, Ffast, Fslow, x.device)

        # per-Rx tokens
        tok = self.in_proj(x)  # (B,512,256,16,2) -> (B,512,256,16,d_model)
        tok = tok + self.rx_embed(rx_idx) + self.tx_embed(tx_idx)

        # flatten windows for attention
        total_windows = B * Ffast * Fslow
        tok = tok.view(total_windows, self.n_rx, self.d_model).contiguous()

        # micro-batched attention across Rx tokens (length=16)
        out = torch.empty_like(tok)
        chunk = self.attn_chunk
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            for start in range(0, total_windows, chunk):
                end = min(start + chunk, total_windows)
                h = self.attn_ln(tok[start:end])
                attn_out, _ = self.attn(h, h, h, need_weights=False)
                t = tok[start:end] + attn_out
                h2 = self.ff_ln(t)
                out[start:end] = t + self.ff(h2)

        # Project to (n_tx*2)=24 per Rx
        per_rx = self.out_per_rx(out)  # (total_windows, 16, 24)

        # back to (B,512,256,16*24=384)
        per_rx = per_rx.view(B, Ffast, Fslow, self.n_rx, self.n_tx * 2)
        y = per_rx.reshape(B, Ffast, Fslow, self.n_rx * self.n_tx * 2).contiguous()

        # final LN over channel
        y = self.out_ln(y)
        return y

# --------------------------------------------------------
# RADFE
# --------------------------------------------------------
class RADFE(nn.Module):
    def __init__(self, 
                 fast_time_len=512, fast_time_layers=1, fast_time_linear_dims=[16],  
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[256],
                 conv_channels=16, receiver_channels=NbRxAntenna*2, radar_channels=NbRxAntenna*NbTxAntenna*2,
                 dropout_prob=0.3):
        super().__init__()
        self.channels = radar_channels
        self.dropout_prob = dropout_prob

        # Encoder front: LN over (I,Q) per Rx
        self.inputNorm = nn.LayerNorm(receiver_channels)

        # Mixer preserving antenna structure
        self.window_mixer = AntennaAttentionMixer(
            n_rx=NbRxAntenna, n_tx=NbTxAntenna, d_model=16, n_heads=4, ff_mult=2, dropout=0.0
        )

        # Fast-time MLP stack
        self.fast_linears = nn.ModuleList()
        self.fast_norms = nn.ModuleList()

        last_dim = self.channels  # 384
        for next_dim in fast_time_linear_dims:
            self.fast_linears.append(nn.Linear(last_dim, next_dim))
            self.fast_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim

        self.fast_time_dim = last_dim
        self.fast_time_len = fast_time_len
        self.fast_time_positional_encoding = nn.Parameter(torch.zeros(1, self.fast_time_len, self.fast_time_dim))

        self.fast_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.fast_time_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(fast_time_layers)
        ])

        # Conv1d along time dimension (length = 512)
        self.fast_down = nn.Sequential(
            nn.Conv1d(self.fast_time_len, self.fast_time_len//2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(self.fast_time_len//2, 32, kernel_size=1),
        )
        self.fast_norm_compress = nn.LayerNorm(32)

        # Slow-time MLP stack
        self.slow_linears = nn.ModuleList()
        self.slow_norms = nn.ModuleList()

        last_dim = self.fast_time_dim * 32
        for next_dim in slow_time_linear_dims:
            self.slow_linears.append(nn.Linear(last_dim, next_dim))
            self.slow_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim

        self.chirp_feature_pooling = nn.AdaptiveAvgPool1d(1)
        self.slow_time_dim = last_dim
        self.slow_time_len = slow_time_len
        self.slow_time_positional_encoding = nn.Parameter(torch.zeros(1, self.slow_time_len, self.slow_time_dim))

        self.slow_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.slow_time_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(slow_time_layers)
        ])

        # Segmentation decoder
        self.dim2D = [32, 56]
        self.project = nn.Conv1d(in_channels=self.slow_time_dim, out_channels=self.dim2D[0]*self.dim2D[1], kernel_size=1)
        self.pool = nn.Conv1d(in_channels=self.slow_time_len, out_channels=16, kernel_size=1)

        self.conv0 = nn.Conv2d(16, conv_channels, kernel_size=3, padding='same')
        self.norm0 = LayerNorm2d(conv_channels)

        self.upsample1 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')
        self.norm1_1 = LayerNorm2d(conv_channels)
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')
        self.norm1_2 = LayerNorm2d(conv_channels//2)

        self.upsample2 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')
        self.norm2_1 = LayerNorm2d(conv_channels//2)
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')

        # Detection decoder
        self.dim2D_det = [32, 56]
        self.project_det = nn.Conv1d(in_channels=self.slow_time_dim, out_channels=self.dim2D_det[0]*self.dim2D_det[1], kernel_size=1)
        self.pool_det = nn.Conv1d(in_channels=self.slow_time_len, out_channels=32, kernel_size=1)

        self.conv1 = nn.Conv2d(32, conv_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = LayerNorm2d(conv_channels)

        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = LayerNorm2d(conv_channels)
        self.up2 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)

        self.conv3 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = LayerNorm2d(conv_channels//2)
        self.up3 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)

        self.conv4 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4 = LayerNorm2d(conv_channels//2)

        self.clshead = nn.Conv2d(conv_channels//2, 1, kernel_size=3, stride=1, padding='same', bias=True)
        self.reghead = nn.Conv2d(conv_channels//2, 2, kernel_size=3, stride=1, padding='same', bias=True)

    def encode(self, x):
        # x: (B, 512, 256, 32)
        B, sample, chirp, C = x.shape
        x_t = self.inputNorm(x)
        x_t = self.window_mixer(x_t)  # -> (B,512,256,384)

        # Fast-time MLP
        for linear, norm in zip(self.fast_linears, self.fast_norms):
            x_t = norm(F.silu(linear(x_t)))

        # (B,512,256,fast_dim)->(B*256,512,fast_dim)
        x_transpose = x_t.permute(0, 2, 1, 3).contiguous()
        x_fast = x_transpose.reshape(B*self.slow_time_len, self.fast_time_len, self.fast_time_dim)
        x_fast = (x_fast + self.fast_time_positional_encoding).contiguous()

        for layer in self.fast_ssm_layers:
            x_fast = x_fast + layer(x_fast)

        # Compress fast-time and prep for slow-time processing
        x_fast_compress = F.silu(self.fast_down(x_fast.contiguous()))
        x_fast_expand = self.fast_norm_compress(x_fast_compress.transpose(1,2).contiguous()).reshape(
            B*self.slow_time_len, 32*self.fast_time_dim)

        for linear, norm in zip(self.slow_linears, self.slow_norms):
            x_fast_expand = norm(F.silu(linear(x_fast_expand)))

        x_chirp = x_fast_expand.reshape(B, self.slow_time_len, self.slow_time_dim)

        # Slow-time SSM
        x_slow = x_chirp
        for layer in self.slow_ssm_layers:
            x_slow = x_slow + layer(x_slow)

        return x_slow, {"x_fast": x_fast, "x_chirp": x_chirp, "x_slow": x_slow}

    def decode_det(self, x):
        B, S, C = x.shape
        x_spatial_features = x.permute(0, 2, 1).contiguous()                 # (B, C, S)
        x_spatial_proj = self.project_det(x_spatial_features)                # (B, 1792, S)
        x_proj_t = x_spatial_proj.permute(0, 2, 1).contiguous()              # (B, S, 1792)
        x_spatial_proj_pooled = self.pool_det(x_proj_t)                      # (B, 32, 1792)
        x_spatial = x_spatial_proj_pooled.reshape(B, 32, 32, 56).contiguous()

        x = self.bn1(F.silu(self.conv1(x_spatial)))
        x = self.up2(self.bn2(F.silu(self.conv2(x))))
        x = self.up3(self.bn3(F.silu(self.conv3(x))))
        x = self.bn4(F.silu(self.conv4(x)))

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)

    def decode_seg(self, x):
        B, S, C = x.shape
        x_spatial_features = x.permute(0, 2, 1).contiguous()                 # (B, C, S)
        x_spatial_proj = self.project(x_spatial_features)                    # (B, 1792, S)
        x_proj_t = x_spatial_proj.permute(0, 2, 1).contiguous()              # (B, S, 1792)
        x_spatial_proj_pooled = self.pool(x_proj_t)                          # (B, 16, 1792)
        x_spatial = x_spatial_proj_pooled.reshape(B, 16, 32, 56).contiguous()

        x_out = self.norm0(F.silu(self.conv0(x_spatial)))
        x_out = self.upsample1(self.norm1_1(F.silu(self.conv1_1(x_out))))
        x_out = self.norm1_2(F.silu(self.conv1_2(x_out)))
        x_out = self.upsample2(self.norm2_1(F.silu(self.conv2_1(x_out))))
        x_out = self.conv2_2(x_out)
        return x_out

    def forward(self, x):
        x_encoded, _ = self.encode(x)
        output_det = self.decode_det(x_encoded)
        output_seg = self.decode_seg(x_encoded)
        return output_det, output_seg

# --------------------------------------------------------
# Wrapper
# --------------------------------------------------------
class FFTRadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.RADFe = RADFE()

    def forward(self, x):
        out_det, out_seg = self.RADFe(x)
        return {
            'Detection': out_det,
            'Segmentation': F.interpolate(out_seg, (256, 224))
        }

# -----------------------------
# Quick check
# -----------------------------
if __name__=='__main__':
    model = FFTRadNet().to("cuda")
    input_size = (512, 256, NbRxAntenna*2)
    x = torch.randn(1, *input_size, device="cuda", dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"RAVEN_attention_mixing MACs (ptflops): {macs},  Params: {params}")

    # from fvcore.nn import FlopCountAnalysis

    # flops = FlopCountAnalysis(model, input)
    # print(f"RAVEN_attention GFLOPs (fvcore): {flops.total()/1e9}")