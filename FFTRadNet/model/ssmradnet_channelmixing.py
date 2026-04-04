import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# -----------------------------
# Constants
# -----------------------------
NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

# -----------------------------
# Mamba SSM wrapper
# -----------------------------
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
# Per-Channel Fast-Time SSM
# -----------------------------
class PerChannelFastSSM(nn.Module):
    """
    Run an independent SSM over the fast-time sequence for each channel.

    For each channel c:
      x[..., c]  ∈ R^{B×Ls×Lf×1} --lift--> R^{B×Ls×Lf×D} --SSM--> R^{B×Ls×Lf×D}
      --pool Lf--> R^{B×Ls×D}

    Finally concat across C channels -> R^{B×Ls×(C*D)}.
    """
    def __init__(self, channels: int, d_embed: int = 8, d_state: int = 16):
        super().__init__()
        self.C = channels
        self.D = d_embed

        # Vectorized channel-wise lift/proj with depthwise 1×1 conv over the feature axis.
        # We operate on (N=B*Ls, C, Lf).
        self.lift = nn.Conv1d(in_channels=self.C, out_channels=self.C * self.D,
                              kernel_size=1, groups=self.C, bias=True)

        # One distinct Mamba block per channel (no sharing).
        self.ssm = nn.ModuleList([
            MambaSSMBlock(self.D, d_state=d_state, d_conv=4, expand=2)
            for _ in range(self.C)
        ])

        # Optional small LN for stability after pooling
        self.post_ln = nn.LayerNorm(self.C * self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Lf, Ls, C)
        returns: chan_feat (B, Ls, C*D)  # “concar” vector
        """
        B, Lf, Ls, C = x.shape
        assert C == self.C, f"expected {self.C} channels, got {C}"

        # Arrange to (N=B*Ls, C, Lf)
        x_win = x.permute(0, 2, 3, 1).contiguous().view(B * Ls, C, Lf)

        # Channel-wise lift: (N, C, Lf) -> (N, C*D, Lf) -> (N, C, Lf, D)
        z = self.lift(x_win)                                 # (N, C*D, Lf)
        z = z.view(B * Ls, C, self.D, Lf).permute(0, 1, 3, 2)  # (N, C, Lf, D)

        # Run a separate SSM for each channel over Lf (sequence), collect pooled embeddings
        pooled = []
        for c in range(C):
            zc = z[:, c, :, :]          # (N, Lf, D)
            zc = self.ssm[c](zc)        # (N, Lf, D)
            zc = zc.mean(dim=1)         # (N, D)  # avg pool over fast-time
            pooled.append(zc)

        # Concat per-channel embeddings: (N, C*D) -> (B, Ls, C*D)
        chan_feat = torch.cat(pooled, dim=1).view(B, Ls, C * self.D)
        chan_feat = self.post_ln(chan_feat)
        return chan_feat  # “concar” vector per chirp

class RADFE(nn.Module):
    def __init__(self, 
                 fast_time_len=512, fast_time_layers=0, fast_time_linear_dims=[],   # not used now
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[256],
                 conv_channels=8, radar_channels=32, dropout_prob=0.0,
                 chan_d_embed=8):
        """
        - Per-channel SSM produces (C * chan_d_embed) features per chirp.
        - Slow-time MLP maps that to slow_time_dim (last of slow_time_linear_dims).
        - Slow-time SSM models chirp-to-chirp.
        """
        super().__init__()
        self.channels = radar_channels
        self.fast_time_len = fast_time_len
        self.slow_time_len = slow_time_len
        self.dropout_prob = dropout_prob

        # Normalize per window across channels
        self.inputNorm = nn.LayerNorm(self.channels)

        # --- per-channel SSM over fast-time ---
        self.chan_ssm = PerChannelFastSSM(channels=self.channels, d_embed=chan_d_embed, d_state=16)
        slow_input_dim = self.channels * chan_d_embed  # C * D

        # Slow-time MLP stack: (slow_input_dim -> ... -> slow_time_dim)
        self.slow_linears = nn.ModuleList()
        self.slow_norms   = nn.ModuleList()
        last_dim = slow_input_dim
        for nxt in slow_time_linear_dims:
            self.slow_linears.append(nn.Linear(last_dim, nxt))
            self.slow_norms.append(nn.LayerNorm(nxt))
            last_dim = nxt
        self.slow_time_dim = last_dim

        # Slow-time SSM stack
        self.slow_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.slow_time_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(slow_time_layers)
        ])

        # 2D projection (32×32 grid)
        self.dim2D = [32, 32]
        self.project = nn.Conv1d(
            in_channels=2 * self.slow_time_dim,
            out_channels=self.dim2D[0] * self.dim2D[1],
            kernel_size=1
        )
        self.pool_seg = nn.AdaptiveAvgPool1d(1)

        # ---------------- Segmentation decoder ----------------
        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')

        self.upsample1 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels,     conv_channels,   kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(conv_channels,     conv_channels // 2, kernel_size=3, padding='same')

        self.upsample2 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels // 2,  conv_channels // 2, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(conv_channels // 2,  1, kernel_size=3, padding='same')

        # ---------------- Detection decoder ----------------
        self.pool_det = nn.AdaptiveAvgPool1d(32)

        self.conv1 = nn.Conv2d(self.slow_time_len, 32, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.up2 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)

        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.up3 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)

        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(8)

        self.clshead = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding='same', bias=True)
        self.reghead = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding='same', bias=True)

    # ---------------- Encoder ----------------
    def encode(self, x):
        """
        x: (B, 512, 256, C=32)
        returns:
            x_slow: (B, 256, slow_time_dim)
            x_skip: (B, 256, slow_time_dim)  (pre-SSM skip)
        """
        B, Lf, Ls, C = x.shape
        assert Lf == self.fast_time_len and Ls == self.slow_time_len and C == self.channels

        x_t = self.inputNorm(x)                    # (B, 512, 256, 32)

        # Per-channel SSM over fast-time → concat vector per chirp
        # chan_feat: (B, 256, C*D)
        chan_feat = self.chan_ssm(x_t)

        # Slow-time MLP(s)
        h = chan_feat
        for lin, ln in zip(self.slow_linears, self.slow_norms):
            h = ln(F.silu(lin(h)))

        x_skip = h                                  # pre-SSM skip
        x_slow = x_skip
        for layer in self.slow_ssm_layers:
            x_slow = x_slow + layer(x_slow)

        return x_slow, x_skip

    # ---------------- Detection head ----------------
    def decode_det(self, x, skip):
        B, S, C = x.shape
        x = torch.cat([x, skip], dim=-1)                 # (B, S, 2*slow_dim)
        x_feat = x.permute(0, 2, 1).contiguous()         # (B, 2*slow_dim, S)
        x_proj = self.project(x_feat)                    # (B, 1024, S)
        x_spatial = x_proj.transpose(1, 2).reshape(
            B, self.slow_time_len, self.dim2D[0], self.dim2D[1]
        )  # (B, 256, 32, 32)

        x = F.silu(self.conv1(x_spatial))
        x = self.bn1(x)

        x = F.silu(self.conv2(x))
        x = self.bn2(x)
        x = self.up2(x)

        x = F.silu(self.conv3(x))
        x = self.bn3(x)
        x = self.up3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)

    # ---------------- Segmentation head ----------------
    def decode_seg(self, x, skip):
        B, S, C = x.shape
        x = torch.cat([x, skip], dim=-1)                 # (B, S, 2*slow_dim)
        x_feat = x.permute(0, 2, 1).contiguous()         # (B, 2*slow_dim, S)
        x_proj = self.project(x_feat)                    # (B, 1024, S)
        x_pooled = self.pool_seg(x_proj)                 # (B, 1024, 1)
        x_spatial = x_pooled.transpose(1, 2).reshape(B, 1, self.dim2D[0], self.dim2D[1])  # (B,1,32,32)

        x_out = F.silu(self.conv0(x_spatial))

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = F.silu(self.conv1_2(x_out))

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        x_out = self.conv2_2(x_out)                     # (B,1,128,128)
        return x_out

    def forward(self, x):
        x_encoded, skip = self.encode(x)
        output_det = self.decode_det(x_encoded, skip)    # (B, 3, 128, 224)
        output_seg = self.decode_seg(x_encoded, skip)    # (B, 1, 128, 128)
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
        out = {}

        out['Detection'] = out_det
        # FIX: upsample segmentation logits to match label map (256,224)
        out['Segmentation'] = F.interpolate(
            out_seg, size=(256, 224), mode='bilinear', align_corners=False
        )

        return out

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FFTRadNet().to(device).eval()
    x = torch.rand(1, 512, 256, 32, device=device)
    with torch.no_grad():
        out = model(x)
        print("Detection:", tuple(out["Detection"].shape))
        print("Segmentation:", tuple(out["Segmentation"].shape))

    macs, params = get_model_complexity_info(
        model, (512, 256, 32), as_strings=True, print_per_layer_stat=False, verbose=False
    )
    print(f"RAVEN_channel_independent MACs  (ptflops): {macs},  Params: {params}")

    # from fvcore.nn import FlopCountAnalysis

    # flops = FlopCountAnalysis(model, input)
    # print(f"RAVEN_channel_independent GFLOPs(fvcore): {flops.total()/1e9}")