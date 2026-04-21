import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
import math

from ptflops import get_model_complexity_info

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna


from mamba_ssm import Mamba

class MambaSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None, bidirectional: bool = True,
                 d_conv: int = 4, expand: int = 2):
        """
        Convenience wrapper that applies the Mamba SSM block and then projects the output back to d_model.
        """
        super(MambaSSMBlock, self).__init__()
        self.ssm = Mamba(d_model, d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SSM block followed by an output projection.
        """
        y = self.ssm(x)
        return F.silu(self.out_proj(y))
    
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)  # normalize over C

    def forward(self, x):            # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)    # -> (N, H, W, C)
        x = self.ln(x)               # LN over last dim (C)
        x = x.permute(0, 3, 1, 2)    # -> (N, C, H, W)
        return x
    
class ComplexRadarMixer(nn.Module):
    """
    Input:  x  (B?, 512, 256, 16)  complex64/complex128
    Params: w256   (12, 256)       learnable complex (12 1D signals across 256)
            w2d    (512, 256)      learnable complex (single 2D signal)
    Output: y  (B?, 512, 256, 192) = (16 * 12)
    """
    def __init__(self, n_signals_along_256: int = 12, dtype=torch.cfloat, device=None):
        super().__init__()
        assert dtype in (torch.cfloat, torch.cdouble), "Use a complex dtype."

        self.n_sig = n_signals_along_256
        self.dtype = dtype

        # --- Initialize 12 complex 1D signals over length 256 ---
        # Initialize with unit-magnitude complex phases + small noise for stability
        phase_1d = torch.rand(self.n_sig, 256, device=device) * 2 * math.pi
        w256_init = torch.cos(phase_1d) + 1j * torch.sin(phase_1d)
        w256_init = w256_init.to(dtype)

        # --- Initialize single complex 2D signal over (512, 256) ---
        phase_2d = torch.rand(512, 256, device=device) * 2 * math.pi
        w2d_init = torch.cos(phase_2d) + 1j * torch.sin(phase_2d)
        w2d_init = w2d_init.to(dtype)

        self.w256 = nn.Parameter(w256_init)   # (12, 256)
        self.w2d  = nn.Parameter(w2d_init)    # (512, 256)

        self.w256_norm = nn.LayerNorm (16*12*2)
        self.w2d_norm = nn.LayerNorm(16*12*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 512, 256, 16) complex tensor
        returns: (..., 512, 256, 192)
        """
        x = torch.complex(x[:, :, :, :16], x[:, :, :, 16:])
        if not torch.is_complex(x):
            raise TypeError("Input x must be a complex tensor (torch.cfloat/cdouble).")
        if x.shape[-3:] != (512, 256, 16):
            raise ValueError(f"Expected last dims (512, 256, 16), got {x.shape[-3:]}")

        B_prefix = x.shape[:-3]  # allow optional batch or leading dims
        # Step 1: multiply by 12 complex signals along the 256-dim and expand channels
        # Broadcast w256: (12,256) -> (1,...,1, 1, 256, 1, 12)
        w256_b = self.w256.view(*([1]*len(B_prefix)), 1, 256, 1, self.n_sig)
        # x: (..., 512, 256, 16) -> (..., 512, 256, 16, 1)
        x_exp = x.unsqueeze(-1)
        y = x_exp * w256_b  # (..., 512, 256, 16, 12)
        # Merge (16, 12) -> 192
        y = y.reshape(*B_prefix, 512, 256, 16 * self.n_sig)  # (..., 512, 256, 192)
        
        
        y = torch.cat([y.real, y.imag], dim=-1)
        y = self.w256_norm(y)
        y = torch.complex(y[:, :, :, :192], y[:, :, :, 192:])

        # Step 2: multiply by single complex 2D signal over (512,256)
        # w2d: (512,256) -> (1,...,1, 512, 256, 1)
        w2d_b = self.w2d.view(*([1]*len(B_prefix)), 512, 256, 1)
        y = y * w2d_b  # (..., 512, 256, 192)
        
        y = torch.cat([y.real, y.imag], dim=-1)
        y = self.w2d_norm(y)

        return y
    

class RadarMixer(nn.Module):
    """
    Input:  x  (B?, 512, 256, 32) 
    Params: w256   (12, 256)       learnable  (12 1D signals across 256)
            w2d    (512, 256)      learnable  (single 2D signal)
    Output: y  (B?, 512, 256, 192*2) = (16 * 12*2)
    """
    def __init__(self, n_signals_along_256: int = 12, dtype=torch.cfloat, device=None):
        super().__init__()
        assert dtype in (torch.cfloat, torch.cdouble), "Use a complex dtype."

        self.n_sig = n_signals_along_256
        self.dtype = dtype

        # --- Initialize 12 complex 1D signals over length 256 ---
        # Initialize with unit-magnitude complex phases + small noise for stability


        self.w256 = nn.Parameter(torch.ones(1, 12, 256))   # (12, 256)
        self.w2d  = nn.Parameter(torch.ones(512, 256))   # (512, 256)

        self.w256_norm = nn.LayerNorm (16*12*2)
        self.w2d_norm = nn.LayerNorm(16*12*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 512, 256, 16) complex tensor
        returns: (..., 512, 256, 192)
        """
        B_prefix = x.shape[:-3]  # allow optional batch or leading dims
        # Step 1: multiply by 12 complex signals along the 256-dim and expand channels
        # Broadcast w256: (12,256) -> (1,...,1, 1, 256, 1, 12)
        w256_b = self.w256.view(*([1]*len(B_prefix)), 1, 256, 1, self.n_sig)
        # x: (..., 512, 256, 16) -> (..., 512, 256, 16, 1)
        x_exp = x.unsqueeze(-1)
        y = x_exp * w256_b  # (..., 512, 256, 16, 12)
        # Merge (16, 12) -> 192
        y = y.reshape(*B_prefix, 512, 256, 32 * self.n_sig)  # (..., 512, 256, 192)
        
        
        y = self.w256_norm(F.silu(y))

        # Step 2: multiply by single complex 2D signal over (512,256)
        # w2d: (512,256) -> (1,...,1, 512, 256, 1)
        w2d_b = self.w2d.view(*([1]*len(B_prefix)), 512, 256, 1)
        y = y * w2d_b  # (..., 512, 256, 192)
        
        y = self.w2d_norm(F.silu(y))

        return y
    


class RADFE(nn.Module):
    def __init__(self, 
                 fast_time_len=512, fast_time_layers=1, fast_time_linear_dims=[192],  
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[256],
                 conv_channels=16, receiver_channels = 16*2, radar_channels=16*12*2, dropout_prob=0.3): 
        # num_features = sample_len = 512
        # seq_len = chirp_len = 256
        super(RADFE, self).__init__()
        self.channels = radar_channels
        self.dropout_prob = dropout_prob

        # Encoder: Linear layers with normalization.
        self.inputNorm = nn.LayerNorm(receiver_channels)
        self.window_mixer = RadarMixer()
        self.fast_linears = nn.ModuleList()
        self.fast_norms = nn.ModuleList()

        last_dim = self.channels
        for next_dim in fast_time_linear_dims:
            self.fast_linears.append(nn.Linear(last_dim, next_dim))
            self.fast_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim

        self.fast_time_dim = last_dim
        self.fast_time_len = fast_time_len

        self.fast_time_positional_encoding = nn.Parameter(torch.zeros(1, self.fast_time_len, self.fast_time_dim))

        # Use MambaSSMBlock as VIM layers; stacking multiple layers helps learn deeper representations.
        self.fast_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.fast_time_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(fast_time_layers)
        ])

        # self.fast_linear_compress = nn.Linear(self.fast_time_len, 32)       # compressing the 512 into 32
        self.fast_down = nn.Sequential(
            nn.Conv1d(self.fast_time_len, self.fast_time_len//2, kernel_size=1),  # depthwise in time
            nn.SiLU(),
            nn.Conv1d(self.fast_time_len//2, 32, kernel_size=1),
        )
        self.fast_norm_compress = nn.LayerNorm(32)

        self.slow_linears = nn.ModuleList()
        self.slow_norms = nn.ModuleList()

        last_dim = self.fast_time_dim * 32      # 32*32 size = 1024 flattened array of necessary range x azimuth information
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

        self.dim2D = [32, 56]

        self.project = nn.Conv1d(in_channels=self.slow_time_dim, out_channels=self.dim2D[0]*self.dim2D[1], kernel_size=1)
        self.pool = nn.Conv1d(in_channels=self.slow_time_len, out_channels=16, kernel_size=1)

        # Segmentation decoder
        self.conv0 = nn.Conv2d(16, conv_channels, kernel_size=3, padding='same')
        self.norm0 = LayerNorm2d(conv_channels)

        self.upsample1 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.norm1_1 = LayerNorm2d(conv_channels)
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.norm1_2 = LayerNorm2d(conv_channels//2)

        self.upsample2 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.norm2_1 = LayerNorm2d(conv_channels//2)
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')  # Output layer


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
        self.reghead = nn.Conv2d(conv_channels//2, 2, kernel_size=3, stride=1, padding='same',bias=True)


    def encode(self, x):
        # x has dimension (B, fast_time_len, slow_time_len, channels)
        B, sample, chirp, C = x.shape
        # assert sample == self.fast_time_len and chirp == self.slow_time_len and C == self.channels

        x_t = self.inputNorm(x)
        x_t = self.window_mixer(x_t)
        for linear, norm in zip(self.fast_linears, self.fast_norms):
            x_t = F.silu(linear(x_t))
            x_t = norm(x_t)

        x_transpose = x_t.permute(0,2,1,3)   # converting from batch*fast_time*slow_time*channel to batch*slow_time*fast_time*channel
        x_fast = x_transpose.reshape(B*self.slow_time_len, self.fast_time_len, self.fast_time_dim)
        
        x_fast = x_fast + self.fast_time_positional_encoding

        for layer in self.fast_ssm_layers:
            x_fast = x_fast + layer(x_fast)     # adding residuals
            # x_fast = x_fast_ssm + x_fast

        # x_chirp = x_fast[:, -1, :].reshape(B, self.slow_time_len, C) # taking only the last state output for each chirp
        # x_chirp = self.chirp_feature_conv(x_fast).squeeze()

        # batch*slow_time_len, fast_time_len, fast_time_dim
        # x_fast_compress = F.silu(self.fast_linear_compress(x_fast.permute(0, 2, 1))) # batch*slow_time_len, fast_time_dim, fast_time_len -> batch*slow_time_len, fast_time_dim, 32
        x_fast_compress = F.silu(self.fast_down(x_fast))
        x_fast_expand = self.fast_norm_compress(x_fast_compress.transpose(1,2)).reshape(B*self.slow_time_len, 32*self.fast_time_dim) # batch*slow_time_len, fast_time_dim* 32

        for linear, norm in zip(self.slow_linears, self.slow_norms):
            x_fast_expand = F.silu(linear(x_fast_expand))
            x_fast_expand = norm(x_fast_expand)
            

        # x_chirp = self.chirp_feature_pooling(x_fast_expand.transpose(1, 2)).squeeze()
        x_chirp = x_fast_expand.reshape(B, self.slow_time_len, self.slow_time_dim)

        x_s = x_chirp
        x_slow = x_s # + self.slow_time_positional_encoding

        for layer in self.slow_ssm_layers:
            x_slow = x_slow + layer(x_slow)     # adding residuals
            # x_slow = x_slow_ssm + x_slow

        
        # slow ssm output dimension = B, slow_time_len, augmented num_channels

        encoder_features = {"x_fast":x_fast, "x_chirp":x_chirp, "x_slow":x_slow}

        return x_slow, encoder_features
    
        
    def decode_det(self, x):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        
        x_spatial_features = x.permute(0, 2, 1) # Shape: (batch,  slow_time_dim, slow_time_len)
        x_spatial_proj = self.project_det(x_spatial_features) # (batch,  self.dim2D_det[0]* self.dim2D_det[1], slow_time_len)
        x_spatial_proj_pooled = self.pool_det(x_spatial_proj.permute(0, 2, 1)) # (batch, 1, slow_time_len)
        # x_spatial = x_spatial_proj.squeeze(-1).reshape(B, 1, 32, 32)
        x_spatial = x_spatial_proj_pooled.reshape(B, 32, self.dim2D_det[0], self.dim2D_det[1])


        x = F.silu(self.conv1(x_spatial))
        x = self.bn1(x)

        x = F.silu(self.conv2(x))
        x = self.bn2(x)
        x = self.up2(x)

        x = F.silu(self.conv3(x))
        x = self.bn3(x)
        x = self.up3(x)

        x = F.silu(self.conv4(x))
        x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)
    


    def decode_seg(self, x):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        
        x_spatial_features = x.permute(0, 2, 1) # Shape: (batch,  slow_time_dim, slow_time_len)
        x_spatial_proj = self.project(x_spatial_features) # (batch,  slow_time_dim, slow_time_len)
        x_spatial_proj_pooled = self.pool(x_spatial_proj.permute(0, 2, 1))
        # x_spatial = x_spatial_proj.squeeze(-1).reshape(B, 1, 32, 32)
        x_spatial = x_spatial_proj_pooled.reshape(B, 16, self.dim2D[0], self.dim2D[1])
    
        x_out = F.silu(self.conv0(x_spatial))
        x_out = self.norm0(x_out)

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = self.norm1_1(x_out)
        x_out = F.silu(self.conv1_2(x_out))
        x_out = self.norm1_2(x_out)

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        x_out = self.norm2_1(x_out)
        x_out = self.conv2_2(x_out)

        # decoder_features = {"x_slow_proj": x_spatial_proj, "x_spatial": x_spatial}
        return x_out # , decoder_features


    def forward(self, x):
        x_encoded, encoder_features = self.encode(x)
        output_det = self.decode_det(x_encoded)
        output_seg = self.decode_seg(x_encoded)
        # features = {"encoder_features": encoder_features, "decoder_features":decoder_features}
        return output_det, output_seg # , features



class FFTRadNet(nn.Module):
    def __init__(self):
        super(FFTRadNet, self).__init__()


        # self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        # self.RA_decoder = RangeAngle_Decoder()
        self.RADFe = RADFE()
        


    def forward(self,x):
                       
        out = {'Detection':[],'Segmentation':[]}
        
        out_det, out_seg = self.RADFe(x)

        out['Detection'] = out_det

        Y =  F.interpolate(out_seg, (256, 224))
        out['Segmentation'] = Y
        
        return out
    
    # [1, 256, 128, 224])

if __name__=='__main__':

    model = FFTRadNet().to("cuda")
    input_size = (512, 256, 32)

    input = torch.rand(1, 512, 256, 32).to("cuda")
    
    model.eval()

    # with torch.no_grad():
    #     output = model(input)

    macs, params = get_model_complexity_info(model, input_size , as_strings=True, print_per_layer_stat=False, verbose=False)

    print(f"FFTRadNet MACs  : {macs}, FFTRadNet Params: {params}")


'''
(mamba) sayeed@sayeed-Legion:~/RADIal/FFTRadNet$ python 3-Evaluation_RADFE.py 
===========  Dataset  ==================:
      Mode: sequence
      Training: 6118
      Validation: 786
      Test: 744

===========  Loading the model ==================:
===========  Running the evaluation ==================:
Generating Predictions...
743/744 [==================>.] - ETA: 0s------- Detection Scores ------------
  mAP: 0.7955509554970974
  mAR: 0.677040677040677
  F1 score: 0.7315271194389268
------- Regression Errors------------
  Range Error: 0.1355103862370349 m
  Angle Error: 0.2675563071135708 degree
------- Freespace Scores ------------
  mIoU 74.45484956426972 %
'''