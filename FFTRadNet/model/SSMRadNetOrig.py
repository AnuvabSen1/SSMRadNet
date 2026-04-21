import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence

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
    
class RADFE(nn.Module):
    def __init__(self, 
                 fast_time_len=512, fast_time_layers=1, fast_time_linear_dims=[32],  
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[64],
                 conv_channels=16, radar_channels=32, dropout_prob=0.3): 
        # num_features = sample_len = 512
        # seq_len = chirp_len = 256
        super(RADFE, self).__init__()
        self.channels = radar_channels
        self.dropout_prob = dropout_prob

        # Encoder: Linear layers with normalization.
        self.inputNorm = nn.LayerNorm(self.channels)
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


        self.slow_linears = nn.ModuleList()
        self.slow_norms = nn.ModuleList()

        last_dim = self.fast_time_dim
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

        self.project = nn.Conv1d(in_channels=2*self.slow_time_dim, out_channels=32*32, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')

        self.upsample1 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer

        self.upsample2 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')  # Output layer


    def encode(self, x):
        # x has dimension (B, fast_time_len, slow_time_len, channels)
        B, sample, chirp, C = x.shape
        assert sample == self.fast_time_len and chirp == self.slow_time_len and C == self.channels

        x_t = self.inputNorm(x)
        for linear, norm in zip(self.fast_linears, self.fast_norms):
            x_t = F.silu(linear(x_t))
            x_t = norm(x_t)

        x_transpose = x_t.permute(0,2,1,3)   # converting from batch*fast_time*slow_time*channel to batch*slow_time*fast_time*channel
        x_fast = x_transpose.reshape(B*self.slow_time_len, self.fast_time_len, C)
        
        x_fast = x_fast # + self.fast_time_positional_encoding

        for layer in self.fast_ssm_layers:
            x_fast = layer(x_fast)
            # x_fast = x_fast_ssm + x_fast

        # x_chirp = x_fast[:, -1, :].reshape(B, self.slow_time_len, C) # taking only the last state output for each chirp
        # x_chirp = self.chirp_feature_conv(x_fast).squeeze()

        x_fast_expand = x_fast

        for linear, norm in zip(self.slow_linears, self.slow_norms):
            x_fast_expand = F.silu(linear(x_fast_expand))
            x_fast_expand = norm(x_fast_expand)


        x_chirp = self.chirp_feature_pooling(x_fast_expand.transpose(1, 2)).squeeze()
        x_chirp = x_chirp.reshape(B, self.slow_time_len, self.slow_time_dim)

        x_s = x_chirp
        x_slow = x_s # + self.slow_time_positional_encoding

        for layer in self.slow_ssm_layers:
            x_slow = layer(x_slow)
            # x_slow = x_slow_ssm + x_slow

        
        # slow ssm output dimension = B, slow_time_len, num_channels

        encoder_features = {"x_fast":x_fast, "x_chirp":x_chirp, "x_slow":x_slow}
        return x_slow, x_s, encoder_features

    def decode(self, x, skip):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, slow_time_len, slow_time_dim*2)
        
        x_spatial_features = x.permute(0, 2, 1)
        x_spatial_proj = self.project(x_spatial_features)
        x_spatial_proj_pooled = self.pool(x_spatial_proj)
        x_spatial = x_spatial_proj_pooled.squeeze(-1).reshape(B, 1, 32, 32)
    
        x_out = F.silu(self.conv0(x_spatial))

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = F.silu(self.conv1_2(x_out))

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        x_out = self.conv2_2(x_out)

        decoder_features = {"x_slow_proj": x_spatial_proj, "x_spatial": x_spatial}

        return x_out, decoder_features

    def forward(self, x):
        x_encoded, skip, encoder_features = self.encode(x)
        output, decoder_features = self.decode(x_encoded, skip)
        features = {"encoder_features": encoder_features, "decoder_features":decoder_features}
        return output, features