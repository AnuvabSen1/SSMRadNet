import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence

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
    
class RADFE(nn.Module):
    def __init__(self, 
                 fast_time_len=512, fast_time_layers=1, fast_time_linear_dims=[32],  
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[64],
                 conv_channels=8, radar_channels=32): 
        # num_features = sample_len = 512
        # seq_len = chirp_len = 256
        super(RADFE, self).__init__()
        self.channels = radar_channels

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

        self.dim2D = [32, 56]

        self.range_project = nn.Linear(self.fast_time_len, self.dim2D[0])
        self.range_norm = nn.LayerNorm(self.dim2D[0])
        self.azimuth_project = nn.Linear(self.slow_time_dim, self.dim2D[1])
        self.azimuth_norm = nn.LayerNorm(self.dim2D[1])

        # self.project = nn.Conv1d(in_channels=2*self.slow_time_dim, out_channels=self.dim2D[0]*self.dim2D[1], kernel_size=1)


        # self.pool_seg = nn.AdaptiveAvgPool2d(1)
        self.pool_seg = nn.Conv2d(256, 1, kernel_size=1)

        # Segmentation decoder
        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')

        self.upsample1 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer

        self.upsample2 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')  # Output layer


        # Detection decoder
        # self.pool_det = nn.AdaptiveAvgPool2d(16)
        self.pool_det = nn.Conv2d(256, 16, kernel_size=1)

        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same', bias=False)
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
        self.reghead = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding='same',bias=True)


    def encode(self, x):
        # x has dimension (B, fast_time_len, slow_time_len, channels)
        B, sample, chirp, C = x.shape
        assert sample == self.fast_time_len and chirp == self.slow_time_len and C == self.channels

        x_t = self.inputNorm(x)
        for linear, norm in zip(self.fast_linears, self.fast_norms):
            x_t = F.silu(linear(x_t))
            x_t = norm(x_t)

        x_transpose = x_t.permute(0,2,1,3)   # converting from batch*fast_time*slow_time*channel to batch*slow_time*fast_time*channel
        x_fast = x_transpose.reshape(B*self.slow_time_len, self.fast_time_len, self.fast_time_dim)
        
        x_fast = x_fast # + self.fast_time_positional_encoding

        for layer in self.fast_ssm_layers:
            x_fast = layer(x_fast)
            # x_fast = x_fast_ssm + x_fast

        # x_chirp = x_fast[:, -1, :].reshape(B, self.slow_time_len, C) # taking only the last state output for each chirp
        # x_chirp = self.chirp_feature_conv(x_fast).squeeze()

        x_fast_expand = x_fast.reshape(B, self.slow_time_len, self.fast_time_len, self.fast_time_dim) # retrieve the slow time axis

        for linear, norm in zip(self.slow_linears, self.slow_norms):
            x_fast_expand = F.silu(linear(x_fast_expand))
            x_fast_expand = norm(x_fast_expand)

        x_chirp = x_fast_expand.permute(0, 2, 1, 3) # converting from batch, slow_time, fast_time, channel to batch, fast_time, slow_time, channel
        # x_chirp = self.chirp_feature_pooling(x_fast_expand.transpose(1, 2)).squeeze()
        x_chirp = x_chirp.reshape(B*self.fast_time_len, self.slow_time_len, self.slow_time_dim)

        x_slow = x_chirp # + self.slow_time_positional_encoding

        for layer in self.slow_ssm_layers:
            x_slow = layer(x_slow)
            # x_slow = x_slow_ssm + x_slow

        x_output = x_slow.reshape(B, self.fast_time_len, self.slow_time_len, self.slow_time_dim) # recovering fast_time_axis (256, 512, 64)
        x_output = x_output.permute(0, 2, 1, 3) # going back to batch, doppler axis, range axis, channel axis

        # slow ssm output dimension = B, slow_time_len, num_channels

        encoder_features = {"x_fast":x_fast, "x_chirp":x_chirp, "x_slow":x_slow}
        return x_output, encoder_features
    
    def spatial_projection(self, x):
                # (batch, slow_time_len, fast_time_len, slow_time_dim)
        x_range_projected = x.permute(0, 1, 3, 2) # Shape: (batch, slow_time_len, slow_time_dim, fast_time_len)
        x_range_projected = F.silu(self.range_project(x_range_projected))
        x_range_projected = self.range_norm(x_range_projected)

        x_azimuth_projected = x_range_projected.permute(0, 1, 3, 2) # (batch, slow_time_len, fast_time_len, slow_time_dim)
        x_azimuth_projected = F.silu(self.azimuth_project(x_azimuth_projected))
        x_spatial = self.azimuth_norm(x_azimuth_projected) # (batch, slow_time_len, 32, 56)
        return x_spatial

    def decode_det(self, x_spatial):
        # Combine the sequence output with the skip connection.

        x_spatial = self.pool_det(x_spatial) # (batch,  32, slow_time_len)

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
    


    def decode_seg(self, x_spatial):
        # Combine the sequence output with the skip connection.

        x_spatial = self.pool_seg(x_spatial)
        # x_spatial = x_spatial_proj.squeeze(-1).reshape(B, 1, 32, 32)
    
        x_out = F.silu(self.conv0(x_spatial))

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = F.silu(self.conv1_2(x_out))

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        x_out = self.conv2_2(x_out)

        return x_out 


    def forward(self, x):
        x_encoded, encoder_features = self.encode(x)
        x_spatial = self.spatial_projection(x_encoded)
        output_det = self.decode_det(x_spatial)
        output_seg = self.decode_seg(x_spatial)
        # features = {"encoder_features": encoder_features, "decoder_features":decoder_features}
        return output_det, output_seg # , features



class FFTRadNet(nn.Module):
    def __init__(self):
        super(FFTRadNet, self).__init__()
    
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

    model.eval()

    input_size = (512, 256, 32)

    # real = torch.rand(1, 512, 256, 32)
    # imag = torch.rand(1, 512, 256, 32)

    # input = torch.complex(real, imag).to("cuda")

    input = torch.rand(1, 512, 256, 32).to("cuda")


    with torch.no_grad():
        output = model(input)

    print(output['Detection'].shape, output['Segmentation'].shape)


    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"MACs  : {macs}, Params: {params}")