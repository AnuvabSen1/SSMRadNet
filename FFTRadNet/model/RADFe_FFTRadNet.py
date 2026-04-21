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
                 conv_channels=8, radar_channels=32, dropout_prob=0.3): 
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

        self.dim2D = [32, 56]

        self.project = nn.Conv1d(in_channels=2*self.slow_time_dim, out_channels=self.dim2D[0]*self.dim2D[1], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Segmentation decoder
        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')

        self.upsample1 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer

        self.upsample2 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')  # Output layer


        # Detection decoder
        self.pool_det = nn.AdaptiveAvgPool1d(32)

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same', bias=False)
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

    def decode_det(self, x, skip):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, slow_time_len, slow_time_dim*2)
        
        x_spatial_features = x.permute(0, 2, 1) # Shape: (batch,  slow_time_dim*2, slow_time_len)
        x_spatial_proj = self.project(x_spatial_features) # (batch,  slow_time_dim*2, slow_time_len)
        x_spatial_proj_pooled = self.pool_det(x_spatial_proj) # (batch,  32, slow_time_len)
        # x_spatial = x_spatial_proj.squeeze(-1).reshape(B, 1, 32, 32)
        x_spatial = x_spatial_proj_pooled.transpose(1, 2).reshape(B, 32, self.dim2D[0], self.dim2D[1])
    
        # x_out = F.silu(self.conv0(x_spatial))

        # x_out = self.upsample1(x_out)
        # x_out = F.silu(self.conv1_1(x_out))
        # x_out = F.silu(self.conv1_2(x_out))

        # x_out = self.upsample2(x_out)
        # x_out = F.silu(self.conv2_1(x_out))
        # x_out = self.conv2_2(x_out)

        # decoder_features = {"x_slow_proj": x_spatial_proj, "x_spatial": x_spatial}
        # return x_out, decoder_features


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
    


    def decode_seg(self, x, skip):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, slow_time_len, slow_time_dim*2)
        
        x_spatial_features = x.permute(0, 2, 1) # Shape: (batch,  slow_time_dim*2, slow_time_len)
        x_spatial_proj = self.project(x_spatial_features) # (batch,  slow_time_dim*2, slow_time_len)
        x_spatial_proj_pooled = self.pool(x_spatial_proj)
        # x_spatial = x_spatial_proj.squeeze(-1).reshape(B, 1, 32, 32)
        x_spatial = x_spatial_proj_pooled.transpose(1, 2).reshape(B, 1, self.dim2D[0], self.dim2D[1])
    
        x_out = F.silu(self.conv0(x_spatial))

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = F.silu(self.conv1_2(x_out))

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        x_out = self.conv2_2(x_out)

        # decoder_features = {"x_slow_proj": x_spatial_proj, "x_spatial": x_spatial}
        return x_out # , decoder_features


    def forward(self, x):
        x_encoded, skip, encoder_features = self.encode(x)
        output_det = self.decode_det(x_encoded, skip)
        output_seg = self.decode_seg(x_encoded, skip)
        # features = {"encoder_features": encoder_features, "decoder_features":decoder_features}
        return  output_seg, output_det # , features





class FFTRadNet(nn.Module):
    def __init__(self):
        super(FFTRadNet, self).__init__()
    
        self.RADFe = RADFE()
        

    def forward(self,x):
                       
        out = {'Detection':[],'Segmentation':[]}
        
        out_det, out_seg = self.RADFe(x)

        if(self.detection_head):
            out['Detection'] = out_det

        if(self.segmentation_head):
            Y =  F.interpolate(out_seg, (256, 224))
            out['Segmentation'] = Y
        
        return out
    
    # [1, 256, 128, 224])


'''
Epoch: 1/100
1558/1558 [====================] - 843s 541ms/step - loss: 1402.3377 - class: 1142.9916 - reg: 170.8421 - freeSpace: 88.5041 - val_loss: 1120668.4999 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 2/100
1558/1558 [====================] - 840s 539ms/step - loss: 328.4868 - class: 94.7007 - reg: 167.6559 - freeSpace: 66.1302 - val_loss: 414405.6152 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 3/100
1558/1558 [====================] - 840s 539ms/step - loss: 308.0918 - class: 80.3030 - reg: 163.7742 - freeSpace: 64.0146 - val_loss: 3250922.6947 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 4/100
1558/1558 [====================] - 840s 539ms/step - loss: 295.0860 - class: 73.3468 - reg: 159.0987 - freeSpace: 62.6405 - val_loss: 1938012.0915 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 5/100
1558/1558 [====================] - 840s 539ms/step - loss: 287.6198 - class: 70.3374 - reg: 155.5261 - freeSpace: 61.7564 - val_loss: 7372348.4239 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 6/100
1558/1558 [====================] - 840s 539ms/step - loss: 281.4308 - class: 67.9908 - reg: 152.3821 - freeSpace: 61.0579 - val_loss: 367958.2460 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 7/100
1558/1558 [====================] - 841s 540ms/step - loss: 276.6788 - class: 66.2254 - reg: 149.8565 - freeSpace: 60.5969 - val_loss: 5674521.0343 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 8/100
1558/1558 [====================] - 840s 539ms/step - loss: 271.6250 - class: 64.5809 - reg: 146.7353 - freeSpace: 60.3088 - val_loss: 3277307.4018 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 9/100
1558/1558 [====================] - 840s 539ms/step - loss: 267.2769 - class: 63.2792 - reg: 144.0724 - freeSpace: 59.9253 - val_loss: 4370132.7049 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 10/100
1558/1558 [====================] - 840s 539ms/step - loss: 263.3705 - class: 61.8909 - reg: 141.7819 - freeSpace: 59.6977 - val_loss: 1095043.0678 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 11/100
1558/1558 [====================] - 855s 549ms/step - loss: 259.2617 - class: 60.8118 - reg: 139.1525 - freeSpace: 59.2975 - val_loss: 351470.1552 - mAP: 0.4598 - mAR: 0.5766 - mIoU: 0.4439

Epoch: 12/100
1558/1558 [====================] - 956s 614ms/step - loss: 255.5444 - class: 59.9024 - reg: 136.7410 - freeSpace: 58.9010 - val_loss: 2505649.3620 - mAP: 0.0555 - mAR: 0.8307 - mIoU: 0.1243

Epoch: 13/100
1558/1558 [====================] - 882s 566ms/step - loss: 252.9839 - class: 59.0571 - reg: 135.1622 - freeSpace: 58.7645 - val_loss: 326474.1184 - mAP: 0.1285 - mAR: 0.7846 - mIoU: 0.3029

Epoch: 14/100
1558/1558 [====================] - 859s 551ms/step - loss: 249.9885 - class: 58.3184 - reg: 133.2213 - freeSpace: 58.4489 - val_loss: 349112.1510 - mAP: 0.2871 - mAR: 0.7150 - mIoU: 0.1270

Epoch: 15/100
1558/1558 [====================] - 852s 547ms/step - loss: 247.8518 - class: 57.6263 - reg: 131.9541 - freeSpace: 58.2714 - val_loss: 349279.5571 - mAP: 0.4272 - mAR: 0.6515 - mIoU: 0.4057

Epoch: 16/100
1558/1558 [====================] - 843s 541ms/step - loss: 245.5730 - class: 56.8452 - reg: 130.7388 - freeSpace: 57.9890 - val_loss: 527395.3730 - mAP: 0.8653 - mAR: 0.2719 - mIoU: 0.2146

Epoch: 17/100
1558/1558 [====================] - 842s 540ms/step - loss: 243.4499 - class: 56.2117 - reg: 129.5093 - freeSpace: 57.7289 - val_loss: 390680.5520 - mAP: 0.9123 - mAR: 0.0408 - mIoU: 0.1289

Epoch: 18/100
1558/1558 [====================] - 852s 547ms/step - loss: 241.2590 - class: 55.6352 - reg: 128.0208 - freeSpace: 57.6029 - val_loss: 335197.8208 - mAP: 0.4891 - mAR: 0.6192 - mIoU: 0.4605

Epoch: 19/100
1558/1558 [====================] - 863s 554ms/step - loss: 238.9984 - class: 55.0434 - reg: 126.5927 - freeSpace: 57.3623 - val_loss: 541286.2250 - mAP: 0.2255 - mAR: 0.7474 - mIoU: 0.1312

Epoch: 20/100
1558/1558 [====================] - 846s 543ms/step - loss: 237.3839 - class: 54.3841 - reg: 125.9281 - freeSpace: 57.0717 - val_loss: 329325.0866 - mAP: 0.7976 - mAR: 0.5089 - mIoU: 0.3902

Epoch: 21/100
1558/1558 [====================] - 867s 556ms/step - loss: 234.8373 - class: 53.8175 - reg: 124.2544 - freeSpace: 56.7655 - val_loss: 539927.0647 - mAP: 0.1990 - mAR: 0.7623 - mIoU: 0.2279

Epoch: 22/100
1558/1558 [====================] - 903s 580ms/step - loss: 232.8806 - class: 53.2949 - reg: 123.0543 - freeSpace: 56.5314 - val_loss: 400038.2085 - mAP: 0.0870 - mAR: 0.8168 - mIoU: 0.3181

Epoch: 23/100
1558/1558 [====================] - 844s 542ms/step - loss: 231.2016 - class: 52.9042 - reg: 121.9266 - freeSpace: 56.3708 - val_loss: 756100.9961 - mAP: 0.8254 - mAR: 0.4879 - mIoU: 0.1520

Epoch: 24/100
1558/1558 [====================] - 866s 556ms/step - loss: 230.0096 - class: 52.3968 - reg: 121.4386 - freeSpace: 56.1742 - val_loss: 461153.2805 - mAP: 0.2123 - mAR: 0.7535 - mIoU: 0.1713

Epoch: 25/100
1558/1558 [====================] - 846s 543ms/step - loss: 229.0211 - class: 51.9380 - reg: 120.9869 - freeSpace: 56.0961 - val_loss: 277725.4017 - mAP: 0.7138 - mAR: 0.6225 - mIoU: 0.4277

Epoch: 26/100
1558/1558 [====================] - 855s 549ms/step - loss: 226.7550 - class: 51.4761 - reg: 119.5197 - freeSpace: 55.7592 - val_loss: 336919.0544 - mAP: 0.3448 - mAR: 0.7135 - mIoU: 0.3111

Epoch: 27/100
1558/1558 [====================] - 845s 543ms/step - loss: 225.8724 - class: 51.0882 - reg: 119.1394 - freeSpace: 55.6448 - val_loss: 294062.1377 - mAP: 0.8265 - mAR: 0.5796 - mIoU: 0.4410

Epoch: 28/100
1558/1558 [====================] - 845s 542ms/step - loss: 224.4340 - class: 50.6119 - reg: 118.3701 - freeSpace: 55.4521 - val_loss: 308120.5972 - mAP: 0.8612 - mAR: 0.5672 - mIoU: 0.2332

Epoch: 29/100
1558/1558 [====================] - 851s 546ms/step - loss: 223.3813 - class: 50.3892 - reg: 117.6582 - freeSpace: 55.3339 - val_loss: 244059.7998 - mAP: 0.5412 - mAR: 0.6756 - mIoU: 0.5479

Epoch: 30/100
1558/1558 [====================] - 846s 543ms/step - loss: 222.0295 - class: 49.9179 - reg: 116.9913 - freeSpace: 55.1203 - val_loss: 1522955.9608 - mAP: 0.7976 - mAR: 0.6252 - mIoU: 0.1300

Epoch: 31/100
1558/1558 [====================] - 856s 549ms/step - loss: 220.4182 - class: 49.4913 - reg: 116.1509 - freeSpace: 54.7759 - val_loss: 241107.7630 - mAP: 0.3127 - mAR: 0.7437 - mIoU: 0.5230

Epoch: 32/100
1558/1558 [====================] - 844s 542ms/step - loss: 218.4030 - class: 49.1613 - reg: 114.6291 - freeSpace: 54.6126 - val_loss: 258030.3894 - mAP: 0.8792 - mAR: 0.5352 - mIoU: 0.5326

Epoch: 33/100
1558/1558 [====================] - 863s 554ms/step - loss: 217.3870 - class: 48.6615 - reg: 114.2235 - freeSpace: 54.5020 - val_loss: 537740.9796 - mAP: 0.2222 - mAR: 0.7700 - mIoU: 0.1705

Epoch: 34/100
1558/1558 [====================] - 846s 543ms/step - loss: 216.3060 - class: 48.3359 - reg: 113.6647 - freeSpace: 54.3054 - val_loss: 367431.9746 - mAP: 0.7805 - mAR: 0.6270 - mIoU: 0.3311

Epoch: 35/100
1558/1558 [====================] - 865s 555ms/step - loss: 214.8089 - class: 47.9724 - reg: 112.6757 - freeSpace: 54.1608 - val_loss: 265571.5396 - mAP: 0.1901 - mAR: 0.7700 - mIoU: 0.4763

Epoch: 36/100
1558/1558 [====================] - 859s 552ms/step - loss: 213.4511 - class: 47.5318 - reg: 111.8290 - freeSpace: 54.0902 - val_loss: 259070.4041 - mAP: 0.2328 - mAR: 0.7496 - mIoU: 0.5250

Epoch: 37/100
1558/1558 [====================] - 884s 568ms/step - loss: 212.0300 - class: 47.1130 - reg: 111.0204 - freeSpace: 53.8966 - val_loss: 256218.6490 - mAP: 0.1073 - mAR: 0.7970 - mIoU: 0.4586

Epoch: 38/100
1558/1558 [====================] - 969s 622ms/step - loss: 210.6129 - class: 46.6356 - reg: 110.2529 - freeSpace: 53.7245 - val_loss: 356021.2858 - mAP: 0.0499 - mAR: 0.8396 - mIoU: 0.4022

Epoch: 39/100
1558/1558 [====================] - 895s 575ms/step - loss: 209.6795 - class: 46.1529 - reg: 109.8887 - freeSpace: 53.6380 - val_loss: 330427.7438 - mAP: 0.0891 - mAR: 0.8112 - mIoU: 0.4501

Epoch: 40/100
1558/1558 [====================] - 857s 550ms/step - loss: 208.4302 - class: 45.6322 - reg: 109.2583 - freeSpace: 53.5397 - val_loss: 230024.4910 - mAP: 0.2546 - mAR: 0.7559 - mIoU: 0.5682

Epoch: 41/100
1558/1558 [====================] - 878s 564ms/step - loss: 206.0642 - class: 45.0596 - reg: 107.7829 - freeSpace: 53.2217 - val_loss: 239721.9729 - mAP: 0.1144 - mAR: 0.7983 - mIoU: 0.5219

Epoch: 42/100
1558/1558 [====================] - 850s 546ms/step - loss: 204.8906 - class: 44.6748 - reg: 107.1727 - freeSpace: 53.0431 - val_loss: 237608.5408 - mAP: 0.4702 - mAR: 0.7198 - mIoU: 0.5235

Epoch: 43/100
1558/1558 [====================] - 848s 545ms/step - loss: 204.1420 - class: 44.3836 - reg: 106.7987 - freeSpace: 52.9597 - val_loss: 220468.1048 - mAP: 0.5215 - mAR: 0.7049 - mIoU: 0.5489

Epoch: 44/100
1558/1558 [====================] - 847s 544ms/step - loss: 203.0193 - class: 44.0572 - reg: 106.1527 - freeSpace: 52.8095 - val_loss: 219983.4593 - mAP: 0.5536 - mAR: 0.6971 - mIoU: 0.5812

Epoch: 45/100
1558/1558 [====================] - 879s 564ms/step - loss: 201.9879 - class: 43.6897 - reg: 105.6161 - freeSpace: 52.6821 - val_loss: 747383.0793 - mAP: 0.1087 - mAR: 0.7994 - mIoU: 0.1531

Epoch: 46/100
1558/1558 [====================] - 844s 542ms/step - loss: 201.5485 - class: 43.3882 - reg: 105.5686 - freeSpace: 52.5917 - val_loss: 240836.7110 - mAP: 0.8473 - mAR: 0.6262 - mIoU: 0.5255

Epoch: 47/100
1558/1558 [====================] - 1307s 839ms/step - loss: 201.0132 - class: 43.1146 - reg: 105.3741 - freeSpace: 52.5245 - val_loss: 478548.5906 - mAP: 0.0242 - mAR: 0.8740 - mIoU: 0.4055

Epoch: 48/100
1558/1558 [====================] - 881s 565ms/step - loss: 199.6743 - class: 42.8643 - reg: 104.4767 - freeSpace: 52.3333 - val_loss: 393425.0568 - mAP: 0.1098 - mAR: 0.8051 - mIoU: 0.2520

Epoch: 49/100
1558/1558 [====================] - 906s 582ms/step - loss: 199.0583 - class: 42.7260 - reg: 104.0971 - freeSpace: 52.2352 - val_loss: 279581.4959 - mAP: 0.0735 - mAR: 0.8163 - mIoU: 0.5342

Epoch: 50/100
1558/1558 [====================] - 867s 557ms/step - loss: 198.0182 - class: 42.4059 - reg: 103.5348 - freeSpace: 52.0775 - val_loss: 329146.2165 - mAP: 0.1451 - mAR: 0.7937 - mIoU: 0.3542

Epoch: 51/100
1558/1558 [====================] - 852s 547ms/step - loss: 196.7831 - class: 42.1005 - reg: 102.7985 - freeSpace: 51.8841 - val_loss: 247085.4632 - mAP: 0.3461 - mAR: 0.7531 - mIoU: 0.5482

Epoch: 52/100
1558/1558 [====================] - 946s 607ms/step - loss: 196.1307 - class: 41.9568 - reg: 102.3623 - freeSpace: 51.8115 - val_loss: 260115.8177 - mAP: 0.0559 - mAR: 0.8347 - mIoU: 0.6023

Epoch: 53/100
1558/1558 [====================] - 859s 551ms/step - loss: 195.1234 - class: 41.5403 - reg: 101.8793 - freeSpace: 51.7039 - val_loss: 294806.6019 - mAP: 0.2164 - mAR: 0.7821 - mIoU: 0.3902

Epoch: 54/100
1558/1558 [====================] - 845s 543ms/step - loss: 194.1955 - class: 41.3867 - reg: 101.2249 - freeSpace: 51.5839 - val_loss: 226311.3312 - mAP: 0.6677 - mAR: 0.6839 - mIoU: 0.5902

Epoch: 55/100
1558/1558 [====================] - 1301s 835ms/step - loss: 193.7301 - class: 41.2299 - reg: 100.9662 - freeSpace: 51.5340 - val_loss: 467389.7477 - mAP: 0.0232 - mAR: 0.8594 - mIoU: 0.4268

Epoch: 56/100
1558/1558 [====================] - 981s 630ms/step - loss: 193.0815 - class: 40.9908 - reg: 100.6827 - freeSpace: 51.4080 - val_loss: 271550.1070 - mAP: 0.0455 - mAR: 0.8417 - mIoU: 0.5848

Epoch: 57/100
1558/1558 [====================] - 842s 540ms/step - loss: 192.5738 - class: 40.8142 - reg: 100.4346 - freeSpace: 51.3250 - val_loss: 282261.3600 - mAP: 0.9770 - mAR: 0.4655 - mIoU: 0.4705

Epoch: 58/100
1558/1558 [====================] - 844s 541ms/step - loss: 191.8480 - class: 40.6356 - reg: 99.9508 - freeSpace: 51.2616 - val_loss: 274146.4896 - mAP: 0.8812 - mAR: 0.6507 - mIoU: 0.3734

Epoch: 59/100
1558/1558 [====================] - 850s 546ms/step - loss: 191.5690 - class: 40.4261 - reg: 99.9862 - freeSpace: 51.1568 - val_loss: 334190.7200 - mAP: 0.3765 - mAR: 0.7624 - mIoU: 0.3458

Epoch: 60/100
1558/1558 [====================] - 859s 551ms/step - loss: 190.8403 - class: 40.4370 - reg: 99.3378 - freeSpace: 51.0655 - val_loss: 254284.5554 - mAP: 0.2044 - mAR: 0.7844 - mIoU: 0.4889

Epoch: 61/100
1558/1558 [====================] - 851s 546ms/step - loss: 189.3658 - class: 39.9925 - reg: 98.5048 - freeSpace: 50.8684 - val_loss: 234642.9500 - mAP: 0.3370 - mAR: 0.7645 - mIoU: 0.4456

Epoch: 62/100
1558/1558 [====================] - 955s 613ms/step - loss: 188.6890 - class: 39.8086 - reg: 98.1689 - freeSpace: 50.7114 - val_loss: 298566.2179 - mAP: 0.0486 - mAR: 0.8325 - mIoU: 0.5174

Epoch: 63/100
1558/1558 [====================] - 951s 610ms/step - loss: 188.6460 - class: 39.7188 - reg: 98.2447 - freeSpace: 50.6825 - val_loss: 264241.6155 - mAP: 0.0509 - mAR: 0.8378 - mIoU: 0.5730

Epoch: 64/100
1558/1558 [====================] - 845s 543ms/step - loss: 187.8040 - class: 39.4183 - reg: 97.8561 - freeSpace: 50.5296 - val_loss: 218391.5520 - mAP: 0.7216 - mAR: 0.7047 - mIoU: 0.5972

Epoch: 65/100
1554/1558 [==================>.] - ETA: 2s - loss: 187.2760 - class: 39.3986 - reg: 97.3743 - freeSpace: 50.5031           
'''

'''
Epoch: 1/100
1558/1558 [====================] - 743s 477ms/step - loss: 1312.4202 - class: 1141.8717 - reg: 170.5485 - freeSpace: 276.9525 - val_loss: 546044.2356 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 2/100
1558/1558 [====================] - 720s 462ms/step - loss: 254.8801 - class: 89.9523 - reg: 164.9278 - freeSpace: 277.9352 - val_loss: 545630.6705 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 3/100
1558/1558 [====================] - 709s 455ms/step - loss: 235.3947 - class: 75.6407 - reg: 159.7541 - freeSpace: 277.4497 - val_loss: 525154.6523 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 4/100
1558/1558 [====================] - 698s 448ms/step - loss: 226.8349 - class: 71.7179 - reg: 155.1170 - freeSpace: 276.7656 - val_loss: 678457.3360 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 5/100
1558/1558 [====================] - 702s 451ms/step - loss: 219.2635 - class: 68.1832 - reg: 151.0803 - freeSpace: 275.8384 - val_loss: 537201.0853 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 6/100
1558/1558 [====================] - 700s 449ms/step - loss: 214.0288 - class: 66.0227 - reg: 148.0061 - freeSpace: 274.6801 - val_loss: 2987907.2116 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 7/100
1558/1558 [====================] - 699s 448ms/step - loss: 209.5531 - class: 64.0828 - reg: 145.4703 - freeSpace: 274.4243 - val_loss: 837557.2978 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 8/100
1558/1558 [====================] - 700s 449ms/step - loss: 204.9701 - class: 62.6050 - reg: 142.3651 - freeSpace: 275.1903 - val_loss: 563564.2008 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 9/100
1558/1558 [====================] - 702s 451ms/step - loss: 201.6025 - class: 61.3951 - reg: 140.2073 - freeSpace: 276.5483 - val_loss: 575811.4335 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 10/100
1558/1558 [====================] - 700s 449ms/step - loss: 198.3448 - class: 60.0331 - reg: 138.3118 - freeSpace: 276.9241 - val_loss: 478296.4331 - mAP: 0.0000e+00 - mAR: 0.0000e+00 - mIoU: 0.0000e+00

Epoch: 11/100
1558/1558 [====================] - 703s 451ms/step - loss: 194.6313 - class: 59.0923 - reg: 135.5391 - freeSpace: 276.4170 - val_loss: 490213.9788 - mAP: 0.8003 - mAR: 0.4191 - mIoU: 0.1013

Epoch: 12/100
1558/1558 [====================] - 922s 592ms/step - loss: 191.3082 - class: 58.1033 - reg: 133.2049 - freeSpace: 275.8802 - val_loss: 571814.0587 - mAP: 0.0359 - mAR: 0.8383 - mIoU: 0.1119

Epoch: 13/100
1558/1558 [====================] - 746s 479ms/step - loss: 189.2940 - class: 57.2977 - reg: 131.9964 - freeSpace: 275.6201 - val_loss: 476452.3754 - mAP: 0.1137 - mAR: 0.7721 - mIoU: 0.1006

Epoch: 14/100
1558/1558 [====================] - 869s 558ms/step - loss: 187.0338 - class: 56.6189 - reg: 130.4149 - freeSpace: 275.4045 - val_loss: 524395.3554 - mAP: 0.0454 - mAR: 0.8292 - mIoU: 0.0712

Epoch: 15/100
1558/1558 [====================] - 720s 462ms/step - loss: 184.9278 - class: 55.9785 - reg: 128.9493 - freeSpace: 274.9882 - val_loss: 464813.8653 - mAP: 0.3509 - mAR: 0.6719 - mIoU: 0.0871

Epoch: 16/100
1558/1558 [====================] - 800s 513ms/step - loss: 183.5730 - class: 55.3099 - reg: 128.2631 - freeSpace: 274.7479 - val_loss: 551298.3085 - mAP: 0.0657 - mAR: 0.8148 - mIoU: 0.1235

Epoch: 17/100
1558/1558 [====================] - 714s 458ms/step - loss: 181.2256 - class: 54.6330 - reg: 126.5926 - freeSpace: 275.0231 - val_loss: 580985.9906 - mAP: 0.5357 - mAR: 0.5879 - mIoU: 0.0818

Epoch: 18/100
1558/1558 [====================] - 729s 468ms/step - loss: 179.5289 - class: 54.0478 - reg: 125.4811 - freeSpace: 275.0816 - val_loss: 463562.8862 - mAP: 0.1867 - mAR: 0.7426 - mIoU: 0.1071

Epoch: 19/100
1558/1558 [====================] - 981s 630ms/step - loss: 178.0181 - class: 53.6228 - reg: 124.3954 - freeSpace: 275.4649 - val_loss: 618014.8412 - mAP: 0.0314 - mAR: 0.8535 - mIoU: 0.0572

Epoch: 20/100
1558/1558 [====================] - 815s 523ms/step - loss: 176.1288 - class: 52.8939 - reg: 123.2348 - freeSpace: 275.7758 - val_loss: 545490.6972 - mAP: 0.0536 - mAR: 0.8209 - mIoU: 0.0664

Epoch: 21/100
1558/1558 [====================] - 708s 454ms/step - loss: 174.0239 - class: 52.3040 - reg: 121.7199 - freeSpace: 275.9312 - val_loss: 516578.6116 - mAP: 0.9007 - mAR: 0.4116 - mIoU: 0.0404

Epoch: 22/100
1558/1558 [====================] - 746s 479ms/step - loss: 172.2829 - class: 51.7807 - reg: 120.5023 - freeSpace: 276.2636 - val_loss: 460844.9851 - mAP: 0.1081 - mAR: 0.7918 - mIoU: 0.0881

Epoch: 23/100
1558/1558 [====================] - 714s 458ms/step - loss: 171.2354 - class: 51.5508 - reg: 119.6845 - freeSpace: 276.4639 - val_loss: 469312.2894 - mAP: 0.6077 - mAR: 0.6261 - mIoU: 0.0864

Epoch: 24/100
1558/1558 [====================] - 718s 461ms/step - loss: 169.8299 - class: 50.9112 - reg: 118.9187 - freeSpace: 276.8960 - val_loss: 448561.2981 - mAP: 0.2485 - mAR: 0.7249 - mIoU: 0.0703

Epoch: 25/100
1558/1558 [====================] - 713s 458ms/step - loss: 168.8234 - class: 50.5954 - reg: 118.2280 - freeSpace: 277.0680 - val_loss: 453545.2751 - mAP: 0.5144 - mAR: 0.6723 - mIoU: 0.0970

Epoch: 26/100
1558/1558 [====================] - 728s 468ms/step - loss: 167.2403 - class: 50.1167 - reg: 117.1236 - freeSpace: 277.2612 - val_loss: 458936.7230 - mAP: 0.2450 - mAR: 0.7338 - mIoU: 0.1004

Epoch: 27/100
1558/1558 [====================] - 728s 467ms/step - loss: 166.1464 - class: 49.6634 - reg: 116.4831 - freeSpace: 277.4284 - val_loss: 443693.6514 - mAP: 0.2040 - mAR: 0.7477 - mIoU: 0.0730

Epoch: 28/100
1558/1558 [====================] - 717s 460ms/step - loss: 164.9816 - class: 49.3870 - reg: 115.5946 - freeSpace: 277.4045 - val_loss: 452344.8732 - mAP: 0.4806 - mAR: 0.6700 - mIoU: 0.0843

Epoch: 29/100
1558/1558 [====================] - 712s 457ms/step - loss: 163.9848 - class: 49.0522 - reg: 114.9326 - freeSpace: 277.6316 - val_loss: 443535.6803 - mAP: 0.7403 - mAR: 0.6111 - mIoU: 0.0200

Epoch: 30/100
1558/1558 [====================] - 814s 522ms/step - loss: 162.5254 - class: 48.5409 - reg: 113.9845 - freeSpace: 277.7435 - val_loss: 509744.5012 - mAP: 0.0547 - mAR: 0.8267 - mIoU: 0.0861

Epoch: 31/100
1558/1558 [====================] - 708s 454ms/step - loss: 161.3064 - class: 48.1600 - reg: 113.1464 - freeSpace: 277.1238 - val_loss: 450199.0711 - mAP: 0.7861 - mAR: 0.5670 - mIoU: 0.0633

Epoch: 32/100
1558/1558 [====================] - 712s 457ms/step - loss: 159.3027 - class: 47.6619 - reg: 111.6408 - freeSpace: 277.1398 - val_loss: 504336.8526 - mAP: 0.9118 - mAR: 0.0243 - mIoU: 0.0293

Epoch: 33/100
1558/1558 [====================] - 723s 464ms/step - loss: 158.3832 - class: 47.2678 - reg: 111.1154 - freeSpace: 277.4963 - val_loss: 437806.9856 - mAP: 0.2702 - mAR: 0.7355 - mIoU: 0.0733

Epoch: 34/100
1558/1558 [====================] - 715s 459ms/step - loss: 158.5828 - class: 47.9036 - reg: 110.6792 - freeSpace: 276.4270 - val_loss: 459071.8112 - mAP: 0.8052 - mAR: 0.6111 - mIoU: 0.0722

Epoch: 35/100
1558/1558 [====================] - 725s 465ms/step - loss: 156.1088 - class: 46.6191 - reg: 109.4897 - freeSpace: 276.7997 - val_loss: 441510.1228 - mAP: 0.2256 - mAR: 0.7526 - mIoU: 0.0961

Epoch: 36/100
1558/1558 [====================] - 715s 459ms/step - loss: 155.1942 - class: 46.2616 - reg: 108.9326 - freeSpace: 277.1616 - val_loss: 445116.7333 - mAP: 0.4251 - mAR: 0.7071 - mIoU: 0.0758

Epoch: 37/100
1558/1558 [====================] - 716s 459ms/step - loss: 154.5053 - class: 46.1923 - reg: 108.3130 - freeSpace: 277.1300 - val_loss: 442945.7054 - mAP: 0.7507 - mAR: 0.6482 - mIoU: 0.0735

Epoch: 38/100
1558/1558 [====================] - 832s 534ms/step - loss: 153.6492 - class: 45.7791 - reg: 107.8701 - freeSpace: 277.2609 - val_loss: 492274.1003 - mAP: 0.0478 - mAR: 0.8230 - mIoU: 0.0397

Epoch: 39/100
1558/1558 [====================] - 725s 466ms/step - loss: 152.5857 - class: 45.3179 - reg: 107.2678 - freeSpace: 277.4554 - val_loss: 437973.3894 - mAP: 0.2754 - mAR: 0.7313 - mIoU: 0.0802

Epoch: 40/100
1558/1558 [====================] - 727s 467ms/step - loss: 151.8795 - class: 45.1173 - reg: 106.7621 - freeSpace: 277.4834 - val_loss: 431508.0082 - mAP: 0.2140 - mAR: 0.7714 - mIoU: 0.0891

Epoch: 41/100
1558/1558 [====================] - 757s 486ms/step - loss: 150.1082 - class: 44.5794 - reg: 105.5287 - freeSpace: 277.7851 - val_loss: 456328.5793 - mAP: 0.0820 - mAR: 0.8090 - mIoU: 0.1064

Epoch: 42/100
1558/1558 [====================] - 726s 466ms/step - loss: 148.8885 - class: 44.3518 - reg: 104.5367 - freeSpace: 277.8126 - val_loss: 440170.4402 - mAP: 0.1724 - mAR: 0.7722 - mIoU: 0.0885

Epoch: 43/100
1558/1558 [====================] - 705s 453ms/step - loss: 148.5722 - class: 44.2808 - reg: 104.2914 - freeSpace: 277.9613 - val_loss: 440762.6692 - mAP: 0.7312 - mAR: 0.6542 - mIoU: 0.0828

Epoch: 44/100
1558/1558 [====================] - 716s 460ms/step - loss: 147.8730 - class: 43.8552 - reg: 104.0179 - freeSpace: 277.9444 - val_loss: 437670.4823 - mAP: 0.4332 - mAR: 0.7189 - mIoU: 0.0993

Epoch: 45/100
1558/1558 [====================] - 803s 515ms/step - loss: 146.8091 - class: 43.5537 - reg: 103.2554 - freeSpace: 277.9242 - val_loss: 488586.8172 - mAP: 0.0559 - mAR: 0.8293 - mIoU: 0.1200

Epoch: 46/100
1558/1558 [====================] - 703s 451ms/step - loss: 146.7521 - class: 43.4293 - reg: 103.3228 - freeSpace: 277.9892 - val_loss: 451648.3319 - mAP: 0.7248 - mAR: 0.6583 - mIoU: 0.0987

Epoch: 47/100
1558/1558 [====================] - 6005s 4s/step - loss: 146.4123 - class: 43.2050 - reg: 103.2074 - freeSpace: 277.9467 - val_loss: 2279511.7923 - mAP: 0.0055 - mAR: 0.9161 - mIoU: 0.1204

Epoch: 48/100
1558/1558 [====================] - 726s 466ms/step - loss: 144.9521 - class: 42.9662 - reg: 101.9859 - freeSpace: 278.0176 - val_loss: 430079.7543 - mAP: 0.2230 - mAR: 0.7553 - mIoU: 0.0885

Epoch: 49/100
1558/1558 [====================] - 785s 504ms/step - loss: 144.4991 - class: 42.6805 - reg: 101.8186 - freeSpace: 277.9243 - val_loss: 471488.2196 - mAP: 0.0622 - mAR: 0.8125 - mIoU: 0.1146

Epoch: 50/100
1558/1558 [====================] - 727s 467ms/step - loss: 144.0815 - class: 42.5662 - reg: 101.5153 - freeSpace: 277.9030 - val_loss: 432648.1976 - mAP: 0.2032 - mAR: 0.7791 - mIoU: 0.1101

Epoch: 51/100
1558/1558 [====================] - 718s 461ms/step - loss: 142.7571 - class: 42.2356 - reg: 100.5216 - freeSpace: 277.8137 - val_loss: 434517.1094 - mAP: 0.4010 - mAR: 0.7209 - mIoU: 0.0793

Epoch: 52/100
1558/1558 [====================] - 1094s 702ms/step - loss: 142.0967 - class: 41.9102 - reg: 100.1865 - freeSpace: 277.9283 - val_loss: 602057.9830 - mAP: 0.0260 - mAR: 0.8585 - mIoU: 0.1178

Epoch: 53/100
1558/1558 [====================] - 709s 455ms/step - loss: 141.1775 - class: 41.5601 - reg: 99.6174 - freeSpace: 277.7479 - val_loss: 435636.9069 - mAP: 0.5633 - mAR: 0.7095 - mIoU: 0.0977

Epoch: 54/100
1558/1558 [====================] - 740s 475ms/step - loss: 140.5232 - class: 41.4323 - reg: 99.0909 - freeSpace: 277.7848 - val_loss: 446382.0811 - mAP: 0.1126 - mAR: 0.7961 - mIoU: 0.1103

Epoch: 55/100
1558/1558 [====================] - 723s 464ms/step - loss: 140.1573 - class: 41.4104 - reg: 98.7469 - freeSpace: 277.7810 - val_loss: 431381.6837 - mAP: 0.2421 - mAR: 0.7523 - mIoU: 0.0980

Epoch: 56/100
1558/1558 [====================] - 711s 457ms/step - loss: 139.2296 - class: 40.9856 - reg: 98.2440 - freeSpace: 277.6584 - val_loss: 436844.1700 - mAP: 0.6156 - mAR: 0.7145 - mIoU: 0.1092

Epoch: 57/100
1558/1558 [====================] - 704s 452ms/step - loss: 138.9540 - class: 40.7956 - reg: 98.1584 - freeSpace: 277.4626 - val_loss: 437725.5757 - mAP: 0.7474 - mAR: 0.6761 - mIoU: 0.0703

Epoch: 58/100
1558/1558 [====================] - 708s 454ms/step - loss: 138.3469 - class: 40.6648 - reg: 97.6821 - freeSpace: 277.4897 - val_loss: 447520.2065 - mAP: 0.9033 - mAR: 0.6223 - mIoU: 0.0719

Epoch: 59/100
1558/1558 [====================] - 728s 467ms/step - loss: 138.0421 - class: 40.4991 - reg: 97.5430 - freeSpace: 277.4909 - val_loss: 439660.9108 - mAP: 0.1255 - mAR: 0.7917 - mIoU: 0.1113

Epoch: 60/100
1558/1558 [====================] - 772s 496ms/step - loss: 137.1902 - class: 40.3698 - reg: 96.8204 - freeSpace: 277.5113 - val_loss: 457352.3074 - mAP: 0.0757 - mAR: 0.8217 - mIoU: 0.1012

Epoch: 61/100
1558/1558 [====================] - 712s 457ms/step - loss: 136.2160 - class: 40.0680 - reg: 96.1480 - freeSpace: 277.2183 - val_loss: 428281.1885 - mAP: 0.3203 - mAR: 0.7413 - mIoU: 0.0798

Epoch: 62/100
1558/1558 [====================] - 755s 484ms/step - loss: 135.8508 - class: 39.7689 - reg: 96.0819 - freeSpace: 277.1330 - val_loss: 455139.1478 - mAP: 0.0882 - mAR: 0.8040 - mIoU: 0.1125

Epoch: 63/100
1558/1558 [====================] - 773s 496ms/step - loss: 135.5850 - class: 39.6898 - reg: 95.8952 - freeSpace: 277.1761 - val_loss: 459466.7050 - mAP: 0.0644 - mAR: 0.8115 - mIoU: 0.1051

Epoch: 64/100
1558/1558 [====================] - 710s 456ms/step - loss: 135.0814 - class: 39.4543 - reg: 95.6271 - freeSpace: 277.0905 - val_loss: 434653.5054 - mAP: 0.6835 - mAR: 0.7081 - mIoU: 0.0804

Epoch: 65/100
1558/1558 [====================] - 4600s 3s/step - loss: 134.3768 - class: 39.3435 - reg: 95.0333 - freeSpace: 277.0491 - val_loss: 1645214.7356 - mAP: 0.0060 - mAR: 0.9029 - mIoU: 0.1189

Epoch: 66/100
1558/1558 [====================] - 724s 465ms/step - loss: 133.9989 - class: 39.1193 - reg: 94.8795 - freeSpace: 277.0285 - val_loss: 430385.9233 - mAP: 0.3868 - mAR: 0.7409 - mIoU: 0.0906

Epoch: 67/100
1558/1558 [====================] - 709s 455ms/step - loss: 133.9146 - class: 39.0263 - reg: 94.8883 - freeSpace: 277.1123 - val_loss: 438029.2495 - mAP: 0.7817 - mAR: 0.6899 - mIoU: 0.0716

Epoch: 68/100
1558/1558 [====================] - 700s 449ms/step - loss: 133.3007 - class: 38.7453 - reg: 94.5553 - freeSpace: 276.8729 - val_loss: 511478.5938 - mAP: 1.0000 - mAR: 0.0306 - mIoU: 0.0655

Epoch: 69/100
1558/1558 [====================] - 788s 506ms/step - loss: 133.0144 - class: 38.7122 - reg: 94.3023 - freeSpace: 276.8999 - val_loss: 476191.7303 - mAP: 0.0620 - mAR: 0.8291 - mIoU: 0.1139

Epoch: 70/100
1558/1558 [====================] - 704s 452ms/step - loss: 132.6730 - class: 38.5738 - reg: 94.0992 - freeSpace: 276.9607 - val_loss: 440251.3841 - mAP: 0.7991 - mAR: 0.6695 - mIoU: 0.0752

Epoch: 71/100
1558/1558 [====================] - 1205s 774ms/step - loss: 131.7122 - class: 38.2839 - reg: 93.4284 - freeSpace: 276.8368 - val_loss: 661696.7411 - mAP: 0.0215 - mAR: 0.8610 - mIoU: 0.1149

Epoch: 72/100
1558/1558 [====================] - 758s 486ms/step - loss: 131.0501 - class: 38.0814 - reg: 92.9687 - freeSpace: 276.8891 - val_loss: 450959.8361 - mAP: 0.0868 - mAR: 0.8119 - mIoU: 0.1055

Epoch: 73/100
1558/1558 [====================] - 792s 509ms/step - loss: 130.6366 - class: 37.9005 - reg: 92.7361 - freeSpace: 276.8265 - val_loss: 460279.5709 - mAP: 0.0626 - mAR: 0.8222 - mIoU: 0.1025

Epoch: 74/100
1558/1558 [====================] - 835s 536ms/step - loss: 130.1476 - class: 37.7649 - reg: 92.3827 - freeSpace: 276.8248 - val_loss: 474492.0122 - mAP: 0.0431 - mAR: 0.8414 - mIoU: 0.0981

Epoch: 75/100
1558/1558 [====================] - 715s 459ms/step - loss: 130.5959 - class: 37.6352 - reg: 92.9608 - freeSpace: 276.8381 - val_loss: 423965.7621 - mAP: 0.2882 - mAR: 0.7562 - mIoU: 0.0989

Epoch: 76/100
1558/1558 [====================] - 942s 605ms/step - loss: 129.5322 - class: 37.5371 - reg: 91.9951 - freeSpace: 276.7990 - val_loss: 543984.9441 - mAP: 0.0316 - mAR: 0.8555 - mIoU: 0.1142

Epoch: 77/100
1558/1558 [====================] - 726s 466ms/step - loss: 129.1927 - class: 37.4234 - reg: 91.7693 - freeSpace: 276.8639 - val_loss: 427846.0986 - mAP: 0.2471 - mAR: 0.7654 - mIoU: 0.1063

Epoch: 78/100
1558/1558 [====================] - 1041s 668ms/step - loss: 128.5842 - class: 37.2474 - reg: 91.3368 - freeSpace: 276.9096 - val_loss: 572768.5601 - mAP: 0.0271 - mAR: 0.8696 - mIoU: 0.1114

Epoch: 79/100
1558/1558 [====================] - 701s 450ms/step - loss: 127.9260 - class: 37.1103 - reg: 90.8156 - freeSpace: 276.7894 - val_loss: 459345.8276 - mAP: 0.9509 - mAR: 0.5768 - mIoU: 0.0639

Epoch: 80/100
1558/1558 [====================] - 706s 453ms/step - loss: 127.8415 - class: 36.9530 - reg: 90.8885 - freeSpace: 276.8095 - val_loss: 435383.1773 - mAP: 0.8272 - mAR: 0.6826 - mIoU: 0.0585

Epoch: 81/100
1558/1558 [====================] - 716s 459ms/step - loss: 127.2075 - class: 36.7120 - reg: 90.4956 - freeSpace: 276.5552 - val_loss: 426886.8786 - mAP: 0.3225 - mAR: 0.7556 - mIoU: 0.0944

Epoch: 82/100
1558/1558 [====================] - 717s 460ms/step - loss: 126.3109 - class: 36.6031 - reg: 89.7078 - freeSpace: 276.4754 - val_loss: 429673.5243 - mAP: 0.5005 - mAR: 0.7310 - mIoU: 0.0713

Epoch: 83/100
1558/1558 [====================] - 716s 459ms/step - loss: 126.4124 - class: 36.5404 - reg: 89.8720 - freeSpace: 276.3539 - val_loss: 429990.7604 - mAP: 0.5280 - mAR: 0.7280 - mIoU: 0.0831

Epoch: 84/100
1558/1558 [====================] - 718s 461ms/step - loss: 126.0321 - class: 36.3143 - reg: 89.7178 - freeSpace: 276.3354 - val_loss: 428984.0363 - mAP: 0.2818 - mAR: 0.7636 - mIoU: 0.0822

Epoch: 85/100
1558/1558 [====================] - 710s 456ms/step - loss: 125.7004 - class: 36.2698 - reg: 89.4306 - freeSpace: 276.4243 - val_loss: 426912.5517 - mAP: 0.3958 - mAR: 0.7490 - mIoU: 0.1028

Epoch: 86/100
1558/1558 [====================] - 776s 498ms/step - loss: 125.7114 - class: 36.1668 - reg: 89.5446 - freeSpace: 276.3493 - val_loss: 460594.5502 - mAP: 0.0562 - mAR: 0.8308 - mIoU: 0.1027

Epoch: 87/100
1558/1558 [====================] - 723s 464ms/step - loss: 124.5685 - class: 36.1010 - reg: 88.4675 - freeSpace: 276.2343 - val_loss: 423089.9035 - mAP: 0.2611 - mAR: 0.7578 - mIoU: 0.0879

Epoch: 88/100
1558/1558 [====================] - 716s 460ms/step - loss: 124.7689 - class: 35.9681 - reg: 88.8008 - freeSpace: 276.0980 - val_loss: 423755.1652 - mAP: 0.4734 - mAR: 0.7409 - mIoU: 0.0757

Epoch: 89/100
1558/1558 [====================] - 715s 459ms/step - loss: 124.5075 - class: 35.8259 - reg: 88.6815 - freeSpace: 276.0638 - val_loss: 428036.6181 - mAP: 0.5448 - mAR: 0.7356 - mIoU: 0.0873

Epoch: 90/100
1558/1558 [====================] - 718s 461ms/step - loss: 124.2879 - class: 35.7127 - reg: 88.5753 - freeSpace: 276.0572 - val_loss: 422825.6386 - mAP: 0.2434 - mAR: 0.7706 - mIoU: 0.0775

Epoch: 91/100
1558/1558 [====================] - 718s 461ms/step - loss: 122.8171 - class: 35.3716 - reg: 87.4455 - freeSpace: 276.0073 - val_loss: 425615.9135 - mAP: 0.2989 - mAR: 0.7653 - mIoU: 0.0981

Epoch: 92/100
1558/1558 [====================] - 717s 460ms/step - loss: 123.2972 - class: 35.3407 - reg: 87.9564 - freeSpace: 275.9147 - val_loss: 426804.5771 - mAP: 0.5213 - mAR: 0.7405 - mIoU: 0.0970

Epoch: 93/100
1558/1558 [====================] - 714s 458ms/step - loss: 122.5348 - class: 35.3375 - reg: 87.1973 - freeSpace: 275.7538 - val_loss: 427688.7949 - mAP: 0.7019 - mAR: 0.7173 - mIoU: 0.0758

Epoch: 94/100
1558/1558 [====================] - 715s 459ms/step - loss: 121.8357 - class: 35.1978 - reg: 86.6378 - freeSpace: 275.8783 - val_loss: 429665.7015 - mAP: 0.5960 - mAR: 0.7214 - mIoU: 0.0979

Epoch: 95/100
1558/1558 [====================] - 727s 467ms/step - loss: 122.0393 - class: 35.0964 - reg: 86.9429 - freeSpace: 275.7384 - val_loss: 423081.2930 - mAP: 0.1542 - mAR: 0.7902 - mIoU: 0.0998

Epoch: 96/100
1558/1558 [====================] - 703s 451ms/step - loss: 121.7523 - class: 35.0288 - reg: 86.7235 - freeSpace: 275.6019 - val_loss: 429438.9431 - mAP: 0.6089 - mAR: 0.7256 - mIoU: 0.0877

Epoch: 97/100
1558/1558 [====================] - 710s 456ms/step - loss: 121.0122 - class: 34.8606 - reg: 86.1515 - freeSpace: 275.5633 - val_loss: 424986.9330 - mAP: 0.4211 - mAR: 0.7510 - mIoU: 0.0956

Epoch: 98/100
1558/1558 [====================] - 747s 479ms/step - loss: 121.2789 - class: 34.7986 - reg: 86.4803 - freeSpace: 275.5960 - val_loss: 452210.9605 - mAP: 0.0823 - mAR: 0.8243 - mIoU: 0.1066

Epoch: 99/100
1558/1558 [====================] - 713s 457ms/step - loss: 121.0815 - class: 34.7344 - reg: 86.3471 - freeSpace: 275.6283 - val_loss: 419965.9721 - mAP: 0.2206 - mAR: 0.7778 - mIoU: 0.0848

Epoch: 100/100
1558/1558 [====================] - 716s 460ms/step - loss: 120.3990 - class: 34.6150 - reg: 85.7840 - freeSpace: 275.6476 - val_loss: 426254.1489 - mAP: 0.1918 - mAR: 0.7814 - mIoU: 0.1063
'''