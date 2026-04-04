# Standard library
import glob
import math
import os
import random

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torchvision.models as models
from torchvision import transforms
from torchvision.io import read_image

from PIL import Image
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader, Sampler

from ptflops import get_model_complexity_info


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
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=[32],
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

        self.chirp_feature_pooling = nn.AdaptiveAvgPool1d(1)

        self.slow_linears = nn.ModuleList()
        self.slow_norms = nn.ModuleList()

        last_dim = self.fast_time_dim
        for next_dim in slow_time_linear_dims:
            self.slow_linears.append(nn.Linear(last_dim, next_dim))
            self.slow_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim
        
        self.slow_time_dim = last_dim
        self.slow_time_len = slow_time_len
        
        self.slow_time_positional_encoding = nn.Parameter(torch.zeros(1, self.slow_time_len, self.slow_time_dim))

        self.slow_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.slow_time_dim, d_state=32, d_conv=4, expand=2)
            for _ in range(slow_time_layers)
        ])

        # Decoder: Merge skip connections from encoder and multiple encoder stages if needed.
        # Here we simply merge the final encoder output (skip) with the output of the sequential model.
        # self.decoder_linear = nn.Linear(self.slow_time_len * 2 * self.channels, self.slow_time_dim*self.slow_time_dim)

        ####################################################### Segmentation Head #######################################################
        self.project = nn.Conv1d(in_channels=2*self.slow_time_dim, out_channels=self.slow_time_dim*self.slow_time_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')
        self.bn0 = nn.BatchNorm2d(conv_channels)

        self.upsample1 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.bn1_1 = nn.BatchNorm2d(conv_channels)
        self.bn1_2 = nn.BatchNorm2d(conv_channels//2)


        self.upsample2 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.conv2_2 = nn.Conv2d(conv_channels//2, 1, kernel_size=3, padding='same')  # Output layer
        self.bn2_1 = nn.BatchNorm2d(conv_channels//2)
        self.bn2_2 = nn.BatchNorm2d(1)


        self.upsample3 = nn.Upsample(size=(256, 224), mode='bilinear', align_corners=False)


        ####################################################### Detection Head #######################################################
        self.projectD = nn.Conv1d(in_channels=2*self.slow_time_dim, out_channels=self.slow_time_dim*self.slow_time_dim, kernel_size=1)
        self.poolD = nn.AdaptiveAvgPool1d(1)

        self.conv0D = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')
        self.bn0D = nn.BatchNorm2d(conv_channels)

        self.upsample1D = nn.Upsample(size=(48, 64), mode='bilinear', align_corners=False)
        self.conv1_1D = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')  # Output layer
        self.conv1_2D = nn.Conv2d(conv_channels, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.bn1_1D = nn.BatchNorm2d(conv_channels)
        self.bn1_2D = nn.BatchNorm2d(conv_channels//2)

        self.upsample2D = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv2_1D = nn.Conv2d(conv_channels//2, conv_channels//2, kernel_size=3, padding='same')  # Output layer
        self.conv2_2D = nn.Conv2d(conv_channels//2, 3, kernel_size=3, padding='same')  # Output layer
        self.bn2_1D = nn.BatchNorm2d(conv_channels//2)
        self.bn2_2D = nn.BatchNorm2d(3)

        self.upsample3D = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv3_cls = nn.Conv2d(3, 1, kernel_size=3, padding='same')  # Output layer class
        self.conv3_reg = nn.Conv2d(3, 2, kernel_size=3, padding='same')  # Output layer regression

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
        x_fast = x_fast + self.fast_time_positional_encoding

        for layer in self.fast_ssm_layers:
            x_fast = layer(x_fast)

        # x_chirp = x_fast[:, -1, :].reshape(B, self.slow_time_len, C) # taking only the last state output for each chirp
        # x_chirp = self.chirp_feature_conv(x_fast).squeeze()
        x_chirp = self.chirp_feature_pooling(x_fast.transpose(1, 2)).squeeze()
        x_chirp = x_chirp.reshape(B, self.slow_time_len, C)

        x_s = x_chirp

        # for linear, norm in zip(self.slow_linears, self.slow_norms):
        #     x_s = F.silu(linear(x_s))
        #     x_s = norm(x_s)

        x_slow = x_s + self.slow_time_positional_encoding

        for layer in self.slow_ssm_layers:
            x_slow = layer(x_slow)
        
        # slow ssm output dimension = B, slow_time_len, num_channels

        return x_slow, x_s

    def decode(self, x, skip):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, slow_time_len, slow_time_dim*2)
        
        x_spatial_features = x.permute(0, 2, 1)
        x_spatial_features = self.project(x_spatial_features)
        x_spatial_features = self.pool(x_spatial_features)
        x_spatial_features = x_spatial_features.squeeze(-1).reshape(B, 1, C, C)
        # x_flattened_features = x.reshape(B, S*2*C)
        # x_spatial_features = F.relu(self.decoder_linear(x_flattened_features)) 
        # x_spatial_features = x_spatial_features.reshape(B, 1,  C, C)
        

        x_out = F.silu(self.bn0(self.conv0(x_spatial_features)))

        x_out = self.upsample1(x_out)
        x_out = F.silu(self.bn1_1(self.conv1_1(x_out)))
        x_out = F.silu(self.bn1_2(self.conv1_2(x_out)))

        x_out = self.upsample2(x_out)
        x_out = F.silu(self.bn2_1(self.conv2_1(x_out)))
        # x_out = F.silu(self.bn2_2(self.conv2_2(x_out)))
        x_out = F.relu(self.conv2_2(x_out))

        x_out = self.upsample3(x_out)

        return x_out
    
    def detect(self, x, skip):
        # Combine the sequence output with the skip connection.
        B, S, C = x.shape[0], x.shape[1], x.shape[2]

        x = torch.cat([x, skip], dim=-1)  # Shape: (batch, slow_time_len, slow_time_dim*2)
        
        x_spatial_features = x.permute(0, 2, 1)
        x_spatial_features = self.projectD(x_spatial_features)
        x_spatial_features = self.poolD(x_spatial_features)
        x_spatial_features = x_spatial_features.squeeze(-1).reshape(B, 1, C, C)
        # x_flattened_features = x.reshape(B, S*2*C)
        # x_spatial_features = F.relu(self.decoder_linear(x_flattened_features)) 
        # x_spatial_features = x_spatial_features.reshape(B, 1,  C, C)
        

        x_out = F.silu(self.bn0D(self.conv0D(x_spatial_features)))

        x_out = self.upsample1D(x_out)
        x_out = F.silu(self.bn1_1D(self.conv1_1D(x_out)))
        x_out = F.silu(self.bn1_2D(self.conv1_2D(x_out)))

        x_out = self.upsample2D(x_out)
        x_out = F.silu(self.bn2_1D(self.conv2_1D(x_out)))
        x_out = F.silu(self.bn2_2D(self.conv2_2D(x_out)))

        x_out = self.upsample3D(x_out)
        x_out_cls = torch.sigmoid(self.conv3_cls(x_out))
        x_out_reg = self.conv3_reg(x_out)

        x_cls_reg = torch.cat([x_out_cls, x_out_reg], dim=1)
        # x_out = self.conv3_1D(x_out)

        return x_cls_reg

    def forward(self, x):

        out = {'Detection':[],'Segmentation':[]}


        x_encoded, skip = self.encode(x)
        out['Segmentation'] = self.decode(x_encoded, skip)
        out['Detection'] = self.detect(x_encoded, skip)
        return out

    # forward_w_enc_attn
