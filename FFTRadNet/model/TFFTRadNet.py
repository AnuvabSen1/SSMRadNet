import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from functools import partial
from .Swin import SwinTransformer
import numpy as np
import json

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx, CplxAngle
from cplxmodule.nn.modules.casting import TensorToCplx,CplxToTensor
from cplxmodule.nn import RealToCplx, CplxToReal
import torch.nn as nn
import torch.nn.functional as F

class Range_Fourier_Net(nn.Module):
    def __init__(self):
        super(Range_Fourier_Net, self).__init__()
        self.range_nn = CplxLinear(512, 512, bias = False)
        range_weights = np.zeros((512, 512), dtype = np.complex64)
        for j in range(0, 512):
            for h in range(0, 512):
                range_weights[h][j] = np.exp(-1j * 2 * np.pi *(j*h/512))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        x = self.range_nn(x)
        return x

class Doppler_Fourier_Net(nn.Module):
    def __init__(self):
        super(Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(256, 256, bias = False)
        doppler_weights = np.zeros((256, 256), dtype=np.complex64)
        for j in range(0, 256):
            for h in range(0, 256):
                hh = h + 128
                if hh >= 256:
                    hh = hh - 256
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * hh /256))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x


class NoShift_Doppler_Fourier_Net(nn.Module):
    def __init__(self):
        super(NoShift_Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(256, 256, bias = False)
        doppler_weights = np.zeros((256, 256), dtype=np.complex64)
        for j in range(0, 256):
            for h in range(0, 256):
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * h /256))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x

class ComplexAct(nn.Module):
    def __init__(self, act, use_phase=False):
        super(ComplexAct,self).__init__()
        # act can be either a function from nn.functional or a nn.Module if the
        # activation has learnable parameters
        self.act = act
        self.use_phase = use_phase
        self.mod = CplxModulus()
        self.angle = CplxAngle()
        self.m = nn.Parameter(torch.tensor(0.0).float())
        self.m.requires_grad = True

    def forward(self, z):
        if self.use_phase:
            return self.act(self.mod(z) + self.m) * torch.exp(1.j * self.angle(z))
        else:
            return self.act(z.real) + 1.j * self.act(z.imag)

class FFT_Net(nn.Module):
    def __init__(self,):
        super(FFT_Net,self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = NoShift_Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.norm = nn.InstanceNorm2d(32)
        #self.activation = ComplexAct(act=nn.LeakyReLU(),use_phase=True)

    def forward(self,x):
        x = x.permute(0,1,3,2)
        x = self.range_net(x)
        x = self.cplx_transpose(2,3)(x)
        x = self.doppler_net(x)
        x = torch.concat([x.real,x.imag],axis=1)
        out = self.norm(x)
        return out

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

    
class Detection_Header(nn.Module):

    def __init__(self, use_bn=True,reg_layer=2,input_angle_size=0):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return

        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out

class block(nn.Module):
    def __init__(self,in_chans,out_chans,stride,kernel_size):
        super(block,self).__init__()
        self.deconv = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=kernel_size,stride=stride,padding=0)
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(out_chans)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bnorm(x)
        x = self.activation(x)

        return x

class ViT_RangeAngle_Decoder(nn.Module):
    def __init__(self):
        super(ViT_RangeAngle_Decoder,self).__init__()
        self.blk1 = block(2,32,2,2)
        self.blk2 = block(32,64,2,2)
        self.blk3 = block(64,256,2,2)
        self.dense = nn.Linear(512,896)
        self.dense_acc = nn.ReLU()

    def forward(self,x):
        x = self.dense(x)
        x = self.dense_acc(x).reshape(-1,2,16,28)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)

        return x

class Swin_RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(Swin_RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))

        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)
        self.L4  = nn.Conv2d(384*1,224,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 224, kernel_size=1, stride=1,padding=0)


    def forward(self,features):

        T4 = self.L4(features[3]).transpose(1, 3)
        T3 = self.L3(features[2]).transpose(1, 3)
        T2 = self.L2(features[1]).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        return out


class FSSM_TFFTRadNet(nn.Module):
    def __init__(self,patch_size,channels,in_chans,embed_dim,depths,num_heads,drop_rates,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FSSM_TFFTRadNet, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.DFT = FFT_Net()
        self.vit = SwinTransformer(
                 pretrain_img_size=None,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=drop_rates[0],
                 attn_drop_rate=drop_rates[1],
                 drop_path_rate=drop_rates[2],
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False)

        self.RA_decoder = Swin_RangeAngle_Decoder()

        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)

        if(self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

    def forward(self,x):

        out = {'Detection':[],'Segmentation':[]}
        # x = x.permute(0, 3, 1, 2)

        real = x[:, :16, :, :]    # [B,16,512,256]
        imag = x[:, 16:, :, :]    # [B,16,512,256]
        
        # 2) pack into complex64
        x_cplx = torch.complex(real, imag)  
        x_cplx = x_cplx.to(torch.complex64)            

        x = self.DFT(x_cplx)
        
        features = self.vit(x)
        RA = self.RA_decoder(features)
        if(self.detection_head):
            out['Detection'] = self.detection_header(RA)

        if(self.segmentation_head):
            Y =  F.interpolate(RA, (256, 224))
            out['Segmentation'] = self.freespace(Y)

        return out

   
def convert_radar_cube(sample):

    numSamplePerChirp = 512
    numRxPerChip = 4
    numChirps = 256
    numRxAnt = 16
    numTxAnt = 12
    numReducedDoppler = 16
    numChirpsPerLoop = 16
    
    adc0 = sample[:, 0]
    adc1 = sample[:, 1]
    adc2 = sample[:, 2]
    adc3 = sample[:, 3]

    frame0 = np.reshape(adc0[0::2] + 1j*adc0[1::2], (numSamplePerChirp,numRxPerChip, numChirps), order ='F').transpose((0,2,1))   
    frame1 = np.reshape(adc1[0::2] + 1j*adc1[1::2], (numSamplePerChirp,numRxPerChip, numChirps), order ='F').transpose((0,2,1))   
    frame2 = np.reshape(adc2[0::2] + 1j*adc2[1::2], (numSamplePerChirp,numRxPerChip, numChirps), order ='F').transpose((0,2,1))   
    frame3 = np.reshape(adc3[0::2] + 1j*adc3[1::2], (numSamplePerChirp,numRxPerChip, numChirps), order ='F').transpose((0,2,1))   
    radar_frame = np.concatenate([frame3,frame0,frame1,frame2],axis=2)

    # radar_frame_mag = np.abs(radar_frame)
    radar_frame_norm = radar_frame / (2**12) # np.std(radar_frame_mag)

    frame_ri = np.concatenate([radar_frame_norm.real, radar_frame_norm.imag], axis=2).astype(np.float32)

    return frame_ri
    


if __name__ == "__main__":

    data = convert_radar_cube(np.load("../../ready_to_use/radar_ADC/adc_000000.npy")).astype(np.float32)
    input = torch.tensor(data).unsqueeze(0).to("cuda")
    input = input.permute(0, 3, 1, 2) #[:, :2, :, :]

    config = json.load(open("../config/ADC_config.json"))


    net = FSSM_TFFTRadNet(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead']).float().to("cuda")
    
    net.eval()

    with torch.no_grad():
        output = net(input)

    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(net, input)
    print(f"FSSM-TFFTRadNet GFLOPs: {flops.total()/1e9}")
