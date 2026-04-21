import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from .ssm_fourier import LearnableFFTSSM
import json


NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

import torchprofile

########################################## FFTNet #######################################

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
    def __init__(self,complex_channels=32, mode=None):
        super(FFT_Net,self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = NoShift_Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.mode = mode
        #self.activation = ComplexAct(act=nn.LeakyReLU(),use_phase=True)

    def forward(self,x):
        out = {}
        x = x.permute(0,1,3,2)
        x_range = self.range_net(x)
        x = self.cplx_transpose(2,3)(x_range)
        x = self.doppler_net(x)
        x = torch.concat([x.real,x.imag],axis=1)
        out['doppler_fft'] = x
        if self.mode == "debug":
            out['range_fft'] = torch.concat([x_range.real,x_range.imag],axis=1).transpose(2,3)
        return out
    
########################################## FSSM_Net #####################################

class FSSM_Net(nn.Module):
    def __init__(self, complex_channels=32, no_samples=512, no_chirps=256, mode=None):
        super(FSSM_Net, self).__init__()
        self.complex_channels = complex_channels
        self.no_samples = no_samples
        self.no_chirps = no_chirps
        self.mode = mode

        # LearnableFFTSSM: input shape (batch, time_series, channels) --> output shape (batch, channels/2, fourier length) complex
        self.range_net = LearnableFFTSSM( 
            d_in=complex_channels,
            n_modes=self.no_samples//2+1,   # standard positive freqs (no Nyquist)
            window_len=self.no_samples,
            exact_fft_init=True,
            spectrum="full",    # calculate whole spectrum or just positive freqs
            in_type="complex",  # "real"
            out_type="complex", # "mag"
            log_mag=False,
            keep_nyquist=True
        )
        self.doppler_net = LearnableFFTSSM(
            d_in=complex_channels,
            n_modes=self.no_chirps//2+1,   # standard positive freqs (no Nyquist)
            window_len=self.no_chirps,
            exact_fft_init=True,
            spectrum="full",    # calculate whole spectrum or just positive freqs
            in_type="complex",  # "real"
            out_type="complex", # "mag"
            log_mag=False,
            keep_nyquist=True
        )

    def forward(self, x):

        out = {}
        B, N, S, C = x.shape
        # we have an input of size (B, 2N, S, C) batch, channels, samples, chirps
        x = x.permute(0, 3, 2, 1) # reshape to batch, chirps, samples, channels
        x = x.reshape(B*C, S, N) # (B* C, S, 2N)

        range_fft = self.range_net(x)   # (B*C, N, Sf) complex array
        range_fft = torch.cat([range_fft.real, range_fft.imag], dim=1) # (B*C, 2N, Sf) real array
        range_fft = range_fft.transpose(1, 2) # (B*C, Sf, 2N) real array 

        range_fft = range_fft.reshape(B, C, S, N) # (batch, chirps, range, channels)
        range_fft_debug = range_fft.permute(0, 2, 1, 3) # (batch, range, chirps, channels)

        range_fft = range_fft_debug.reshape(B*S, C, N)   # (batch*range, chirps, channels)

        doppler_fft = self.doppler_net(range_fft)  # (batch*range, channels, doppler) complex
        doppler_fft = torch.cat([doppler_fft.real, doppler_fft.imag], dim=1) # (batch*range, channels, doppler) real

        doppler_fft = doppler_fft.reshape(B, S, N, C) # (batch, range, channels, doppler)
        doppler_fft = doppler_fft.permute(0, 2, 1, 3) # (batch, channels, range, doppler)

        out['doppler_fft'] = doppler_fft
        if self.mode == "debug":
            out['range_fft'] = range_fft_debug.permute(0, 3, 1, 2)
        return out


#########################################################################################


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

class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer,out_layer,kernel_size=(1,12),dilation=(1,16),use_bn = False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size, 
                              stride=(1, 1), padding=0,dilation=dilation, bias= (not use_bn) )
     
        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self,x):
        width = x.shape[-1]
        x = torch.cat([x[...,-self.padding:],x,x[...,:self.padding]],axis=3)
        x = self.conv(x)
        x = x[...,int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
        return x

class FPN_BackBone(nn.Module):

    def __init__(self, num_block,channels,block_expansion,mimo_layer,use_bn=True):
        super(FPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32,mimo_layer,
                                        kernel_size=(1,NbTxAntenna),
                                        dilation=(1,NbRxAntenna),
                                        use_bn = True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])
                                       
    def forward(self, x):

        x_expanded = self.pre_enc(x)
        x = self.conv(x_expanded)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features, x_expanded


    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)

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

class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()
        
        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        
        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L3  = nn.Conv2d(192, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(160, 224, kernel_size=1, stride=1,padding=0)
        
        
    def forward(self,features):

        T4 = features['x4'].transpose(1, 3) 
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        
        return out


class FFTRadNet(nn.Module):
    def __init__(self,mimo_layer,channels,blocks,regression_layer = 2, detection_head=True,segmentation_head=True, fourier_encoder = None, mode=None):
        super(FFTRadNet, self).__init__()
    
        self.fourier_encoder = fourier_encoder
        self.mode = mode

        if self.fourier_encoder == "FourierNet":
            self.fftNet = FFT_Net(mode=self.mode)
        elif self.fourier_encoder == "FSSM":
            self.fftNet = FSSM_Net(mode=self.mode)
        else:
            self.fftNet = None

        self.norm = nn.InstanceNorm2d(32)

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        self.RA_decoder = RangeAngle_Decoder()

        # self.unet = UNet(n_channels=32, n_classes=256, bilinear=True)
        
        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)

        if(self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

    def forward(self,x):
                       
        out = {'Detection':[],'Segmentation':[]}

        # x = x.permute(0, 3, 1, 2) # [B,32,512,256]
        
        # features= self.FPN(x)
        # RA = self.RA_decoder(features)
        if self.fourier_encoder == "FourierNet":
            real = x[:, :16, :, :]    # [B,16,512,256]
            imag = x[:, 16:, :, :]    # [B,16,512,256]
            
            # 2) pack into complex64
            x_cplx = torch.complex(real, imag)  
            x_cplx = x_cplx.to(torch.complex64)            

            fft_out = self.fftNet(x_cplx)
            complex_features = fft_out['doppler_fft']
            complex_features = self.norm(complex_features)
        elif self.fourier_encoder == "FSSM":
            fft_out = self.fftNet(x)
            complex_features = fft_out['doppler_fft']
            complex_features = self.norm(complex_features)
        else:
            complex_features = x

        # features = self.unet(complex_features)

        features, RA_expanded = self.FPN(complex_features)
        features = self.RA_decoder(features)

        debug_features = {}
        if self.mode == "debug":
            debug_features['doppler_fft'] = fft_out['doppler_fft']
            debug_features['range_fft'] = fft_out['range_fft']
            debug_features['RA_map'] = RA_expanded
        # [1, 256, 128, 224])
        # print(features.shape)

        if(self.detection_head):
            out['Detection'] = self.detection_header(features)

        if(self.segmentation_head):
            Y =  F.interpolate(features, (256, 224))
            out['Segmentation'] = self.freespace(Y)
        
        if self.mode == "debug":
            return out, debug_features
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
    # radar_frame_norm = radar_frame / (2**12) # np.std(radar_frame_mag)

    frame_ri = np.concatenate([radar_frame.real, radar_frame.imag], axis=2).astype(np.float32)

    return frame_ri
    

def copy_shared_weights(src: nn.Module, dst: nn.Module, exclude_prefix=("fftNet",)):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k, v in src_sd.items():
        # skip anything under excluded prefixes (e.g., fftNet.*)
        if any(k.startswith(p + ".") for p in exclude_prefix):
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k].copy_(v)

# usage

def diff_report(a, b, prefix=""):
    diffs = []
    for (ka, va), (kb, vb) in zip(a.named_parameters(), b.named_parameters()):
        if any(ka.startswith(p + ".") for p in ("fftNet",)) or ka != kb: 
            continue
        if not torch.allclose(va, vb):
            diffs.append(ka)
    return diffs


if __name__ == "__main__":

    data = convert_radar_cube(np.load("../../ready_to_use/radar_ADC/adc_000000.npy")).astype(np.float32)
    input = torch.tensor(data).unsqueeze(0).to("cuda")
    input = input.permute(0, 3, 1, 2) #[:, :2, :, :]

    config = json.load(open("../config/config_FFTRadNet_192_56.json"))


    # net0 = FFTRadNet(blocks = config['model']['backbone_block'],
    #                     mimo_layer  = config['model']['MIMO_output'],
    #                      channels = config['model']['channels'], 
    #                      regression_layer = 2, 
    #                      detection_head = config['model']['DetectionHead'], 
    #                      segmentation_head = config['model']['SegmentationHead'], 
    #                      fourier_encoder=None).to("cuda").float()
    
    # net0 = torch.nn.DataParallel(net0)
    

    net1 = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                         channels = config['model']['channels'], 
                         regression_layer = 2, 
                         detection_head = config['model']['DetectionHead'], 
                         segmentation_head = config['model']['SegmentationHead'], 
                         fourier_encoder="FourierNet",
                         mode='debug').to("cuda").float()
    
    net1 = torch.nn.DataParallel(net1)

    checkpoint1 = torch.load("FFTNet_FFTRadNet.pth", weights_only=False)
    net1.load_state_dict(checkpoint1['net_state_dict'])

    quit() 
    
    net2 = FFTRadNet(blocks = config['model']['backbone_block'],
                        mimo_layer  = config['model']['MIMO_output'],
                         channels = config['model']['channels'], 
                         regression_layer = 2, 
                         detection_head = config['model']['DetectionHead'], 
                         segmentation_head = config['model']['SegmentationHead'], 
                         fourier_encoder="FSSM",
                         mode='debug').to("cuda").float()
    
    net2 = torch.nn.DataParallel(net2)

    checkpoint2 = torch.load("FSSM_FFTRadNet.pth", weights_only=False)
    net2.load_state_dict(checkpoint2['net_state_dict'])
    
    quit()
    # copy_shared_weights(net1, net2)
    # print("Diffs after copy:", diff_report(net1, net2))

    # x = torch.randn(1, 32, 512, 256)
    y1, features1 = net1(input)
    y2, features2 = net2(input)


    # detection_difference = torch.linalg.norm(y1['Segmentation'] - y2['Segmentation'])
    # print(detection_difference)

    # print(y1['Detection'].shape) # torch.Size([1, 3, 128, 224])
    # print(y1['Segmentation'].shape) # torch.Size([1, 1, 256, 224])

    # from fvcore.nn import FlopCountAnalysis

    # flops = FlopCountAnalysis(net0, input)
    # print(f"Basic FFTRadNet GFLOPs: {flops.total()/1e9}")

    # flops = FlopCountAnalysis(net1, input)
    # print(f"FourierNet FFTRadnet GFLOPs: {flops.total()/1e9}")

    # flops = FlopCountAnalysis(net2, input)
    # print(f"FSSM-FFTRadNet GFLOPs: {flops.total()/1e9}")

    ######################## Simple Testing FSSM Encoder block ########################

    # model1 = FSSM_Net().to("cuda")
    # model1.eval()

    # input = (2*torch.rand(16, 32, 512, 256)-1).to("cuda") # 8 batch works, 16 doesnt

    # with torch.no_grad():
    #     output_FSSM = model1(input).cpu().numpy().squeeze()
    
    # print(output_FSSM.shape)


    ######################## Testing module for comparing FourierNet and FSSM ########################


    # print(input.shape)

    # chan = 32

    # torch.manual_seed(3)

    # # input = (2*torch.rand(1, chan, 512, 256)-1).to("cuda")


    # model1 = FSSM_Net(complex_channels=chan).to("cuda")
    # model1.eval()

    # with torch.no_grad():
    #     output_FSSM = model1(input).cpu().numpy().squeeze()
    
    # print(output_FSSM.shape)

    # model2 = FFT_Net(complex_channels=chan).to("cuda")
    # model2.eval()

    # # 2) pack into complex64
    # x_cplx = torch.complex(input[:, :chan//2, :, :], input[:, chan//2:, :, :])  
    # x_cplx = x_cplx.to(torch.complex64)            

    # with torch.no_grad():
    #     output_fourierNet = model2(x_cplx).cpu().numpy().squeeze()

    # print(output_fourierNet.shape)

    # print(np.linalg.norm(output_fourierNet - output_FSSM))



    


    