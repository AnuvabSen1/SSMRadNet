import os
import json
import argparse
import torch
import numpy as np
from model.FFTRadNet_ViT import FFTRadNet_ViT_ADC
from model.TFFTRadNet import FSSM_TFFTRadNet
from model.FourierNet_FFTRadNet import FFTRadNet
# from model.UNet import FFTRadNet
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
import cv2
import time
# from utils.util import DisplayHMI

from ptflops import get_model_complexity_info

def main(config, checkpoint_filename,difficult):

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)
    
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=difficult)


    # Create the model
    net = FSSM_TFFTRadNet(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])
    
    
    net.to('cuda')

    # Load the model
    # dict = torch.load(checkpoint_filename)
    # net.load_state_dict(dict['net_state_dict'])
    net.eval()


    input_size = (32, 512, 256)

    macs, params = get_model_complexity_info(net, input_size, as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"MACs  : {macs}, Params: {params}")


    inputs = torch.rand(1, 32, 512, 256).to('cuda')

    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(net, inputs)
    print(f"FSSM-TFFTRadNet GFLOPs: {flops.total()/1e9}")


    latencies = []

    with torch.no_grad():
        for i in range(10):
            # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
            
            _ = net(inputs)

        for i in range(200):
            # inputs = torch.randn(1, 32, 512, 256).to('cuda')

            torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = net(inputs)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append(t1-t0)


        
    avg_latency_s = sum(latencies) / len(latencies)
    print(f"Avg model latency: {avg_latency_s*1000:.2f} ms ")

    ###################################### Memory Test ###############################################
    # input = torch.randn(1, 32, 512, 256, device='cuda')

    # 3) Reset peak stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()  # optional, to clear out any cached blocks

    # 4) Warm‐up (optional, to allocate any lazily‐created CUDA kernels)
    with torch.no_grad():
        _ = net(inputs)

    # 5) Measure
    torch.cuda.synchronize()   # wait for all kernels to finish
    with torch.no_grad():
        _ = net(inputs)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()   # bytes
    current_alloc = torch.cuda.memory_allocated()    # bytes
    reserved     = torch.cuda.memory_reserved()      # bytes

    print(f"Peak alloc:  {peak_alloc/1024**2:.1f} MB")
    print(f"Current alloc: {current_alloc/1024**2:.1f} MB")

    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='config/config_FFTRadNet_192_56.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open("config/ADC_config.json"))
    
    main(config, args.checkpoint,args.difficult)
