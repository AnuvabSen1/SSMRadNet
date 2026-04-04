import os
import json
import argparse
import torch
import numpy as np
from model.raven import RADFE
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
import cv2
from utils.util import DisplayHMI

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
    # net = FFTRadNet(blocks = config['model']['backbone_block'],
    #                     mimo_layer  = config['model']['MIMO_output'],
    #                     channels = config['model']['channels'], 
    #                     regression_layer = 2, 
    #                     detection_head = config['model']['DetectionHead'], 
    #                     segmentation_head = config['model']['SegmentationHead'])

    net = RADFE()

    net.to('cuda')

    # Load the model
    checkpoint_filename = "weight/Raven.pth"
    dict = torch.load(checkpoint_filename, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()


    for data in dataset:
        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.tensor(data[0]).to('cuda').float().unsqueeze(0)

        # print(inputs.shape)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        
        # input_shape = (512, 256, 32)

        # macs, params = get_model_complexity_info(net, input_shape , as_strings=True, print_per_layer_stat=False, verbose=False)

        # print(f"FFTRadNet MACs  : {macs}, FFTRadNet Params: {params}")

        # break

        try:
            hmi = DisplayHMI(data[4], data[0],outputs,enc)

            cv2.imshow('FFTRadNet',hmi)
        
        # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except:
            None

    cv2.destroyAllWindows()
        

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='config/config_Raven_attn.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult)
