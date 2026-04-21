import os
import json
import argparse
import torch
import random
import numpy as np
from model.raven_fixed_attn import RADFE
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation import run_FullEvaluation
import torch.nn as nn



def main(config, checkpoint,difficult):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

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

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])


    # Create the model
    # net = FFTRadNet(blocks = config['model']['backbone_block'],
    #                     mimo_layer  = config['model']['MIMO_output'],
    #                     channels = config['model']['channels'], 
    #                     regression_layer = 2, 
    #                     detection_head = config['model']['DetectionHead'], 
    #                     segmentation_head = config['model']['SegmentationHead'])

    net = RADFE()

    net.to('cuda')

    checkpoint = "weight/Raven_attn_fixed.pth"
    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint, weights_only=False)
    # print(dict['net_state_dict'].keys())
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(net,test_loader,enc, iou_threshold=0.001)

   
        
        

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='RAVEN Evaluation')
    parser.add_argument('-c', '--config', default='config/config_Raven_attn.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult)

