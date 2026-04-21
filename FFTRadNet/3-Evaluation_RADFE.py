import os
import json
import argparse
import torch
import random
import numpy as np
from model.RADFe_FFTRadNet_Det2 import FFTRadNet
from model.RADFe import RADFE
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

    net = FFTRadNet()

    net.to('cuda')

    #path = "/home/sayeed/RADIal/experiments/RADFE_third___Aug-18-2025___02:13:30/RADFE_third_epoch99_loss_151210.6569_AP_0.7637_AR_0.7343_IOU_0.6339.pth"
    path = "/home/sayeed/RADIal/experiments/RADFE_third___Aug-18-2025___02:13:30/RADFE_third_epoch97_loss_149945.1543_AP_0.8036_AR_0.7312_IOU_0.6323.pth"
    
    print('===========  Loading the model ==================:')
    dict = torch.load(path, weights_only=False)
    # print(dict['net_state_dict'].keys())
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(net,test_loader,enc, iou_threshold=0.5)

   
        
        

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config/config_RADFE_third.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult)

