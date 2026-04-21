import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.Unetpp_RA import RaTok as RADFE
# from model.RADFe import RADFE
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation import run_evaluation, run_FullEvaluation
import torch.nn as nn

import torch
from collections import OrderedDict

def load_state_dict_flexible(model, ckpt_path, state_key="net_state_dict", map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # get raw state dict
    if isinstance(ckpt, dict) and state_key in ckpt:
        sd = ckpt[state_key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # sometimes the checkpoint itself is already a state dict-like mapping
        # (heuristic: keys look like layer names)
        sd = ckpt
    else:
        sd = ckpt

    # strip 'module.' if present
    stripped = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        stripped[k] = v

    # filter by key + shape
    model_sd = model.state_dict()
    filtered = {k: v for k, v in stripped.items()
                if k in model_sd and hasattr(v, "size") and v.size() == model_sd[k].size()}

    print(f"[CKPT] Loading {len(filtered)}/{len(model_sd)} tensors from {ckpt_path}")
    missing = [k for k in model_sd.keys() if k not in filtered]
    if missing:
        print(f"[CKPT] Missing (showing up to 10): {missing[:10]}")

    model_sd.update(filtered)
    model.load_state_dict(model_sd)
    return model

# ===================== REPRO / SNAPSHOT LOGGING =====================
def write_repro_log(exp_dir: Path, config: dict, model_cls, resume_path=None):
    """
    Writes a single log file into exp_dir that contains:
      1) the config (pretty JSON)
      2) the full model source file that defines `model_cls` (RADFE)
      3) the full training script source (this file)

    Also drops separate snapshot files for convenience.
    """
    import sys
    import inspect
    import platform
    from datetime import datetime

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # --- locate source files ---
    model_file = Path(inspect.getfile(model_cls))  # e.g., model/RaEvTok.py
    train_file = Path(__file__) if "__file__" in globals() else Path(sys.argv[0])

    # --- read contents safely ---
    def _safe_read(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(errors="ignore")

    model_src = _safe_read(model_file)
    train_src = _safe_read(train_file)

    # --- write separate snapshots (optional but handy) ---
    (exp_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (exp_dir / f"model_snapshot_{model_file.name}").write_text(model_src, encoding="utf-8")
    (exp_dir / f"train_snapshot_{train_file.name}").write_text(train_src, encoding="utf-8")

    # --- write ONE combined log that "pastes" everything ---
    log_path = exp_dir / "RUN_REPRO_LOG.txt"
    header = [
        f"Timestamp: {datetime.now().isoformat()}",
        f"CWD: {os.getcwd()}",
        f"Train file: {str(train_file)}",
        f"Model file: {str(model_file)}",
        f"Resume ckpt: {resume_path}",
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"Torch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        header += [
            f"CUDA device: {torch.cuda.get_device_name(0)}",
            f"CUDA capability: {torch.cuda.get_device_capability(0)}",
            f"CUDA version (torch): {torch.version.cuda}",
            f"cuDNN: {torch.backends.cudnn.version()}",
        ]

    with log_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("CONFIG (pretty JSON)\n")
        f.write("=" * 90 + "\n")
        f.write(json.dumps(config, indent=2, sort_keys=True))
        f.write("\n\n" + "=" * 90 + "\n")
        f.write(f"MODEL SOURCE: {model_file}\n")
        f.write("=" * 90 + "\n")
        f.write(model_src)
        f.write("\n\n" + "=" * 90 + "\n")
        f.write(f"TRAIN SOURCE: {train_file}\n")
        f.write("=" * 90 + "\n")
        f.write(train_src)

# ---- Part B: CALL SITE ----
# Put this inside main(), immediately AFTER you create (output_folder/exp_name)
# and write config.json (right after your current json.dump(config, outfile) block).
#
# Example:
#   exp_dir = output_folder / exp_name
#   ... mkdir ...
#   ... write config.json ...
#   write_repro_log(exp_dir, config, RADFE, resume_path=resume)
# ===================== END REPRO / SNAPSHOT LOGGING =====================

def main(config, resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)
    exp_dir = output_folder / exp_name
    write_repro_log(exp_dir, config, RADFE, resume_path=resume)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)
    
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=True)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])


    # # Create the model
    # net = FFTRadNet(blocks = config['model']['backbone_block'],
    #                     mimo_layer  = config['model']['MIMO_output'],
    #                     channels = config['model']['channels'], 
    #                     regression_layer = 2, 
    #                     detection_head = config['model']['DetectionHead'], 
    #                     segmentation_head = config['model']['SegmentationHead'])

    net = RADFE().cuda()
    # net = load_state_dict_flexible(
    #     net,
    #     "/home/sayeed/RADIal/FFTRadNet/weight/Raven_attn_fixed_noearly.pth",
    #     state_key="net_state_dict",
    # )


    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    best_mAP = 0

    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')


    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume, weights_only=False)
        net.load_state_dict(dict['net_state_dict'])
        # optimizer.load_state_dict(dict['optimizer'])
        # scheduler.load_state_dict(dict['scheduler'])
        # startEpoch = dict['epoch']+1
        # history = dict['history']
        # global_step = dict['global_step']

        # print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch,num_epochs):
        
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        
        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            inputs = data[0].to('cuda').float()
            # print(inputs.shape)
            label_map = data[1].to('cuda').float()
            if(config['model']['SegmentationHead']=='True'):
                seg_map_label = data[2].to('cuda').double()

            # reset the gradient
            optimizer.zero_grad()
            
            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)


            classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map,config['losses'])           
               
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()        
            loss_seg = freespace_loss(prediction, label)
            loss_seg *= inputs.size(0)

            classif_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]
            loss_seg *=config['losses']['weight'][2]


            loss = classif_loss + reg_loss + loss_seg

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
            writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        
            kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("freeSpace", loss_seg.item())])
            # kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item())])

            
            global_step += 1


        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        
        ######################
        ## validation phase ##
        ######################

        eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=0),
                                detection_loss=pixor_loss,segmentation_loss=freespace_loss,
                                losses_params=config['losses'])

        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])
        history['mIoU'].append(eval['mIoU'])

        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])

        # run_FullEvaluation(net,val_loader,enc)


        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/test', eval['loss'], global_step)
        writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
        writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
        writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)
          
        print('')

        
        

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ChirpNet4 Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open("config/config_RaEvTok.json"))
    
    main(config, args.resume)
