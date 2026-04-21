import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar
import os
import matplotlib.pyplot as plt

def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        seg_map_label = data[2].to('cuda').double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None and segmentation_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs['Detection'], label_map,losses_params)           
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()        
            loss_seg = segmentation_loss(prediction, label)
            loss_seg *= inputs.size(0)
                

            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            loss_seg *=losses_params['weight'][2]


            loss = classif_loss + reg_loss + loss_seg

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[3]

            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()

            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels,label_freespace):

                metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                            threshold=0.2,range_min=5,range_max=100) 
                


        kbar.update(i)
        

    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}


def run_FullEvaluation(net,loader,encoder,iou_threshold=0.1):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        
        labels_object = data[3]
        label_freespace = data[2].numpy().copy()
            
        for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
            
            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)
                

        kbar.update(i)
        
    GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_threshold)

    mIoU = []
    for i in range(len(predictions['prediction']['freespace'])):
        # 0 to 124 means 0 to 50m
        pred = predictions['prediction']['freespace'][i][:124].reshape(-1)>=0.5
        label = predictions['label']['freespace'][i][:124].reshape(-1)
        
        intersection = np.abs(pred*label).sum()
        union = np.sum(label) + np.sum(pred) -intersection
        iou = intersection /union
        mIoU.append(iou)


    mIoU = np.asarray(mIoU).mean()
    print('------- Freespace Scores ------------')
    print('  mIoU',mIoU*100,'%')


def run_FullEvaluation_sweeping(net,loader,encoder,iou_threshold=0.1):

    net.eval()

    chirps = np.arange(32, 256+1, 16)
    # chirps = np.arange(256, 64-1, -32)

    score_values = {'chirps':[],
                    'mAP': [],
                    'mAR': [],
                    'F1': [],
                    'RE': [],
                    'AE': [],
                    'mIoU': []}
    
    for chirp in chirps:
        kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

        print(f'Generating Predictions for {chirp} chirps')
        predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
        for i, data in enumerate(loader):

            # input, out_label,segmap,labels
            inputs = data[0].to('cuda').float()

            with torch.set_grad_enabled(False):
                outputs = net(inputs[:, :, :chirp, :])

            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            
            labels_object = data[3]
            label_freespace = data[2].numpy().copy()
                
            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
                
                predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
                predictions['label']['objects'].append(true_obj)

                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)
                    

            kbar.update(i)
            
        detection_scores = GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_threshold, verbose=True)

        mIoU = []
        for i in range(len(predictions['prediction']['freespace'])):
            # 0 to 124 means 0 to 50m
            pred = predictions['prediction']['freespace'][i][:124].reshape(-1)>=0.5
            label = predictions['label']['freespace'][i][:124].reshape(-1)
            
            intersection = np.abs(pred*label).sum()
            union = np.sum(label) + np.sum(pred) -intersection
            iou = intersection /union
            mIoU.append(iou)


        mIoU = np.asarray(mIoU).mean()
        print('------- Freespace Scores ------------')
        print('  mIoU',mIoU*100,'%')

        score_values['chirps'].append(chirp)
        score_values['mAP'].append(detection_scores['mAP'])
        score_values['mAR'].append(detection_scores['mAR'])
        score_values['F1'].append(detection_scores['F1'])
        score_values['RE'].append(detection_scores['RE'])
        score_values['AE'].append(detection_scores['AE'])
        score_values['mIoU'].append(mIoU)

    save_dir = "outputs"
    
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    def _one_plot(y, ylabel, fname=None):
        plt.figure()
        plt.plot(score_values['chirps'], y, marker='o')
        plt.xlabel('Number of Chirps')
        plt.ylabel(ylabel)
        plt.grid(True)
        if save_dir and fname:
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight')

    _one_plot(score_values['mAP'],        'mAP',         'mAP_vs_chirps.png')
    _one_plot(score_values['mAR'],        'mAR',         'mAR_vs_chirps.png')
    _one_plot(score_values['F1'],         'F1 Score',    'F1_vs_chirps.png')
    _one_plot(score_values['RE'], 'Range Error (m)', 'RangeError_vs_chirps.png')
    _one_plot(score_values['AE'], 'Angle Error (deg)', 'AngleError_vs_chirps.png')
    _one_plot(score_values['mIoU'],       'Freespace mIoU', 'mIoU_vs_chirps.png')

    # show at the end so all figures render together if running interactively
    plt.show()

    import pickle
    with open('values.pkl', 'wb') as file:
        pickle.dump(score_values, file)

def plot_miou_density(sweep_results, nbins=50, normalize=False, overlay_mean=True, save_path=None):
    """
    Creates a 2D density (heatmap) of freespace IoU vs. chirp count.
    - x-axis: chirp counts
    - y-axis: IoU bins [0, 1]
    - color: number of frames falling in each (chirp, IoU-bin) cell
    Set normalize=True to show per-chirp probability density instead of raw counts.
    """
    chirps = np.asarray(sweep_results['chirps'])
    miou_all = sweep_results['mIoU_all']   # list of np.array per chirp

    # Build 2D histogram: rows=IoU bins, cols=chirp indices
    bins = np.linspace(0.0, 1.0, nbins + 1)
    H = np.zeros((nbins, len(chirps)), dtype=float)

    for j, vals in enumerate(miou_all):
        # vals: IoU per frame for chirp j
        counts, _ = np.histogram(vals, bins=bins)
        if normalize:
            s = counts.sum()
            if s > 0:
                counts = counts / s
        H[:, j] = counts

    # Plot heatmap
    # Compute x extent so each chirp column is centered on its value
    if len(chirps) > 1:
        dx = np.min(np.diff(chirps))
    else:
        dx = max(1, chirps[0] * 0.05)
    extent = [chirps[0] - dx/2, chirps[-1] + dx/2, 0.0, 1.0]

    plt.figure()
    im = plt.imshow(H, origin='lower', aspect='auto', extent=extent)
    cbar = plt.colorbar(im)
    cbar.set_label('Frame count' if not normalize else 'Density')

    plt.xlabel('Number of Chirps')
    plt.ylabel('Freespace IoU')

    # Optional: overlay mean IoU per chirp
    if overlay_mean:
        means = [float(np.mean(v)) if len(v) else np.nan for v in miou_all]
        plt.plot(chirps, means, marker='o', linewidth=1.5, label='Mean IoU')
        plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def run_FullEvaluation_sweeping_mIoU(net, loader, encoder, iou_threshold=0.1,
                                chirp_start=32, chirp_stop=256, chirp_step=16,
                                plot=True, save_dir=None):
    """
    Sweeps chirp counts, collects metrics.
    Now also stores per-frame freespace IoUs for each chirp, and plots mean ± std.
    """
    net.eval()

    chirps = np.arange(chirp_start, chirp_stop + 1, chirp_step)

    # storage for sweep results
    sweep_results = {
        'chirps': [],
        'mAP': [],
        'mAR': [],
        'F1': [],
        'RangeError': [],
        'AngleError': [],
        # freespace IoU stats
        'mIoU_mean': [],
        'mIoU_std': [],
        'mIoU_all': []   # list of np.array, one per chirp
    }

    for chirp in chirps:
        print(f"Chirp count {chirp} ")
        kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

        predictions = {'prediction': {'objects': [], 'freespace': []},
                       'label': {'objects': [], 'freespace': []}}

        for i, data in enumerate(loader):
            inputs = data[0].to('cuda').float()

            with torch.no_grad():
                outputs = net(inputs[:, :, :chirp, :])

            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()

            labels_object = data[3]
            label_freespace = data[2].numpy().copy()

            for pred_obj, pred_map, true_obj, true_map in zip(out_obj, out_seg, labels_object, label_freespace):
                predictions['prediction']['objects'].append(np.asarray(encoder.decode(pred_obj, 0.05)))
                predictions['label']['objects'].append(true_obj)

                predictions['prediction']['freespace'].append(pred_map[0])  # take channel 0
                predictions['label']['freespace'].append(true_map)

            kbar.update(i)

        # ---- freespace IoUs (keep ALL per-frame values) ----
        iou_vals = []
        for i in range(len(predictions['prediction']['freespace'])):
            # 0..124 means 0..50 m
            pred = predictions['prediction']['freespace'][i][:124].reshape(-1) >= 0.5
            label = predictions['label']['freespace'][i][:124].reshape(-1)

            intersection = np.abs(pred * label).sum()
            union = np.sum(label) + np.sum(pred) - intersection
            iou = (intersection / union) if union > 0 else 0.0
            iou_vals.append(iou)

        iou_vals = np.asarray(iou_vals, dtype=float)
        iou_mean = float(np.nanmean(iou_vals)) if iou_vals.size else 0.0
        iou_std  = float(np.nanstd(iou_vals))  if iou_vals.size else 0.0

        # ---- collect sweep results ----
        sweep_results['chirps'].append(int(chirp))

        sweep_results['mIoU_all'].append(iou_vals)     # store all IoUs (per-frame)
        sweep_results['mIoU_mean'].append(iou_mean)
        sweep_results['mIoU_std'].append(iou_std)


    plot_miou_density(sweep_results, nbins=50, normalize=False, overlay_mean=True,
                  save_path=os.path.join(save_dir, 'mIoU_density_vs_chirps.png') if save_dir else None)

    # ---- Plot once: detection metrics (as before) ----
    # if plot:
    #     os.makedirs(save_dir, exist_ok=True) if save_dir else None

    #     # ---- NEW: freespace mIoU mean ± std band ----
    #     x = np.asarray(sweep_results['chirps'])
    #     mu = np.asarray(sweep_results['mIoU_mean'])
    #     sd = np.asarray(sweep_results['mIoU_std'])
    #     lower = np.clip(mu - sd, 0.0, 1.0)
    #     upper = np.clip(mu + sd, 0.0, 1.0)

    #     plt.figure()
    #     plt.plot(x, mu, marker='o', label='mIoU (mean)')
    #     plt.fill_between(x, lower, upper, alpha=0.25, label='±1 std')
    #     plt.xlabel('Number of Chirps')
    #     plt.ylabel('Freespace mIoU')
    #     plt.grid(True)
    #     plt.legend()
    #     if save_dir:
    #         plt.savefig(os.path.join(save_dir, 'mIoU_mean_std_vs_chirps.png'), bbox_inches='tight')

    #     plt.show()

    return sweep_results