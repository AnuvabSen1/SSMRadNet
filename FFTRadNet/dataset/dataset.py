from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt

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

class RADIal(Dataset):

    def __init__(self, root_dir,statistics=None,encoder=None,difficult=False, load_images=False):

        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        self.load_images = load_images
        
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
       
        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]


        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())
        

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 


        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()      

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_ADC',"adc_{:06d}.npy".format(sample_id))
        input = np.load(radar_name,allow_pickle=True)
        radar_FFT = convert_radar_cube(input)


        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the camera image
        # img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        if self.load_images:
            img_name = os.path.join(self.root_dir,'image',"img_{:06d}.png".format(sample_id))
            image = np.asarray(Image.open(img_name))

            return radar_FFT, segmap,out_label,box_labels,image
        
        else:
            return radar_FFT, segmap,out_label,box_labels,None
    

class RADIalEvents(Dataset):
    def __init__(self, root_dir, events_map_file='radeve_dataset.csv', statistics=None, encoder=None, difficult=False):
        
        self.root_dir = root_dir
        self.statistics = statistics
        self.encoder = encoder
        
        # 1. Load Original Labels (Ground Truth Boxes)
        self.labels = pd.read_csv(os.path.join(root_dir, 'labels.csv')).to_numpy()
       
        # Filter difficult samples
        if difficult == False:
            ids_filters = []
            ids = np.where(self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather input entries by sample id
        self.unique_ids = np.unique(self.labels[:, 0])
        self.label_dict = {}
        for i, ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:, 0] == ids)[0]
            self.label_dict[ids] = sample_ids
        self.sample_keys = list(self.label_dict.keys())
        
        # 2. Load Event Mapping (Created in previous steps)
        # We need to map the integer 'sample_id' (e.g., 123) to the event filename
        map_path = os.path.join(root_dir, events_map_file)
        if not os.path.exists(map_path):
            # Fallback checks if user didn't put it in root_dir
            if os.path.exists(events_map_file):
                map_path = events_map_file
            else:
                raise FileNotFoundError(f"Event mapping file not found at {map_path}")

        df_map = pd.read_csv(map_path)
        
        # Build a dictionary: int(sample_id) -> event_file_path
        # The csv has 'label_image' column like "img_000000.png"
        self.event_map = {}
        for idx, row in df_map.iterrows():
            # Extract ID from "img_000123.png" -> 123
            img_name = row['label_image']
            try:
                # Remove 'img_' prefix and '.png' suffix
                s_id = int(img_name.replace('img_', '').replace('.png', ''))
                self.event_map[s_id] = row['event_path']
            except ValueError:
                continue

        # Standard Transforms for Segmentation Maps
        self.resize = Resize((256, 224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512, 448))

        # Event Sensor Dimensions
        self.height = 260
        self.width = 346
        self.num_bins = 10 # Desired number of temporal subframes

    def events_to_voxel_grid(self, events):
        """
        Accumulates events into a (T, H, W) voxel grid.
        events: np.array of shape (N, 4) -> [t, x, y, p]
        """
        voxel_grid = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)

        if len(events) == 0:
            return voxel_grid

        # 1. Normalize timestamps to range [0, num_bins]
        t = events[:, 0]
        t_start = t[0]
        t_end = t[-1]
        
        # Avoid division by zero if all events define a single point in time
        duration = t_end - t_start
        if duration == 0:
            duration = 1e-6
            
        # Calculate bin indices: floor( (t - t_start) / duration * num_bins )
        t_normalized = (t - t_start) / duration * self.num_bins
        t_idx = np.floor(t_normalized).astype(np.int64)
        
        # Handle edge case: the very last timestamp maps to exactly 'num_bins', 
        # which is out of bounds (0 to 9). Clip it to 9.
        t_idx = np.clip(t_idx, 0, self.num_bins - 1)

        # 2. Get Coordinates
        x_idx = events[:, 1].astype(np.int64)
        y_idx = events[:, 2].astype(np.int64)
        p_vals = events[:, 3].astype(np.float32) # Polarity (-1 or 1, or 0 or 1)

        # Ensure coordinates are within bounds (just in case)
        x_idx = np.clip(x_idx, 0, self.width - 1)
        y_idx = np.clip(y_idx, 0, self.height - 1)

        # 3. Accumulate (Vectorized)
        # We assume polarity p is important. If p is 0/1, we might want to map 0->-1
        # If the input events are from v2e, p might be 0 and 1.
        # Let's map 0 to -1 so they don't just disappear if we sum them.
        # If your p is already -1/1, this line can be removed.
        if p_vals.min() >= 0: 
            p_vals[p_vals == 0] = -1

        # np.add.at allows unbuffered accumulation (handling collisions correctly)
        np.add.at(voxel_grid, (t_idx, y_idx, x_idx), p_vals)

        return voxel_grid

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # --- 1. LABELS ---
        entries_indexes = self.label_dict[sample_id]
        box_labels = self.labels[entries_indexes]
        
        # Format: [Range, Angle, Doppler, laser_X, laser_Y, laser_Z, x1, y1, x2, y2]
        box_labels = box_labels[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32) 

        out_label = []
        if self.encoder is not None:
            out_label = self.encoder(box_labels).copy()      

        # --- 2. EVENT DATA (Replaces Radar) ---
        # Retrieve path from our mapping dict
        if sample_id in self.event_map:
            event_path = self.event_map[sample_id]
            try:
                # Load npz
                data = np.load(event_path)
                # Stack to (N, 4) -> t, x, y, p
                events = np.stack((data['t'], data['x'], data['y'], data['p']), axis=1)
            except Exception as e:
                # print(f"Error loading {event_path}: {e}")
                events = np.zeros((0, 4))
        else:
            # print(f"Warning: No event map found for sample {sample_id}")
            events = np.zeros((0, 4))
            
        # Process into (10, H, W)
        event_voxel = self.events_to_voxel_grid(events)

        # --- 3. SEGMENTATION MAP ---
        segmap_name = os.path.join(self.root_dir, 'radar_Freespace', "freespace_{:06d}.png".format(sample_id))
        if os.path.exists(segmap_name):
            segmap = Image.open(segmap_name)
            segmap = self.crop(segmap)
            segmap = np.asarray(self.resize(segmap)) == 255
        else:
            # Fallback if missing
            segmap = np.zeros((256, 224), dtype=bool)

        # --- 4. CAMERA IMAGE ---
        img_name = os.path.join(self.root_dir, 'image', "img_{:06d}.png".format(sample_id))
        if os.path.exists(img_name):
            image = np.asarray(Image.open(img_name))
        else:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Return: Events (Tensor), Segmap, Boxes, Boxes (Raw), Image
        return event_voxel, segmap, out_label, box_labels, image
    

    

# --- 2. VISUALIZATION UTILS ---
def render_voxel_channel(channel_data):
    """
    Render a single time-bin of the voxel grid as an RGB image.
    channel_data: (H, W) array with accumulated polarities.
    """
    H, W = channel_data.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Positive sums -> Green
    pos_mask = channel_data > 0
    # Negative sums -> Red
    neg_mask = channel_data < 0
    
    # We can scale intensity based on the count, but for checking correctness,
    # simple binary visibility is often enough. 
    # Let's cap at 255 for high activity.
    
    # Normalize for visualization: abs(val) * scale
    scale = 50 # Adjust brightness sensitivity
    
    img[pos_mask, 1] = np.clip(channel_data[pos_mask] * scale, 0, 255).astype(np.uint8)
    img[neg_mask, 0] = np.clip(np.abs(channel_data[neg_mask]) * scale, 0, 255).astype(np.uint8)
    
    return img

def main_verify():
    # CONFIG
    # Point this to your RADIal/ready_to_use folder
    ROOT_DIR = os.path.expanduser('~/RADIal/ready_to_use') 
    
    print(f"Initializing Dataset from {ROOT_DIR}...")
    try:
        dataset = RADIalEvents(root_dir=ROOT_DIR, events_map_file='radeve_dataset.csv')
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        return

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("Dataset is empty. Check CSV paths.")
        return

    # Pick 3 random samples
    indices = random.sample(range(len(dataset)), 5)
    
    for idx in indices:
        print(f"\nLoading Sample Index: {idx} (Sample ID: {dataset.sample_keys[idx]})")
        
        # Load item
        event_voxel, segmap, label, box_labels, image = dataset[idx]
        
        print(f"  Image Shape: {image.shape}")
        print(f"  Voxel Grid Shape: {event_voxel.shape}")
        
        # Setup Plot
        # 1 Row for Image, 2 Rows for 10 Event Frames (5 per row)
        fig = plt.figure(figsize=(15, 8))
        
        # Plot Original Image
        ax = plt.subplot(3, 5, 3) # Centered in top row
        ax.imshow(image)
        ax.set_title("Camera Image")
        ax.axis('off')
        
        # Plot 10 Event Frames
        for t in range(10):
            # Calculate subplot index: start at 6 (row 2, col 1)
            # t=0 -> 6, t=4 -> 10, t=5 -> 11
            plot_idx = t + 6 
            
            ax_ev = plt.subplot(3, 5, plot_idx)
            
            # Extract channel t
            channel = event_voxel[t]
            vis_frame = render_voxel_channel(channel)
            
            ax_ev.imshow(vis_frame)
            ax_ev.set_title(f"Event Bin {t}")
            ax_ev.axis('off')
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main_verify()


