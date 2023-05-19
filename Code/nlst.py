import json
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pytorch_lightning as pl
from typing import Optional
import numpy as np
import nibabel as nib
import torchio as tio

def image_norm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


# Ref: https://github.com/MDL-UzL/L2R/blob/main/examples/task_specific/NLST/Example_NLST.ipynb
class NLST(torch.utils.data.Dataset):
    def __init__(self, root_dir, json_conf='NLST_dataset.json', masked=False, 
                 downsampled=False, train_transform = None, train=True, is_norm=False):
       
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir,'imagesTr')
        self.keypoint_dir = os.path.join(root_dir,'keypointsTr')
        self.masked = masked
        with open(os.path.join(root_dir,json_conf)) as f:
            self.dataset_json = json.load(f)
        self.shape = self.dataset_json['tensorImageShape']['0']
        self.H, self.W, self.D = self.shape
        self.downsampled = downsampled
        self.train = train
        
        self.is_norm = is_norm
        if self.train :
            self.type_data = 'training_paired_images'
        
        else:
            self.type_data = 'registration_val'
        
    def __len__(self):
        
        if self.train:
            return self.dataset_json['numPairedTraining']
        else:
            return len(self.dataset_json['registration_val'])

    def get_shape(self):
        if self.downsampled:
            return [x//2 for x in self.shape]
        else:
            return self.shape
    
    def __getitem__(self, idx):
        fix_idx = self.dataset_json[self.type_data][idx]['fixed']
        mov_idx = self.dataset_json[self.type_data][idx]['moving']
        
        fix_path=os.path.join(self.root_dir,fix_idx)        
        mov_path=os.path.join(self.root_dir,mov_idx)            
        

        fixed_img = nib.load(fix_path)
        fixed_affine = fixed_img.affine
        fixed_img = fixed_img.get_fdata()
        
        moving_img = nib.load(mov_path).get_fdata()
        
        if self.is_norm:
            fixed_img = image_norm(fixed_img)
            moving_img = image_norm(moving_img)
        
        fixed_img=torch.from_numpy(fixed_img).float()
        moving_img=torch.from_numpy(moving_img).float()
        fixed_img = fixed_img.unsqueeze(0)
        moving_img = moving_img.unsqueeze(0)
        fixed_mask=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).float()
        fixed_mask = fixed_mask.unsqueeze(0)
        moving_mask=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).float()
        moving_mask = moving_mask.unsqueeze(0)
                 
        if self.masked:            
            fixed_img = fixed_img * fixed_mask
            moving_img = moving_img * moving_mask

        
        shape = fixed_img.shape[1:-1]
        
        zeros = torch.zeros((1, *shape, len(shape)))
        
        return { "fixed_name" : fix_idx,
                "moving_name" : mov_idx,
                "fixed_img" : fixed_img, 
                "moving_img" : moving_img, 
                "fixed_mask" : fixed_mask, 
                "moving_mask" : moving_mask,
                "zero_flow_field" : zeros,
                "fixed_affine" : fixed_affine}

