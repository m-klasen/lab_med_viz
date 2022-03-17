import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from fastprogress.fastprogress import master_bar, progress_bar

from PIL import Image

from monai import transforms as T

import json
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import ndimage
from dpipe.im.shape_ops import zoom
import cv2
import os
from collections import defaultdict
from pathlib import Path



import ctypes
import multiprocessing as mp

class CC359_Dataset(Dataset):
    def __init__(self, CFG, df, root_dir, voxel_spacing, transforms=None, mode="train", cache=True, cached_x=None,cached_y=None):
        self.CFG = CFG
        self.df = df
        self.root_dir = root_dir
        self.voxel_spacing = voxel_spacing
        self.transform = transforms
        self.mode = mode
        self.cache = cache
        self.id = df['id']
        self.img_paths = df['MRI_scaled_voxel_spacing']
        self.segm_paths = df['brain_mask_scaled_voxel_spacing']
        self.sample_voxel_spacing = np.array([df['x'],df['y'],df['z']])

        
        self.T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=self.CFG.img_size),
                            T.SpatialPadd(keys=("image","seg"),spatial_size=self.CFG.img_size),
                            T.Resized(keys=("image","seg"), spatial_size=[224,224,224]),
                           ])
        
        if self.cache:
            self.shared_array_x = cached_x
            self.shared_array_y = cached_y     

            self.curr_idx = self.shared_array_x.shape[0]
        
    def __len__(self):
        if self.cache==True:
            return self.curr_idx
        else:
            return len(self.id)

    def scale_voxel_spacing(self, idx, img, segm):
        sample_vxsp = self.sample_voxel_spacing[:,idx]
        scale_factor = sample_vxsp / self.voxel_spacing
        
        img = zoom(img, scale_factor, order=3)
        segm = zoom(segm, scale_factor, order=3)
        return img,segm
    
    def scale_mri(self, image, q_min=1, q_max=99):
        image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
        image -= np.min(image)
        image /= np.max(image)
        return np.float32(image)
    
    def __getitem__(self, idx):
        if self.cache:
            img  = self.shared_array_x[idx][None,...]
            segm = self.shared_array_y[idx][None,...]
        else:
            img_fn = self.img_paths[idx]
            sgm_fn = self.segm_paths[idx]
            img = nib.load(f"{self.root_dir}/{img_fn}").get_fdata()
            segm = nib.load(f"{self.root_dir}/{sgm_fn}").get_fdata()  

            #img, segm = self.scale_voxel_spacing(idx, img, segm)
            img = self.scale_mri(img)
            
            img = np.transpose(img,(2,0,1))[None,:,:,:]
            segm = np.transpose(segm,(2,0,1))[None,:,:,:]
                
            tfmed = self.T({'image':img, 'seg':segm})       
            img = tfmed['image']
            segm = tfmed['seg']
            
            

        
        if self.transform:
            out = self.transform(image=img)
            img = out["image"]

            
        if self.mode=="train":
            return torch.from_numpy(img).to(dtype=torch.float32), torch.from_numpy(segm).to(dtype=torch.float)
        elif self.mode in ["val","test"]:
            return  torch.from_numpy(img).to(dtype=torch.float32), torch.from_numpy(segm).to(dtype=torch.float), self.id[idx]

        
        
