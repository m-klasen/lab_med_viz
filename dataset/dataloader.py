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



scale = tuple((0.08, 1.0))  # default imagenet scale range
ratio = tuple((3./4., 4./3.)) # default imagenet ratio range
mean = [0.5]
_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]
aa_params = dict(
            translate_const=int(256 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
#aa_params['interpolation'] = str_to_pil_interp(interpolation)

auto_augment = "rand-m9-mstd0.5"
interpolation = 'random'

from rand_augment import rand_augment_transform
from rrc import RandomResizedCropAndInterpolation, ToNumpy
from torchvision import transforms

from skimage.restoration import denoise_tv_chambolle

from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
class TV_aug(DualTransform):
    def apply(self, img, **params):
        return denoise_tv_chambolle(img, weight=0.1)
    
## Augmentations used
# A.GaussianBlur(blur_limit=(3, 7), p=0.5),
# A.RandomGamma(gamma_limit=(80, 120))
tfms = A.Compose([
    TV_aug(p=0.5) 
    ])

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
                            T.SpatialPadd(keys=("image","seg"),spatial_size=self.CFG.img_size)])
        
        self.tfms2 = RandomResizedCropAndInterpolation(CFG.img_size, scale=scale, ratio=ratio, interpolation=interpolation)
        
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
            img  = self.shared_array_x[idx]
            segm = self.shared_array_y[idx]
            #img = self.imgs[idx]
            #segm = self.segms[idx]
        else:
            img_fn = self.img_paths[idx]
            sgm_fn = self.segm_paths[idx]
            img = nib.load(f"{self.root_dir}/{img_fn}").get_fdata()
            segm = nib.load(f"{self.root_dir}/{sgm_fn}").get_fdata()  

            #img, segm = self.scale_voxel_spacing(idx, img, segm)
            img = self.scale_mri(img)
            
            img = np.transpose(img,(2,0,1))
            segm = np.transpose(segm,(2,0,1))
                
            tfmed = self.T({'image':img, 'seg':segm})       
            img = tfmed['image']
            segm = tfmed['seg']
            
            

        
        if self.transform:
            out = self.transform(image=img)
            img = out["image"]
            
            #img, segm = Image.fromarray(img*255).convert('L'),Image.fromarray(segm*255).convert('L')
            #img, segm = self.tfms2(img,segm)
            #img, segm = (ToNumpy()(img)/255.).squeeze(), np.array(segm)/255.
            
        if self.mode=="train":
            return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0), torch.from_numpy(segm).to(dtype=torch.float)
        elif self.mode in ["val","test"]:
            return  torch.from_numpy(img).to(dtype=torch.float32), torch.from_numpy(segm).to(dtype=torch.float), self.id[idx]

        
        
