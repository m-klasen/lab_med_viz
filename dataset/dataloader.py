import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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


from config import CFG

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
            translate_const=int(CFG.img_size[0] * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            )
#aa_params['interpolation'] = str_to_pil_interp(interpolation)

auto_augment = "rand-m9-mstd0.5"
interpolation = 'random'

from rand_augment import rand_augment_transform
from rrc import RandomResizedCropAndInterpolation, ToNumpy
from torchvision import transforms
tfms = A.Compose([
    A.CenterCrop(256,256),
    A.PadIfNeeded(256,256)])

import ctypes
import multiprocessing as mp

class CC359_Dataset(Dataset):
    def __init__(self, df, root_dir, voxel_spacing, transforms=None, mode="train", cache=True):
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
        
        self.tfms2 = RandomResizedCropAndInterpolation(CFG.img_size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.tfms3 = rand_augment_transform(auto_augment, aa_params)
        
        self.T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=CFG.img_size),
                            T.SpatialPadd(keys=("image","seg"),spatial_size=CFG.img_size)])
        
        if self.cache:
            if mode=="train": n_samples = 12000
            elif mode=="val": n_samples = 1000
            shared_array_base = mp.Array(ctypes.c_float, n_samples*256*256)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            self.shared_array = shared_array.reshape(n_samples,256,256)
            print(self.shared_array.shape)
            
            shared_array_base2 = mp.Array(ctypes.c_float, n_samples*256*256)
            shared_array2 = np.ctypeslib.as_array(shared_array_base2.get_obj())
            self.shared_array2 = shared_array2.reshape(n_samples, 256,256)            
            
            self.imgs, self.segms = [],[]
            curr_idx = 0
            for idx in progress_bar(range(len(self.df)),len(self.df)):
                img_fn = self.img_paths[idx]
                sgm_fn = self.segm_paths[idx]
                img = nib.load(f"{self.root_dir}/{img_fn}").get_fdata()
                segm = nib.load(f"{self.root_dir}/{sgm_fn}").get_fdata()  
                #img, segm = self.scale_voxel_spacing(idx, img, segm)
                img = self.scale_mri(img)

                
                tfmed = self.T({'image':img, 'seg':segm})       
                img = tfmed['image']
                segm = tfmed['seg']

                c = img.shape[0]
                self.shared_array[curr_idx:curr_idx+c] = img
                self.shared_array2[curr_idx:curr_idx+c] = segm
                curr_idx += c
                #self.imgs.append(img); self.segms.append(segm)
            #self.imgs = np.concatenate(self.imgs)
            #self.segms = np.concatenate(self.segms)
            self.curr_idx = curr_idx
        
    def __len__(self):
        return self.curr_idx

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
            img = self.shared_array[idx]
            segm = self.shared_array2[idx]
            #img = self.imgs[idx]
            #segm = self.segms[idx]
        else:
            img_fn = self.img_paths[idx]
            sgm_fn = self.segm_paths[idx]
            img = nib.load(f"{self.root_dir}/{img_fn}").get_fdata()
            segm = nib.load(f"{self.root_dir}/{sgm_fn}").get_fdata()  

            #img, segm = self.scale_voxel_spacing(idx, img, segm)
            img = self.scale_mri(img)
            
            slc = np.random.randint(img.shape[0] // 1)*1
            img,segm = img[slc,:,:],segm[slc,:,:]

        
        if self.transform:
            #monai tfms
            #tfmed = self.T({'image':img, 'seg':segm})       
            #img = tfmed['image']; 
            #segm = tfmed['seg']

            img, segm = Image.fromarray(img*255).convert('L'),Image.fromarray(segm*255).convert('L')
            img, segm = self.tfms2(img,segm)
            img, segm = self.tfms3(img,segm)
            img, segm = (ToNumpy()(img)/255.).squeeze(), np.array(segm)/255.
            
        if self.mode in ["train", "val"]:
            return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0), torch.from_numpy(segm).to(dtype=torch.long)
        elif self.mode=="test":
            return torch.as_tensor(img, dtype=torch.float), torch.as_tensor(segm, dtype=torch.float), self.id[idx]
