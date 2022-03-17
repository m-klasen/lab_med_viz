from webbrowser import get
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A
from dataset.augment import auto_augment_fkt, get_PIL_tfms, get_transforms
from dataset.rrc import *
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

from .dataloader_utils import *


import ctypes
import multiprocessing as mp

def id_to_scanner(id, test_df):
    df = test_df[test_df['id']==id]
    return df['tomograph_model'].values[0] + str(df['tesla_value'].values[0])

class CC359_Dataset(Dataset):
    def __init__(self, CFG, df, root_dir, voxel_spacing, transforms=None, mode="train", cache=True, cached_x=None,cached_y=None, cached_fcm_mask=None, td_specific_aug=False):
        self.CFG = CFG
        self.df = df
        self.root_dir = root_dir
        self.voxel_spacing = voxel_spacing
        self.transform = transforms
        self.mode = mode
        self.cache = cache
        self.td_specific_aug = td_specific_aug
        self.id = df['id']
        self.img_paths = df[CFG.img_paths]
        self.segm_paths = df[CFG.segm_paths]
        self.sample_voxel_spacing = np.array([df['x'],df['y'],df['z']])
        if self.CFG.fcm_mask and self.mode=="train":
            self.cached_fcm_masks = cached_fcm_mask
        self.csf_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_csf")[:-7] + f"csf_membership.nii.gz" for p in self.img_paths]
        self.gm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_gm")[:-7] + f"gm_membership.nii.gz" for p in self.img_paths]
        self.wm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_wm")[:-7] + f"wm_membership.nii.gz" for p in self.img_paths]
        if self.CFG.fcm_mask:
            if CFG.fcm_mask=="all":
                self.csf_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_csf")[:-7] + f"csf_membership.nii.gz" for p in self.img_paths]
                self.gm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_gm")[:-7] + f"gm_membership.nii.gz" for p in self.img_paths]
                self.wm_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_wm")[:-7] + f"wm_membership.nii.gz" for p in self.img_paths]
            else:
                self.csf_mask_paths = [p.replace("images_scaled_voxel_spacing", f"fcm_norm_{CFG.fcm_mask}")[:-7] + f"{CFG.fcm_mask}_membership.nii.gz" for p in self.img_paths]    


        
        self.T = T.Compose([T.CenterSpatialCropd(keys=("image","seg"),roi_size=CFG.img_size),
                            T.SpatialPadd(keys=("image","seg"),spatial_size=CFG.img_size)])
        if self.CFG.rand_aug==True:
            self.random_augment = auto_augment_fkt()
        if self.CFG.PIL_tfms:
            self.pil_tfms = get_PIL_tfms(self.CFG.PIL_tfms)
        if self.cache:
            self.shared_array_x = cached_x
            self.shared_array_y = cached_y     

            self.curr_idx = self.shared_array_x.shape[0]
        
    def __len__(self):
        if self.cache==True:
            return self.curr_idx
        else:
            return len(self.id)
    
    def __getitem__(self, idx):
        
        if self.cache:
            img  = self.shared_array_x[idx]
            segm = self.shared_array_y[idx]
            #img = self.imgs[idx]
            #segm = self.segms[idx]
            
            #img = np.clip(img, *np.percentile(img, [1, 99]))
            #if self.mode=="train":
            #    m = self.df.loc[idx//256,"mean"]; s = self.df.loc[idx//256,"std"]
            #    img = (img-m)/s
            #elif self.mode=="val":
            #    m = self.df.loc[idx,"mean"]; s = self.df.loc[idx,"std"]
            #    img = (img-m)/s 
            
        else:
            img_fn = self.img_paths[idx]
            sgm_fn = self.segm_paths[idx]
            img = nib.load(f"{self.root_dir}/{img_fn}").get_fdata()
            segm = nib.load(f"{self.root_dir}/{sgm_fn}").get_fdata()  

            #img, segm = self.scale_voxel_spacing(idx, img, segm)
            if self.CFG.scale_mri:
                img = scale_mri(img, self.CFG.percentile)
            
            img = np.transpose(img,self.CFG.transpose)[None,:,:,:]
            segm = np.transpose(segm,self.CFG.transpose)[None,:,:,:]
                
            tfmed = self.T({'image':img, 'seg':segm})       
            img = tfmed['image'].squeeze()
            segm = tfmed['seg'].squeeze()
            
            scanner = id_to_scanner(self.id[idx],self.df)
            if self.td_specific_aug and scanner=="philips3":
                print(self.id[idx])
                csf_mask_fn = self.csf_mask_paths[idx]
                csf_mask = np.array(nib.load(f"{self.root_dir}/{csf_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
                fcm_mask = csf_mask
                tfmed = self.T({'image':fcm_mask[None,:,:,:], 'seg':fcm_mask[None,:,:,:]})       
                fcm_mask = tfmed['image'].squeeze()
        
            if self.CFG.fcm_mask:
                if self.CFG.fcm_mask=="all":
                    csf_mask_fn = self.csf_mask_paths[idx]
                    gm_mask_fn = self.gm_mask_paths[idx]
                    wm_mask_fn = self.wm_mask_paths[idx]
                    csf_mask = np.array(nib.load(f"{self.root_dir}/{csf_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
                    gm_mask = np.array(nib.load(f"{self.root_dir}/{gm_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
                    wm_mask = np.array(nib.load(f"{self.root_dir}/{wm_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
                    fcm_mask = csf_mask & gm_mask & wm_mask
                    tfmed = self.T({'image':fcm_mask[None,:,:,:], 'seg':fcm_mask[None,:,:,:]})       
                    fcm_mask = tfmed['image'].squeeze()
                else:
                    csf_mask_fn = self.csf_mask_paths[idx]
                    fcm_mask = np.array(nib.load(f"{self.root_dir}/{csf_mask_fn}").get_fdata() > 0.5,dtype=np.bool_)
                    tfmed = self.T({'image':fcm_mask[None,:,:,:], 'seg':fcm_mask[None,:,:,:]})       
                    fcm_mask = tfmed['image'].squeeze()
            
        if self.CFG.fcm_mask and self.mode=="train":
            #mask_pth = self.csf_mask_paths[idx // self.CFG.img_size[0]]
            fcm_mask = self.cached_fcm_masks[idx]
            
        if self.transform:
            

            if self.CFG.rand_aug==True and self.mode=="train":
                img, segm = Image.fromarray(img.astype(np.float64)*255).convert('L'),Image.fromarray(segm*255).convert('L')
                img, segm = self.random_augment(img,segm)
                img, segm = (np.array(img)/255.).squeeze(), np.array(segm)/255.
            if self.CFG.PIL_tfms==True and self.mode=="train":
                img, segm = Image.fromarray(img.astype(np.float64)*255).convert('L'),Image.fromarray(segm*255).convert('L')
                img, segm = self.pil_tfms(img,segm)
                img, segm = (np.array(img)/255.).squeeze(), np.array(segm)/255.

            if self.td_specific_aug and scanner=="philips3":
                tfms = T.MaskIntensity()
                fcm_masked_img = tfms(img,mask_data=fcm_mask)
                inv_masked_img = tfms(img,mask_data=~fcm_mask)
                tfmed = get_transforms("intensity_fcm")({'image':fcm_masked_img})       
                fcm_masked_img = tfmed['image']
                
                img =  inv_masked_img + fcm_masked_img


            # Check for albumentation or monai tfms:
            if self.CFG.fcm_mask:
                tfms = T.MaskIntensity()
                fcm_masked_img = tfms(img,mask_data=fcm_mask)
                inv_masked_img = tfms(img,mask_data=~fcm_mask)
                if isinstance(self.transform, A.Compose):
                    out = self.transform(image=fcm_masked_img)
                    fcm_masked_img = out["image"]
                elif isinstance(self.transform, T.Compose):
                    tfmed = self.transform({'image':fcm_masked_img})       
                    fcm_masked_img = tfmed['image']
                
                #Need to norm the non masked input too
                img =  inv_masked_img + fcm_masked_img
                inv_tfms = get_transforms("default")
                tfms = inv_tfms(image=img)
                img = tfms["image"]
                
            else:
                if isinstance(self.transform, A.Compose):
                    out = self.transform(image=img,mask=segm)
                    img = out["image"]
                    segm = out["mask"]
                elif isinstance(self.transform, T.Compose):
                    tfmed = self.transform({'image':img, 'seg':segm})       
                    img = tfmed['image']
                    segm = tfmed['seg']
        
        
        if self.mode=="train":
            return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0), torch.from_numpy(segm).to(dtype=torch.float)
        elif self.mode in ["val","test"]:
            return  torch.from_numpy(img).to(dtype=torch.float32), torch.from_numpy(segm).to(dtype=torch.float), self.id[idx]

        
        
