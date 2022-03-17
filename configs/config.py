import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from pytorch_lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.lr_scheduler import OneCycleLR
from losses import *
from dataset.dataloader import *

from scheduler import LinearWarmupCosineAnnealingLR
from segmentation_models_pytorch.losses import JaccardLoss, LovaszLoss


class CFG:
    dataset_path="/home/mlk/cc359"
    # Img Processing
    voxel_spacing=(1, 0.95, 0.95)
    img_size=[256,256,256]
    transpose=(1,0,2)  # (2,0,1)#  (0,1,2)
    img_paths="MRI_scaled_voxel_spacing" #"fcm_norm"
    segm_paths="brain_mask_scaled_voxel_spacing"
    scale_mri=True
    percentile=None #[1,99]
    val_size=2
    slice_sampling_interval=1 # 1, 3, 6, 12, 24, 36, 48
    n_add_ids=1
    # Training Params
    cache=True
    fold=[3] #0-5 each is a specific domain (specific MRI scanner settings)
    bs=64
    num_workers=16
    epochs=40
    accumulation_steps=1
    fp16=False
    early_stopping=False
    #Model
    model_name="Unet2D"
    n_filters = 16
    n_chans_in = 1
    n_chans_out = 1
    #Optimizer
    optim='Adam'
    lr=1e-4
    wd=1e-2
    scheduler=LinearWarmupCosineAnnealingLR #MultiStepLR #MultiplicativeLR
    scheduler_multi_lr_fact=0.96
    warmup_epochs=5
    warmup_lr=1e-5
    crit=LovaszFocal(10.0,2.0)#nn.BCEWithLogitsLoss()#LovaszLoss('binary')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_norm=1.0 #1.0 # 1.0, float or False
    sdice_tolerance = 1
    #Augmentations
    fcm_mask="gm" # or "gm"
    tfms="default" #"default"
    rand_aug=False
    PIL_tfms=["Brightness"]
    mixup=None #0.2
    smoothing=0.0
    cutmix=1.0
    mixup_prob=1.0
    mixup_switch_prob=0.5
    exp_name=f"baseline_focal_lovasz_SGD_{tfms}"
    results_dir=f"baseline_results/{exp_name}/"

# SGD fallback that worked for baseline training
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.optim import Adam, S   GD
# from pytorch_lamb import Lamb
# from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
# from torch.optim.lr_scheduler import MultiplicativeLR
# from torch.optim.lr_scheduler import OneCycleLR
# from losses import *
# from dataset.dataloader import *

# from scheduler import LinearWarmupCosineAnnealingLR
# from segmentation_models_pytorch.losses import JaccardLoss, LovaszLoss


# class CFG:
#     dataset_path="/home/mlk/cc359"
#     # Img Processing
#     voxel_spacing=(1, 0.95, 0.95)
#     img_size=[256,256,256]
#     transpose=(0,1,2) #(0,1,2) #
#     img_paths="MRI_scaled_voxel_spacing" #"fcm_norm"
#     segm_paths="brain_mask_scaled_voxel_spacing"
#     scale_mri=True
#     percentile=[1,99]
#     val_size=2
#     slice_sampling_interval=1 # 1, 3, 6, 12, 24, 36, 48
#     n_add_ids=1
#     # Training Params
#     cache=True
#     fold=[3] #0-5 each is a specific domain (specific MRI scanner settings)
#     bs=64
#     num_workers=16
#     epochs=40
#     accumulation_steps=1
#     fp16=False
#     early_stopping=False
#     #Model
#     model_name="Unet2D"
#     n_filters = 16
#     n_chans_in = 1
#     n_chans_out = 1
#     #Optimizer
#     optim='Lamb'
#     lr=1e-3
#     wd=1e-2
#     scheduler=MultiStepLR#LinearWarmupCosineAnnealingLR #OneCycleLR #MultiplicativeLR
#     scheduler_multi_lr_fact=0.96
#     warmup_epochs=5
#     warmup_lr=1e-5
#     crit=nn.BCEWithLogitsLoss() #LovaszFocal(10.0,2.0)#LovaszLoss('binary')
    
#     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     max_norm=None #1.0 # 1.0, float or False
#     sdice_tolerance = 1
#     #Augmentations
#     fcm_mask=None # or "gm"
#     tfms=None #"default"
#     rand_aug=False
#     PIL_tfms=["Brightness"]
#     mixup=None #0.2
#     smoothing=0.0
#     cutmix=1.0
#     mixup_prob=1.0
#     mixup_switch_prob=0.5
#     exp_name=f"baseline_focal_lovasz_SGD_{tfms}"
#     results_dir=f"baseline_results/{exp_name}/"
