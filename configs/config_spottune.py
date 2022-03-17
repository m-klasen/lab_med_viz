# import torch
# import numpy as np
# import torch.nn as nn
# from torch.optim import Adam, SGD
# from pytorch_lamb import Lamb
# from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, MultiplicativeLR, MultiStepLR
# from losses import *
# from scheduler import LinearWarmupCosineAnnealingLR
# from segmentation_models_pytorch.losses import JaccardLoss, LovaszLoss


# class CFG:
#     dataset_path="/home/mlk/cc359"
#     voxel_spacing=(1, 0.95, 0.95)
#     img_size=[256,256,256]
#     transpose=(0,1,2)
#     img_paths="MRI_scaled_voxel_spacing" #"fcm_norm"
#     segm_paths="brain_mask_scaled_voxel_spacing"
#     scale_mri=True
#     percentile=[1,99]
#     cache=True
#     fold=[10,11,12,13,14] #0-5 each is a specific domain (specific MRI scanner settings)
#     bs=16
#     num_workers=8
#     epochs=100
#     accumulation_steps=1
#     fp16=False
#     early_stopping=False
#     model_name="Unet2D"
#     optim="SGD"
#     scheduler=MultiStepLR
#     scheduler_multi_lr_fact=0.96
#     crit=nn.BCEWithLogitsLoss()
#     n_filters = 16
#     n_chans_in = 1
#     n_chans_out = 1
#     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     lr=1e-3
#     lr_policy=1e-2
#     warmup_epochs=5
#     warmup_lr=1e-5
#     wd=0.
#     max_norm=None
#     sdice_tolerance = 1
#     val_size=2
#     slice_sampling_interval=48 # 1, 3, 6, 12, 24, 36, 48
#     n_add_ids=1
#     temperature=0.1
#     tfms="default"
#     rand_aug=False
#     PIL_tfms=None #["Brightness"]
#     fcm_mask=None
#     mixup=None #0.2
#     smoothing=0.0
#     cutmix=1.0
#     mixup_prob=1.0
#     mixup_switch_prob=0.5
#     reg_mode='l1'
#     exp_name=f"exp_ssi_{slice_sampling_interval}_nid_{n_add_ids}"
#     baseline_exp_path="baseline_results/baseline23_exp4_weak_rrc"
#     results_dir=f"spottune_results/{exp_name}/"
#     job_type="spottune"




import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from pytorch_lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, MultiplicativeLR
from losses import *
from scheduler import LinearWarmupCosineAnnealingLR
from segmentation_models_pytorch.losses import JaccardLoss, LovaszLoss


class CFG:
    dataset_path="/home/mlk/cc359"
    voxel_spacing=(1, 0.95, 0.95)
    img_size=[256,256,256]
    transpose=(2,0,1)
    img_paths="MRI_scaled_voxel_spacing" #"fcm_norm"
    segm_paths="brain_mask_scaled_voxel_spacing"
    scale_mri=True
    percentile=None #[1,99]
    cache=True
    fold=[10,11,12,13,14] #0-5 each is a specific domain (specific MRI scanner settings)
    bs=16
    num_workers=8
    epochs=100
    accumulation_steps=1
    fp16=False
    early_stopping=False
    model_name="Unet2D"
    optim="Adam"
    scheduler=LinearWarmupCosineAnnealingLR #OneCycleLR #MultiplicativeLR
    scheduler_multi_lr_fact=0.96
    crit=LovaszFocal(10.0,2.0) #LovaszLoss('binary')crit=LovaszLoss('binary') #MixedLoss(10.0,2.0)#nn.BCEWithLogitsLoss()
    n_filters = 16
    n_chans_in = 1
    n_chans_out = 1
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr=1e-5
    lr_policy=1e-3
    warmup_epochs=5
    warmup_lr=1e-6
    fcm_mask=None
    wd=1e-2
    max_norm=1.0 #None or [0.,1.]
    sdice_tolerance = 1
    val_size=2
    slice_sampling_interval=48 # 1, 3, 6, 12, 24, 36, 48
    n_add_ids=1
    temperature=0.1
    tfms="default"
    rand_aug=False
    PIL_tfms=None #["Brightness"]
    mixup=None #0.2
    smoothing=0.0
    cutmix=1.0
    mixup_prob=1.0
    mixup_switch_prob=0.5
    reg_mode='l1'
    exp_name=f"exp_ssi_{slice_sampling_interval}_nid_{n_add_ids}"
    baseline_exp_path="baseline_results/baseline23_exp4_weak_rrc"
    results_dir=f"spottune_results/{exp_name}/"
    job_type="spottune"