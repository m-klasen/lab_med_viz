import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from pytorch_lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.lr_scheduler import OneCycleLR
from losses import *
from dataset.dataloader import *

setup2k_reg_opt = {3: {1: 0.000, }, 2: {1: 0.005, },
                   1: {1: 0.005, 3: 0.007, 6: 0.005, 12: 0.005, 24: 0.012, 36: 0.007, 48: 0.010}}

class CFG:
    exp_name="3d_baseline"
    dataset_path="/home/mlk/cc359"
    # Img Processing
    voxel_spacing=(1, 0.95, 0.95)
    img_size=(256,256,256)
    # Training Params
    cache=True
    fold=[3] #0-5 each is a specific domain (specific MRI scanner settings)
    bs=1
    num_workers=8
    epochs=100
    accumulation_steps=2
    fp16=True
    model_name="UNet3D"
    #Optimizer
    optim='Lamb'
    lr=1e-3
    wd=1e-5
    scheduler="warmup_cosine" #OneCycleLR #MultiplicativeLR
    scheduler_multi_lr_fact=0.96
    warmup_epochs=5
    crit=MixedLoss(10.0,2.0)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_norm=False # 1.0, float or False
    sdice_tolerance = 1
    results_dir=f"baseline_results/{exp_name}/"
    #Augmentations
    tfms="default"
    mixup=None #0.2
    smoothing=0.0
    cutmix=1.0
    mixup_prob=1.0
    mixup_switch_prob=0.5
