import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from pytorch_lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR
from losses import *

setup2k_reg_opt = {3: {1: 0.000, }, 2: {1: 0.005, },
                   1: {1: 0.005, 3: 0.007, 6: 0.005, 12: 0.005, 24: 0.012, 36: 0.007, 48: 0.010}}

class CFG:
    dataset_path="/home/mlk/cc359"
    voxel_spacing=(1, 0.95, 0.95)
    img_size=(256,256)
    fold=[0] #0-5 each is a specific domain (specific MRI scanner settings)
    bs=8
    num_workers=8
    epochs=100
    accumulation_steps=1
    fp16=True
    model_name="Unet2D"
    optim=Adam
    scheduler=MultiplicativeLR
    scheduler_multi_lr_fact=0.96
    crit=MixedLoss(10.0,2.0)#nn.BCEWithLogitsLoss()
    n_filters = 16
    n_chans_in = 1
    n_chans_out = 1
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr=2e-4
    wd=1e-2
    max_norm=0.5
    sdice_tolerance = 1
    val_size=2
    slice_sampling_interval=48 # 1, 3, 6, 12, 24, 36, 48
    n_add_ids=1
    temperature=0.1
    k_reg=setup2k_reg_opt[n_add_ids][slice_sampling_interval]
    reg_mode='l1'
    exp_name=f"exp_03_ssi_{slice_sampling_interval}_nid_{n_add_ids}_tv"
    results_dir=f"spottune_results/{exp_name}/"