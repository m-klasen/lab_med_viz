import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from pytorch_lamb import Lamb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiplicativeLR

class CFG:
    exp_name="baseline5"
    dataset_path="/home/mlk/cc359"
    voxel_spacing=(1, 0.95, 0.95)
    img_size=(256,256)
    fold=[0,1,2,3,4,5] #0-5 each is a specific domain (specific MRI scanner settings)
    bs=64
    num_workers=8
    epochs=100
    accumulation_steps=1
    fp16=True
    model_name="Unet2D"
    optim=Adam
    scheduler=MultiplicativeLR
    scheduler_multi_lr_fact=0.96
    crit=nn.BCEWithLogitsLoss()
    n_filters = 16
    n_chans_in = 1
    n_chans_out = 1
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr=2e-4
    wd=1e-2
    sdice_tolerance = 1
    results_dir=f"{exp_name}/"