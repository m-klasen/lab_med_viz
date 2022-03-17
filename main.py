import argparse
from configs.config import CFG
from utils import *

from spottunet.dataset.cc359 import *
from spottunet.split import one2all
from spottunet.torch.module.unet import UNet2D
from spottunet.utils import sdice
from dpipe.im.metrics import dice_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image

from monai import transforms as T
from monai.transforms import Compose, apply_transform
from fastprogress.fastprogress import master_bar, progress_bar

import json
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import ndimage
from dpipe.im.shape_ops import zoom
import cv2
import os
import gc
from collections import defaultdict
from pathlib import Path
import segmentation_models_pytorch as smp


import wandb
from dataset.dataloader import *
from dataset.loader import PrefetchLoader, fast_collate
from dataset.dataloader_utils import *

from dataset.mixup import FastCollateMixup
from dataset.augment import get_transforms, get_test_transforms

from scheduler import LinearWarmupCosineAnnealingLR
from trainer import Trainer
from dataset.loader import *

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

def class2str(f):
    return [[name, getattr(f, name)] for name in dir(f) if not name.startswith('__')]

def write_config(CFG):
    with open(f"{CFG.results_dir}/config.txt", "w") as f:
        for n,v in class2str(CFG):
            f.write(f"{n}={v} \n")
    f.close()

def run_fold(fold):
    result_dir = CFG.results_dir + "/mode_"+str(fold)
    os.makedirs(result_dir, exist_ok=True)
    #wandb.tensorboard.patch(root_logdir=result_dir+"/logs")
    run = wandb.init(project="domain_shift",
                     group=CFG.model_name,
                     name=f"mode_{str(fold)}",
                     job_type=CFG.job_type,
                     config=class2dict(CFG),
                     reinit=True,
                     sync_tensorboard=True)
    
    writer = SummaryWriter(log_dir=result_dir+"/logs")
    write_config(CFG)
    cc359_df = pd.read_csv(f"{CFG.dataset_path}/meta.csv",delimiter=",")
    
    mixup_fn = None
    if CFG.mixup:
        mixup_args = dict(
            mixup_alpha=CFG.mixup, cutmix_alpha=CFG.cutmix, cutmix_minmax=None,
            prob=CFG.mixup_prob, switch_prob=CFG.mixup_switch_prob, mode='batch',
            label_smoothing=CFG.smoothing, num_classes=2)
        collate_fn = FastCollateMixup(**mixup_args)
    elif fast_collate:
        collate_fn = fast_collate
        
    seed = 0xBadCafe
    val_size = 4
    n_experiments = len(cc359_df.fold.unique())
    split = one2all(df=cc359_df,val_size=val_size)[:n_experiments]


    train_df = cc359_df.iloc[split[fold][0]].reset_index()
    valid_df = cc359_df.iloc[split[fold][1]].reset_index()
    test_df  = cc359_df.iloc[split[fold][2]].reset_index()

    
    sa_x = None; sa_y = None
    valid_sa_x = None; valid_sa_y = None
    fcm_ma = None
    if CFG.cache:
        print("Caching Train Data ...")
        sa_x,sa_y = create_shared_arrays(CFG,train_df,root_dir=CFG.dataset_path)
        valid_sa_x,valid_sa_y = create_3d_shared_arrays(CFG,valid_df,root_dir=CFG.dataset_path)
        if CFG.fcm_mask:
            fcm_ma = create_shared_fcm_masks(CFG,train_df,root_dir=CFG.dataset_path)
    train_dataset = CC359_Dataset(CFG,df=train_df,root_dir=CFG.dataset_path,
                                  voxel_spacing=CFG.voxel_spacing,transforms=get_transforms(CFG.tfms),
                                  mode="train", cache=CFG.cache, cached_x=sa_x, cached_y=sa_y, cached_fcm_mask=fcm_ma)
    
    valid_dataset = CC359_Dataset(CFG,df=valid_df,root_dir=CFG.dataset_path,
                                  voxel_spacing=CFG.voxel_spacing,transforms=get_test_transforms(),
                                  mode="val", cache=CFG.cache, cached_x=valid_sa_x,cached_y=valid_sa_y)
    test_dataset = CC359_Dataset(CFG,df=test_df,root_dir=CFG.dataset_path,
                                 voxel_spacing=CFG.voxel_spacing,transforms=get_test_transforms(),
                                 mode="test", cache=False)
    
    train_loader = PrefetchLoader(DataLoader(train_dataset,
                                              batch_size=CFG.bs,
                                              shuffle=True,
                                              num_workers=CFG.num_workers,
                                              sampler=None,
                                              collate_fn=collate_fn,
                                              pin_memory=False,
                                              drop_last=True),
                                  fp16=CFG.fp16)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=1,shuffle=False,
                              num_workers=1,pin_memory=False)
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=1,shuffle=False,
                                  num_workers=1,pin_memory=False)

    model = UNet2D(n_chans_in=CFG.n_chans_in, n_chans_out=CFG.n_chans_out, n_filters_init=CFG.n_filters)

    model.to(CFG.device)
    
    optim_dict = dict(optim=CFG.optim,lr=CFG.lr,weight_decay=CFG.wd)
    optimizer = get_optimizer(model, **optim_dict)
    
    #scheduler = CFG.scheduler(optimizer, lr_lambda=lambda epoch: CFG.scheduler_multi_lr_fact )
    if CFG.scheduler==torch.optim.lr_scheduler.OneCycleLR:
        scheduler = CFG.scheduler(optimizer, max_lr=CFG.lr, steps_per_epoch=len(train_loader), epochs=CFG.epochs)
    elif CFG.scheduler==torch.optim.lr_scheduler.MultiStepLR:
        scheduler = CFG.scheduler(optimizer, milestones=[30], gamma=0.1)
    elif CFG.scheduler==LinearWarmupCosineAnnealingLR:
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                    warmup_epochs=CFG.warmup_epochs,
                                                    max_epochs=CFG.epochs,
                                                    warmup_start_lr=CFG.warmup_lr)
    else:
        print("no scheduler selected")
    criterion = CFG.crit    

    
    trainer = Trainer(CFG,
                      model=model, 
                      device=CFG.device, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      criterion=criterion,
                      writer=writer,
                      fold=fold,
                      max_norm=CFG.max_norm,
                      mixup_fn=mixup_fn)
    
    history = trainer.fit(
            CFG.epochs, 
            train_loader, 
            valid_loader, 
            f"{result_dir}/", 
            CFG.epochs,
        )
    trainer.test(test_dataloader,result_dir)
    td_sdice = get_target_domain_metrics(CFG.dataset_path,Path(CFG.results_dir),fold)
    #writer.add_hparams(class2dict(CFG),td_sdice)
    wandb.log(td_sdice)
    writer.close()
    run.finish()

    del trainer
    del train_loader
    del valid_loader
    del train_dataset
    del valid_dataset
    gc.collect()
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline_exp_path', default=None, type=str)
    parser.add_argument('--aug_type', default=None, type=str)
    parser.add_argument('--rand_aug', default=False, type=bool)
    parser.add_argument('--PIL_tfms', nargs="+", default=None)
    parser.add_argument('--folds', nargs="+", default=None)
    parser.add_argument('--optim', default="Lamb", type=str)
    parser.add_argument('--mixup', default=None, type=float)
    parser.add_argument('--lr', default="Lamb", type=float)
    parser.add_argument('--fcm_mask', default=None)  
    
    args = parser.parse_args()
    if args.rand_aug:
        CFG.exp_name=f"baseline_focal_lovasz_{args.optim}_rand_aug_{args.aug_type}"
    elif args.PIL_tfms!=None:
        CFG.exp_name=f"baseline_focal_lovasz_{args.optim}_{args.aug_type}_{''.join(args.PIL_tfms)}"
    elif args.mixup!=None:
        CFG.exp_name=f"baseline_focal_lovasz_{args.optim}_{args.aug_type}_mixup{args.mixup}"
    else:
        CFG.exp_name=f"baseline_focal_lovasz_{args.optim}_{args.aug_type}_{args.fcm_mask}"
    CFG.results_dir=f"baseline_results/{CFG.exp_name}_frontback/"
    CFG.job_type=f"baseline_{args.aug_type}"
    CFG.fold=args.folds
    CFG.optim=args.optim
    CFG.lr=args.lr
    CFG.mixup=args.mixup
    CFG.PIL_tfms=args.PIL_tfms
    CFG.rand_aug=args.rand_aug
    CFG.fcm_mask=args.fcm_mask
    print(class2dict(CFG))
    
    for fold in CFG.fold:
        run_fold(int(fold))

if __name__ == "__main__":
    main()