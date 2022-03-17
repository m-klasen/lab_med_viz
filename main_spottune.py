import argparse
from configs.config_spottune import CFG

from spottunet.dataset.cc359 import *
from spottunet.split import one2one
from models.spottune_unet import UNet2D
from models.resnet import resnet
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

from utils import *

from scheduler import LinearWarmupCosineAnnealingLR
from trainer_spottune import SpotTuneTrainer
from dataset.dataloader import *
from dataset.loader import *
from dataset.dataloader_utils import *
from dataset.augment import *

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

from dataset.dataloader import *
from dataset.loader import *

setup2k_reg_opt = {3: {1: 0.000, }, 
                    2: {1: 0.005, 3: 0.007, 6: 0.005, 12: 0.005, 24: 0.012, 36: 0.007, 48: 0.010 },
                    1: {1: 0.005, 3: 0.007, 6: 0.005, 12: 0.005, 24: 0.012, 36: 0.007, 48: 0.020}}

def run_fold(fold):
    result_dir = CFG.results_dir + "/mode_"+str(fold)
    os.makedirs(result_dir, exist_ok=True)
    #wandb.tensorboard.patch(root_logdir=result_dir+"/logs")
    run = wandb.init(project="domain_shift",
                     group=CFG.model_name,
                     name=f"mode_{str(fold)}_ssi_{CFG.slice_sampling_interval}_nid_{CFG.n_add_ids}",
                     job_type=CFG.job_type,
                     config=class2dict(CFG),
                     reinit=True,
                     sync_tensorboard=True)
    
    writer = SummaryWriter(log_dir=result_dir+"/logs")
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

    k_reg=setup2k_reg_opt[CFG.n_add_ids][CFG.slice_sampling_interval]
    seed = 0xBadCafe
    pretrained = True
    n_first_exclude = 5
    n_exps = 30
    split = one2one(cc359_df, val_size=CFG.val_size, n_add_ids=CFG.n_add_ids,
                train_on_add_only=pretrained, seed=seed)[n_first_exclude:n_exps]
        
    train_df = cc359_df.iloc[split[fold][0]].reset_index()
    valid_df = cc359_df.iloc[split[fold][1]].reset_index()
    test_df  = cc359_df.iloc[split[fold][2]].reset_index()

    print("Caching Train Data ...")
    
    sa_x = None; sa_y = None
    valid_sa_x = None; valid_sa_y = None
    if CFG.cache:
        print("Caching Train Data ...")
        sa_x,sa_y = create_shared_arrays(CFG,train_df,root_dir=CFG.dataset_path)
        valid_sa_x,valid_sa_y = create_3d_shared_arrays(CFG,valid_df,root_dir=CFG.dataset_path)
    train_dataset = CC359_Dataset(CFG,df=train_df,root_dir=CFG.dataset_path,
                                  voxel_spacing=CFG.voxel_spacing,transforms=get_transforms(CFG.tfms),
                                  mode="train", cache=CFG.cache, cached_x=sa_x, cached_y=sa_y)
    
    valid_dataset = CC359_Dataset(CFG,df=valid_df,root_dir=CFG.dataset_path,
                                  voxel_spacing=CFG.voxel_spacing,transforms=get_test_transforms(),
                                  mode="val", cache=CFG.cache, cached_x=valid_sa_x,cached_y=valid_sa_y)
    test_dataset = CC359_Dataset(CFG,df=test_df,root_dir=CFG.dataset_path,
                                 voxel_spacing=CFG.voxel_spacing,transforms=get_test_transforms(),
                                 mode="test", cache=False)
    
    train_loader = DataLoader(train_dataset,
                                              batch_size=CFG.bs,
                                              shuffle=True,
                                              num_workers=CFG.num_workers,
                                              sampler=None,
                                              collate_fn=collate_fn,
                                              pin_memory=False,
                                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=1,shuffle=False,
                              num_workers=1,pin_memory=False)
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=1,shuffle=False,
                                  num_workers=1,pin_memory=False)
    
    model = UNet2D(n_chans_in=CFG.n_chans_in, n_chans_out=CFG.n_chans_out, n_filters_init=CFG.n_filters)
    
    model_policy = resnet(num_class=64)
    
    load_model_state_fold_wise(architecture=model, baseline_exp_path=CFG.baseline_exp_path, exp=fold,
                               modify_state_fn=modify_state_fn_spottune, n_folds=len(cc359_df.fold.unique()),
                               n_first_exclude=n_first_exclude),
    freeze_model_spottune(model)
    
    model.to(CFG.device)
    model_policy.to(CFG.device)
    
    # Optims
    optim_dict = dict(optim=CFG.optim,lr=CFG.lr,weight_decay=CFG.wd)
    optimizer_main = get_optimizer(model, **optim_dict)
    optimizer_policy = get_optimizer(model_policy, **optim_dict)
    # Scheduler
    if CFG.scheduler==torch.optim.lr_scheduler.OneCycleLR:
        steps = max(len(train_loader),1)
        scheduler_main = CFG.scheduler(optimizer_main, max_lr=CFG.lr, 
                                       steps_per_epoch=steps,
                                       epochs=CFG.epochs)
        scheduler_policy = CFG.scheduler(optimizer_policy, max_lr=CFG.lr, 
                                       steps_per_epoch=steps, 
                                       epochs=CFG.epochs)
    elif CFG.scheduler==torch.optim.lr_scheduler.MultiplicativeLR:
        scheduler_main = CFG.scheduler(optimizer_main, lr_lambda=lambda epoch: CFG.scheduler_multi_lr_fact )
        scheduler_policy = CFG.scheduler(optimizer_policy, lr_lambda=lambda epoch: CFG.scheduler_multi_lr_fact )
    elif CFG.scheduler==torch.optim.lr_scheduler.MultiStepLR:
        scheduler_main = CFG.scheduler(optimizer_main, milestones=[75], gamma=0.1)
        scheduler_policy = CFG.scheduler(optimizer_policy, milestones=[75], gamma=0.1)
    elif CFG.scheduler==LinearWarmupCosineAnnealingLR:
        scheduler_main = LinearWarmupCosineAnnealingLR(optimizer_main,
                                                    warmup_epochs=CFG.warmup_epochs,
                                                    max_epochs=CFG.epochs,
                                                    warmup_start_lr=CFG.warmup_lr)
        scheduler_policy = LinearWarmupCosineAnnealingLR(optimizer_policy,
                                                    warmup_epochs=CFG.warmup_epochs,
                                                    max_epochs=CFG.epochs,
                                                    warmup_start_lr=CFG.warmup_lr)

    
    criterion = CFG.crit



    
    trainer = SpotTuneTrainer(CFG,
                      model,
                      model_policy,
                      CFG.device, 
                      optimizer_main,
                      scheduler_main,
                      optimizer_policy,
                      scheduler_policy,
                      criterion,writer,fold,
                      CFG.max_norm,
                      CFG.temperature,k_reg,CFG.reg_mode)
    
    history = trainer.fit(
            CFG.epochs, 
            train_loader, 
            valid_loader, 
            f"{result_dir}/", 
            CFG.epochs,
        )
    
    
    del train_loader
    del valid_loader
    del train_dataset
    del valid_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer.test(test_dataloader,result_dir)
    td_sdice = get_target_domain_metrics(CFG.dataset_path,Path(CFG.results_dir),fold)
    #wandb.log(td_sdice)
    writer.close()
    run.finish()
    del trainer
    gc.collect()
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--slice_sampling_interval', default=1, type=int)    
    parser.add_argument('--nid', default=3, type=int)
    parser.add_argument('--baseline_exp_path', default=None, type=str)
    parser.add_argument('--aug_type', default=None, type=str)
    parser.add_argument('--rand_aug', default=False, type=bool)
    parser.add_argument('--PIL_tfms', nargs="+", default=None)
    parser.add_argument('--folds', nargs="+", default=None)
    
    args = parser.parse_args()
    
    CFG.n_add_ids=args.nid
    CFG.slice_sampling_interval=args.slice_sampling_interval
    CFG.exp_name=f"exp_ssi_{args.slice_sampling_interval}_nid_{args.nid}_{args.aug_type}"
    CFG.baseline_exp_path=args.baseline_exp_path
    CFG.results_dir=f"spottune_results/{args.baseline_exp_path}_{CFG.exp_name}/"
    CFG.job_type=f"spottune_{args.aug_type}"
    CFG.fold=list(map(int, args.folds))
    print(class2dict(CFG))
    
    for fold in CFG.fold:
        run_fold(fold)

if __name__ == "__main__":
    main()