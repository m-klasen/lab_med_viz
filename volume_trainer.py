from spottunet.dataset.cc359 import *
from spottunet.split import one2all
from spottunet.torch.module.unet import UNet2D
from spottunet.utils import sdice
from dpipe.im.metrics import dice_score

import torch
import torch.nn as nn
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

import matplotlib.pyplot as plt


from graphviz import Digraph
import torch
from torch.autograd import Variable, Function


class Trainer:
    def __init__(
        self,
        CFG,
        model, 
        device, 
        optimizer,
        scheduler,
        criterion,
        writer,
        fold,
        max_norm,
        mixup_fn
    ):
        self.CFG = CFG
        self.mode = fold
        self.writer = writer
        self.device = device
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.max_norm = max_norm

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.fp16 = CFG.fp16
        if self.fp16: 
            self.scaler = GradScaler() 
        else: 
            self.scaler = None
        self.step = 0
            
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        mb = master_bar(range(epochs))
        log = ['Epoch', 'train/loss', 'train/dice', 'valid/loss', 'valid/dice', 'valid/sdice', 'LR']
        mb.write(log, table=True)
        for n_epoch in mb:
            
            train_loss, train_dice = self.train_epoch(train_loader, mb, n_epoch)
            #train_loss, train_dice, train_sdice = 0,0,0
            valid_loss, valid_dice, valid_sdice = self.valid_epoch(valid_loader, mb, n_epoch)
            if self.CFG.scheduler!=torch.optim.lr_scheduler.OneCycleLR:
                self.scheduler.step()  
            
            log = [n_epoch,train_loss, train_dice, valid_loss, valid_dice, valid_sdice, self.optimizer.param_groups[0]["lr"]]
            mb.write([f'{l:.4f}' if isinstance(l, float) else str(l) for l in log], table=True)
                
            # if True:
            #if self.best_valid_score < valid_auc: 
            if self.best_valid_score > valid_loss:
                save_p = f"{save_path}mode_{self.mode}_best_epoch_model.pth"
                self.save_model(n_epoch, save_p, valid_loss)
                self.best_valid_score = valid_loss

            if n_epoch%10==0:
                save_p = f"{save_path}e_{n_epoch}.pth"
                self.save_model(n_epoch, save_p, valid_loss)
            
    def train_epoch(self, train_loader, mb, n_epoch):
        self.model.train()
        sum_loss = 0
        all_dice = []
        for step, (x,targets) in progress_bar(enumerate(train_loader, 1),len(train_loader), parent=mb):
            if self.fp16:
                with autocast():
                    self.optimizer.zero_grad()
                    logits = self.model(x)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()

                if (step + 1) % self.CFG.accumulation_steps == 0:
                    # Grad clipping
                    if self.max_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                       self.max_norm, 
                                                       error_if_nonfinite=False)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.CFG.scheduler==torch.optim.lr_scheduler.OneCycleLR:
                        self.scheduler.step() 
            else:
                self.optimizer.zero_grad()
                logits = self.model(x).squeeze()
                loss = self.criterion(logits, targets)
                loss.backward()
                if (step + 1) % self.CFG.accumulation_steps == 0:
                    # Grad clipping
                    if self.max_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                       self.max_norm, 
                                                       error_if_nonfinite=False)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.CFG.scheduler==torch.optim.lr_scheduler.OneCycleLR:
                        self.scheduler.step()  
            
            
            sum_loss += loss.detach().item()
            
            outputs = torch.sigmoid(logits)
            outs = (outputs > .5).cpu().numpy()
            targs = (targets > .5).cpu().numpy()
            d_score = dice_score(targs,outs)
            
            all_dice.append(d_score)
            mb.child.comment = 'train_loss: {:.4f}, LR: {:.2e}'.format(sum_loss/step, self.optimizer.param_groups[0]['lr'])
        
            ## Debugging
            #self.writer.add_images("X", x, self.step)
            #self.writer.add_images("Y", targets.unsqueeze(1), self.step)
            #self.writer.add_images("Preds", outputs.unsqueeze(1), self.step)
            #for name, weight in self.model.named_parameters():
                #self.writer.add_histogram(name,weight, n_epoch)
                #self.writer.add_histogram(f'{name}.grad',weight.grad, self.step)
            
            
            
            self.writer.add_scalar('train/Loss',loss.detach().item(), self.step)
            self.writer.add_scalar('train/Dice', d_score, self.step)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.step)
            
            #Global Step counter
            self.step += 1

        return sum_loss/len(train_loader), np.mean(all_dice)
    
    def valid_epoch(self, valid_loader,mb, n_epoch, bs=16):
        self.model.eval()
        sum_loss = 0
        all_dice = []
        all_sdice = []
        for step,(x,y,_id) in progress_bar(enumerate(valid_loader, 1), len(valid_loader), parent=mb):
            with torch.no_grad():
                x = x.to(self.device); _id=_id[0]
                y = y.to(self.device)
                bs,c,s,h,w = x.shape
                out = self.model(x)
                loss = self.criterion(out, y)
                
                outputs = torch.sigmoid(out.detach())
                preds = (outputs > .5).cpu().numpy()
                targs = (y > .5).cpu().numpy()
                sum_loss += loss.detach().item()

            d_score = dice_score(targs, preds)
            sd_score = sdice(targs.squeeze(0).squeeze(0), 
                             preds.squeeze(0).squeeze(0), 
                             self.CFG.voxel_spacing,self.CFG.sdice_tolerance)
            
            all_dice.append(d_score);all_sdice.append(sd_score)

        self.writer.add_scalar('valid/Loss', sum_loss/len(valid_loader), n_epoch)
        self.writer.add_scalar('valid/Dice', np.mean(all_sdice), n_epoch)
        self.writer.add_scalar('valid/SDice', np.mean(sd_score), n_epoch)

        
        return sum_loss/len(valid_loader), np.mean(sd_score),np.mean(all_sdice)
    
    def test(self, test_loader, result_path, bs=16):
        self.model.eval()
        metrics = {'sdice_score': sdice, 'dice_score': dice_score}

        results = defaultdict(dict)
        for step,(x,y,_id) in progress_bar(enumerate(test_loader, 1), len(test_loader)):
            with torch.no_grad():
                x = x.to(self.device); _id=_id[0]
                y = y.to(self.device)
                bs,c,s,h,w = x.shape
                out = self.model(x)
                
                outputs = torch.sigmoid(out.detach())
                preds = (outputs > .5).cpu().numpy()
                outputs = (outputs > .5)
                
                y = (y > .5).numpy().squeeze()

            results['sdice_score'][_id] = sdice(y.squeeze(0).squeeze(0), 
                                                outputs.squeeze(0).squeeze(0), 
                                                self.CFG.voxel_spacing,self.CFG.sdice_tolerance)
            results['dice_score'][_id]  = dice_score(y, outputs)

        with open(os.path.join(result_path, 'sdice_score' + '.json'), 'w') as f:
            json.dump(results['sdice_score'], f, indent=0)
        with open(os.path.join(result_path, 'dice_score'+ '.json'), 'w') as f:
            json.dump(results['dice_score'], f, indent=0)
    
    def save_model(self, n_epoch, save_path, loss):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )