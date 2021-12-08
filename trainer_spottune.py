
from spottunet.utils import sdice
from dpipe.im.metrics import dice_score

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 

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

import torch.nn.functional as F
from torch.autograd import Variable

from dpipe.torch import get_device

def gumbel_softmax(logits, use_gumbel=True, temperature=5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, use_gumbel=use_gumbel)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    if device == torch.device('cpu'):
        u = torch.rand(shape, requires_grad=False, device=device)
    else:
        u = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(u + eps) + eps))


def gumbel_softmax_sample(logits, temperature, use_gumbel=True):
    y = logits + sample_gumbel(logits.size(), device=get_device(logits)) if use_gumbel else logits
    return F.softmax(y / temperature, dim=-1)


def reg_policy(policy, k, mode='l1'):
    if mode == 'l1':
        reg = k * (1 - policy).sum() / torch.numel(policy)  # shape(policy) [batch_size, n_blocks]
    elif mode == 'l2':
        reg = k * torch.sqrt(((1 - policy) ** 2).sum()) / torch.numel(policy)
    else:
        raise ValueError(f'`mode` should be either `l1` or `l2`; but `{mode}` is given')
    return reg

class SpotTuneTrainer:
    def __init__(
        self,
        CFG,
        model,
        model_policy,
        device, 
        optimizer_main,
        scheduler_main,
        optimizer_policy,
        scheduler_policy,
        criterion,
        writer,
        fold,
        max_norm,
        temperature,
        k_reg,
        reg_mode
        
    ):
        self.CFG = CFG
        self.model = model
        self.model_policy = model_policy
        self.device = device
        self.optimizer_main = optimizer_main
        self.optimizer_policy = optimizer_policy
        self.scheduler_main = scheduler_main
        self.scheduler_policy = scheduler_policy
        self.criterion = criterion
        self.writer = writer
        self.mode = fold
        self.max_norm = max_norm
        self.clip_grads = False
        self.temperature = temperature
        self.k_reg = k_reg
        self.reg_mode = reg_mode
        
        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.fp16 = self.CFG.fp16
        if self.fp16: 
            self.scaler = GradScaler() 
        else: 
            self.scaler = None
            
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        mb = master_bar(range(epochs))
        log = ['Epoch', 'train/loss', 'train/dice', 'train/sdice', 'valid/loss', 'valid/dice', 'valid/sdice', 'LR']
        mb.write(log, table=True)
        for n_epoch in mb:
            
            train_loss, train_dice, train_sdice = self.train_epoch(train_loader, mb, n_epoch)
            valid_loss, valid_dice, valid_sdice = self.valid_epoch(valid_loader, mb, n_epoch)
            self.scheduler_main.step()  
            self.scheduler_policy.step()  
            log = [n_epoch,train_loss, train_dice, train_sdice, 
                   valid_loss, valid_dice, valid_sdice, 
                   self.optimizer_main.param_groups[0]["lr"]]
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
        self.model.save_policy('policy_training_record', self.CFG.results_dir+"/mode_"+str(self.mode))
            
    def train_epoch(self, train_loader, mb, n_epoch, with_source=False):
        self.model.train()
        self.model_policy.train()
        sum_loss = 0
        all_dice = []
        all_sdice = []
        if with_source:
            assert "not defined"
        policies = []
        for step, (x,y) in progress_bar(enumerate(train_loader, 1),len(train_loader), parent=mb):
            x = x.to(self.device)
            targets = y.to(self.device)
            
            with autocast():
                self.optimizer_main.zero_grad()
                self.optimizer_policy.zero_grad()

                probs = self.model_policy(x)  # [32, 16]
                action = gumbel_softmax(probs.view(probs.size(0), -1, 2), temperature=self.temperature)  # [32, 8, 2]
                policy = action[:, :, 1]  # [32, 8]
                
                outputs = self.model(x,policy).squeeze()
                
                
            loss = self.criterion(outputs, targets) + reg_policy(policy=policy, k=self.k_reg, mode=self.reg_mode)

            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            sum_loss += loss.detach().item()
            
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.model_policy.parameters(), 0.25)            
            if (step + 1) % self.CFG.accumulation_steps == 0:
                self.scaler.step(self.optimizer_main)
                self.scaler.step(self.optimizer_policy)
                self.scaler.update()

            outputs = torch.sigmoid(outputs)
            outs = (outputs > .5).cpu().numpy()
            targs = (targets > .5).cpu().numpy()
            d_score = dice_score(outs,targs)
            sd_score = sdice(outs, targs, self.CFG.voxel_spacing,self.CFG.sdice_tolerance)
            assert sd_score != np.nan, "NaN sd_score"
            all_dice.append(d_score);all_sdice.append(sd_score)
            mb.child.comment = 'train_loss: {:.4f}, LR: {:.2e}'.format(sum_loss/step, self.optimizer_main.param_groups[0]['lr'])
            
        self.writer.add_scalar('train/Loss',loss.detach().item(), n_epoch)
        self.writer.add_scalar('train/Dice', d_score, n_epoch)
        self.writer.add_scalar('train/SDice', sd_score, n_epoch)
        self.writer.add_scalar('LR', self.optimizer_main.param_groups[0]['lr'], n_epoch)

        return sum_loss/len(train_loader), np.mean(all_dice),np.mean(all_sdice)
    
    def valid_epoch(self, valid_loader,mb, n_epoch, bs=16):
        self.model.eval()
        self.model_policy.eval()
        self.model.val_flag = True
        sum_loss = 0
        all_dice = []
        all_sdice = []
        for step,(x,y,_id) in progress_bar(enumerate(valid_loader, 1), len(valid_loader), parent=mb):
            with torch.no_grad():
                x = x[0].to(self.device); _id=_id[0]
                y = y[0].to(self.device)
                c,h,w = x.shape
                outputs = []
                for idx in range(0,c,bs):
                    inp = x[idx:min(idx+bs,c)].unsqueeze(1)
                    probs = self.model_policy(inp)
                    action = gumbel_softmax(probs.view(probs.size(0), -1, 2), use_gumbel=False, 
                                                                              temperature=self.temperature)
                    policy = action[:, :, 1]
                    out = self.model(inp, policy).squeeze()
                    if len(out.shape)==2: out = out[None,:,:]
                    outputs.append(out)
                logits = torch.cat(outputs)
                loss = self.criterion(logits, y)
                
                outputs = torch.sigmoid(logits.detach())
                preds = (outputs > .5).cpu().numpy()
                targs = (y > .5).cpu().numpy()
                sum_loss += loss.detach().item()

            d_score = dice_score(targs, preds)
            sd_score = sdice(targs, preds, self.CFG.voxel_spacing,self.CFG.sdice_tolerance)
            
            all_dice.append(d_score);all_sdice.append(sd_score)

        self.writer.add_scalar('valid/Loss', sum_loss/len(valid_loader), n_epoch)
        self.writer.add_scalar('valid/Dice', np.mean(all_sdice), n_epoch)
        self.writer.add_scalar('valid/SDice', np.mean(sd_score), n_epoch)
        train_policies = self.model.get_train_stats()
        valid_policies = self.model.get_val_stats()
        for n in train_policies.keys():
            self.writer.add_scalar(n,train_policies[n],n_epoch)
        for n in valid_policies.keys():
            self.writer.add_scalar(n,valid_policies[n],n_epoch)
        self.model.val_flag = False
        return sum_loss/len(valid_loader), np.mean(sd_score),np.mean(all_sdice)
    
    def test(self, test_loader, result_path, bs=16):
        self.model.eval()
        self.model_policy.eval()
        metrics = {'sdice_score': sdice, 'dice_score': dice_score}

        results = defaultdict(dict)
        for step,(x,y,_id) in progress_bar(enumerate(test_loader, 1), len(test_loader)):
            with torch.no_grad():
                x = x[0].to(self.device); _id=_id[0]
                c,h,w = x.shape
                outputs = []
                for idx in range(0,c,bs):
                    inp = x[idx:min(idx+bs,c)].unsqueeze(1)
                    probs = self.model_policy(inp)
                    action = gumbel_softmax(probs.view(probs.size(0), -1, 2), use_gumbel=False, 
                                                                              temperature=self.temperature)
                    policy = action[:, :, 1]
                    out = self.model(inp,policy)
                    out = out.squeeze().cpu().detach().numpy()
                    if len(out.shape)==2: out = out[None,:,:]
                    outputs.append(out)
                outputs = np.concatenate(outputs)
                outputs = (outputs > .5)
                y = (y > .5).numpy().squeeze()

            results['sdice_score'][_id] = sdice(y, outputs, self.CFG.voxel_spacing,self.CFG.sdice_tolerance)
            results['dice_score'][_id]  = dice_score(y, outputs)

        with open(os.path.join(result_path, 'sdice_score' + '.json'), 'w') as f:
            json.dump(results['sdice_score'], f, indent=0)
        with open(os.path.join(result_path, 'dice_score'+ '.json'), 'w') as f:
            json.dump(results['dice_score'], f, indent=0)
        self.model.save_policy('policy_inference_record', self.CFG.results_dir+"/mode_"+str(self.mode))
            
    def freeze_model_spottune(self, model):
        for name, param in model.named_parameters():
            if 'freezed' in name:
                requires_grad = False
            else:
                requires_grad = True
            param.requires_grad = requires_grad    

    def unfreeze_model(model):
        for params in model.parameters():
            params.requires_grad = True
        
    def save_model(self, n_epoch, save_path, loss):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_policy_state_dict": self.model_policy.state_dict(),
                "optimizer_main_state_dict": self.optimizer_main.state_dict(),
                "optimizer_policy_main_state_dict": self.optimizer_policy.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )