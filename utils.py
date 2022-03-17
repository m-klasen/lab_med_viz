import numpy as np
import pandas as pd
import os

from torch.optim import Adam, SGD
from pytorch_lamb import Lamb

from dpipe.io import load
from dpipe.torch import load_model_state, get_device

from copy import deepcopy

import torch
import torch.nn as nn

def get_target_domain_metrics(dataset_path,path_base,s):
    meta = pd.read_csv(f"{dataset_path}/meta.csv",delimiter=",", index_col='id')
    res_row = {}
    
    # one2all results:
    sdices = load(path_base / f'mode_{s}/sdice_score.json')
    #sdices = dict(sorted(sdices.items()))
    for t in sorted(set(meta['fold'].unique()) - {s}):
        df_row = meta[meta['fold'] == t].iloc[0]
        target_name = df_row['tomograph_model'] + str(df_row['tesla_value'])
        
        ids_t = meta[meta['fold'] == t].index
        res_row[target_name] = np.mean([sdsc for _id, sdsc in sdices.items() if _id in ids_t])
    return res_row


def freeze_model_spottune(model):
    for name, param in model.named_parameters():
        if 'freezed' in name:
            requires_grad = False
        else:
            requires_grad = True
        param.requires_grad = requires_grad

def load_model_state(module: nn.Module, path, modify_state_fn) -> nn.Module:
    """
    Updates the ``module``'s state dict by the one located at ``path``.

    Parameters
    ----------
    module
    path
    modify_state_fn: Callable(current_state, loaded_state)
        if not ``None``, two arguments will be passed to the function:
        current state of the model and the state loaded from the path.
        This function should modify states as needed and return the final state to load.
        For example, it could help you to transfer weights from similar but not completely equal architecture.
    """
    state_to_load = torch.load(path, map_location=get_device(module))['model_state_dict']
    if modify_state_fn is not None:
        current_state = module.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    module.load_state_dict(state_to_load)
        
def load_model_state_fold_wise(architecture, baseline_exp_path, exp, n_folds=6, modify_state_fn=None, n_first_exclude=0):
    n_val = int(exp) + n_first_exclude
    path_to_pretrained_model = os.path.join(baseline_exp_path, f'mode_{n_val // (n_folds - 1)}', 
                                            f'e_39.pth')
    load_model_state(architecture, path=path_to_pretrained_model, modify_state_fn=modify_state_fn)



def modify_state_fn_spottune(current_state, state_to_load, init_random=False):
    add_str = '_freezed'
    state_to_load_parallel = deepcopy(state_to_load)
    for key in state_to_load.keys():
        a = key.split('.')
        a[0] = a[0] + add_str
        a = '.'.join(a)
        value_to_load = torch.rand(state_to_load[key].shape).to(state_to_load[key].device) if init_random else \
                        state_to_load[key]
        state_to_load_parallel[a] = value_to_load
    return state_to_load_parallel


def get_optimizer(model, optim='Lamb',lr=1e-4,weight_decay=1e-2):
    if optim=="Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optim=="SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9,nesterov=True,weight_decay=0)
    elif optim=="Lamb":
        optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(.9, .999), adam=False)
    return optimizer