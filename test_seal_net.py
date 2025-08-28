# test_seal_net.py
import os
import h5py
import numpy as np
from scipy.interpolate import PPoly, PchipInterpolator
from scipy import sparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from neuralop.models import FNO


model_dir='net/'
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# As per user request, these are the 3 models to choose from.
model_files = {
    1: 'mlp_leak_20250826_T_091719.pth',
    2: 'mlp_leak_20250826_T_093534.pth',
    3: 'mlp_leak_20250826_T_095326.pth'
}

for model_id, filename in model_files.items():
    ckpt_path = os.path.join(model_dir, filename)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']

    params = {
        'hparams': hp,
        'scaler_X_mean': ckpt['scaler_X_mean'],
        'scaler_X_std': ckpt['scaler_X_std'],
        'scaler_y_mean': ckpt['scaler_y_mean'],
        'scaler_y_std': ckpt['scaler_y_std'],
    }
    
    print(params)
    
model_files = {
    1: 'deeponet_multihead_20250826_T_091719.pth',
    2: 'deeponet_multihead_20250826_T_093534.pth',
    3: 'deeponet_multihead_20250826_T_095326.pth'
}

for model_id, filename in model_files.items():
    ckpt_path = os.path.join(model_dir, filename)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']

    params = {
        'hparams': hp,
        'scaler_X_mean': ckpt['scaler_X_mean'],
        'scaler_X_std': ckpt['scaler_X_std'],
        # 'scaler_y_mean': ckpt['scaler_y_mean'],
        # 'scaler_y_std': ckpt['scaler_y_std'],
    }
    
    print(params)