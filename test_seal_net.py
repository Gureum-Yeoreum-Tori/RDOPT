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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loader_brg_seal import SealFNOModel, SealDONModel, SealLeakModel

model_dir='net/'
data_dir='dataset/data/tapered_seal'
device = ('cuda' if torch.cuda.is_available() else 'cpu')
w_vec = np.linspace(500*np.pi/30, 6000*np.pi/30, 12)

mat_files = {
    1: '20250826_T_091719',
    2: '20250826_T_093534',
    3: '20250826_T_095326'
}
# As per user request, these are the 3 models to choose from.
model_files = {
    1: 'mlp_leak_20250826_T_091719.pth',
    2: 'mlp_leak_20250826_T_093534.pth',
    3: 'mlp_leak_20250826_T_095326.pth'
}

seal_fno = SealFNOModel()
seal_don = SealDONModel()
seal_leak = SealLeakModel()

for model_id, filename in mat_files.items():
    data_path = os.path.join(data_dir, filename,'dataset.mat')
    data = {}
    with h5py.File(data_path, 'r') as mat:
        input_ = np.array(mat.get('input')) # 실차원 기준
        leak = np.array(mat.get('Leak'))
        rdc = np.array(mat.get('RDC'))
        
        leak_true = leak[6,:].reshape(-1,1)
        rdc_true = rdc[[4, 5, 2, 3],:,:] # C c K k
        
    ## test models
    leak_pred = seal_leak.predict(model_id,input_.transpose())
    rdc_pred_fno = seal_fno.predict(model_id,input_.transpose(),w_vec)
    rdc_pred_don = seal_don.predict(model_id,input_.transpose(),w_vec)
    
    rmse = np.sqrt(mean_squared_error(leak_true, leak_pred))
    mae  = mean_absolute_error(leak_true, leak_pred)
    r2   = r2_score(leak_true, leak_pred)
    yrng = (leak_true.max() - leak_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((leak_true - leak_pred) / (np.abs(leak_true) + 1e-12)))
    txt = '\n'.join([
        rf'$R^2={r2:.4f}$',
        rf'$\mathrm{{RMSE}}={rmse:.3g}$',
        rf'$\mathrm{{MAE}}={mae:.3g}$',
        rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
        rf'$\mathrm{{MAPE}}={100*mape:.2f}\%$'
    ])
    print(txt)
    
    
# for model_id, filename in model_files.items():
#     ckpt_path = os.path.join(model_dir, filename)
    
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
#     hp = ckpt['hparams']

#     params = {
#         'hparams': hp,
#         'scaler_X_mean': ckpt['scaler_X_mean'],
#         'scaler_X_std': ckpt['scaler_X_std'],
#         'scaler_y_mean': ckpt['scaler_y_mean'],
#         'scaler_y_std': ckpt['scaler_y_std'],
#     }
    
#     print(params)
    
# model_files = {
#     1: 'deeponet_multihead_20250826_T_091719.pth',
#     2: 'deeponet_multihead_20250826_T_093534.pth',
#     3: 'deeponet_multihead_20250826_T_095326.pth'
# }

# for model_id, filename in model_files.items():
#     ckpt_path = os.path.join(model_dir, filename)
    
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
#     hp = ckpt['hparams']

#     params = {
#         'hparams': hp,
#         'scaler_X_mean': ckpt['scaler_X_mean'],
#         'scaler_X_std': ckpt['scaler_X_std'],
#         # 'scaler_y_mean': ckpt['scaler_y_mean'],
#         # 'scaler_y_std': ckpt['scaler_y_std'],
#     }
    
#     print(params)
