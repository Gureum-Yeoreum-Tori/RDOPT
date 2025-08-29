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
from matplotlib import pyplot as plt


def evaluate(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


model_dir='net/'
data_dir='dataset/data/tapered_seal'
device = ('cuda' if torch.cuda.is_available() else 'cpu')
w_vec = np.linspace(500*np.pi/30, 6000*np.pi/30, 12)

mat_files = {
    1: '20250826_T_091719',
    # 2: '20250826_T_093534',
    # 3: '20250826_T_095326'
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
        n_data = input_.shape[1]
        
        leak_true = leak[6,:].reshape(-1,1)
        rdc_true = rdc[2:6,:,:].transpose(2, 0, 1) # C c K k
        
    ## test models
    leak_pred = seal_leak.predict(model_id,input_.transpose())
    rdc_pred_fno = seal_fno.predict(model_id,input_.transpose(),w_vec) # (n_case,n_rdc=4,n_w=12)
    rdc_pred_don = seal_don.predict(model_id,input_.transpose(),w_vec)
    
    rmse = np.sqrt(mean_squared_error(leak_true, leak_pred))
    mae  = mean_absolute_error(leak_true, leak_pred)
    r2   = r2_score(leak_true, leak_pred)
    yrng = (leak_true.max() - leak_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((leak_true - leak_pred) / (np.abs(leak_true) + 1e-12)))
    
    metrics_leak = evaluate(leak_true, leak_pred)
    metrics_fno = evaluate(rdc_true, rdc_pred_fno)
    metrics_don = evaluate(rdc_true, rdc_pred_don)
    
    print("Leak  - RMSE: %.5f, MAE: %.5f, R²: %.4f" % metrics_leak)
    print("FNO  - RMSE: %.5f, MAE: %.5f, R²: %.4f" % metrics_fno)
    print("DeepONet - RMSE: %.5f, MAE: %.5f, R²: %.4f" % metrics_don)
    
    index_ = range(0,n_data)
    
    err = (leak_true-leak_pred)/(leak_true+1e-12)*100
    plt.figure()
    plt.plot(index_,err)
    plt.show()

    err1 = np.mean((rdc_true - rdc_pred_fno) / (rdc_true + 1e-12) * 100, axis=2)
    err1 = np.max(np.abs(rdc_true - rdc_pred_fno), axis=2)
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True)
    axes = axes.ravel()
    
    for i in range(4):
        ax = axes[i]
        ax.plot(index_,err1[:,i], marker='o')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(f"RDC {i}")
        ax.set_xlabel("ω index")
        if i == 0:
            ax.set_ylabel("Mean Rel. Error [%]")
        ax.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()
    
    err2 = np.mean((rdc_true - rdc_pred_don) / (rdc_true + 1e-12) * 100, axis=2)
    err2 = np.max(np.abs(rdc_true - rdc_pred_don), axis=2)
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True)
    axes = axes.ravel()
    
    for i in range(4):
        ax = axes[i]
        ax.plot(index_,err2[:,i], marker='o')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(f"RDC {i}")
        ax.set_xlabel("index")
        if i == 0:
            ax.set_ylabel("Mean Rel. Error [%]")
        ax.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()
    
    k = 1
    idx_bad = []
    for i in range(4):
        idx_bad.append(np.argsort(np.abs(err1[:,i]))[-k:])
    for i in range(4):
        idx_bad.append(np.argsort(np.abs(err2[:,i]))[-k:])
    idx_bad = np.concatenate(idx_bad)
    print("큰 오차 케이스 인덱스:", idx_bad)
    
    for idx in idx_bad:
        fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True)
        axes = axes.ravel()
        print(f"ω index {idx}")
        for i in range(4):
            ax = axes[i]
            ax.plot(w_vec*30/np.pi,rdc_true[idx,i],label='true')
            ax.plot(w_vec*30/np.pi,rdc_pred_fno[idx,i],label='pred_fno')
            ax.plot(w_vec*30/np.pi,rdc_pred_don[idx,i],label='pred_don')
        plt.tight_layout()
        plt.legend()
        plt.show()
    
    
    
    
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
