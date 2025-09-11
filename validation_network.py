#%%
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loader_brg_seal import SealDONModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
model_seal = SealDONModel(device=device)

for seal_idx, mat_file in enumerate(mat_files):
# for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # inputNond: [nPara, nData] 형상 파라미터
        input_nond = np.array(mat.get('inputNond'))
        # input: [nPara, nData] [*1e6 *1e6 *1e1]
        input_ = np.array(mat.get('input'))
        # wVec: [1, nVel] 회전 속도 벡터 (좌표 그리드)
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:] # no mass

        n_para, n_data = input_.shape
        _, n_vel = w_vec.shape
        n_rdc_coeffs = rdc.shape[0] # 6 

        # 입력 데이터 (X): 형상 파라미터 [nData, nPara]
        X_params = input_.T

        # 출력 데이터 (y): 동특성 계수 함수 [nData, nVel, nRDC]
        # FNO는 (batch, channels, grid_points) 형태를 선호하므로 [nData, nRDC, nVel]로 변경이라고 GPT가 그럔다
        y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]

        # 회전 속도 그리드: [nVel, 1]
        
        w = w_vec.squeeze()                         # [n_vel]
        w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0 # normalization
        grid = w_norm[:, None] # [nVel, 1]
        # grid = w[:, None] # [nVel, 1]

    X = input_.transpose()
    pop = X.shape[0]
    n_w = w.shape[0]
    rdc_flat  = model_seal.predict(seal_idx+1, X, w).reshape(pop, 1, 4, n_w).squeeze()
    rdc_true = rdc.transpose([2,0,1])

    rdc_flat_ = np.ravel(rdc_flat)
    rdc_true_ = np.ravel(rdc_true)

    mse = mean_squared_error(rdc_true_, rdc_flat_)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(rdc_true_, rdc_flat_)
    r2 = r2_score(rdc_true_, rdc_flat_)
    yrng = (rdc_true_.max() - rdc_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((rdc_true_ - rdc_flat_) / (np.abs(rdc_true_) + 1e-12)))

    print(f"[Overall] RMSE: {rmse:.6g}, MAE: {mae:.6g}, "
        f"R^2: {r2:.6f}, rRMSE: {100*rrmse:.4f}%, MAPE: {100*mape:.4f}%")


    abs_err = rdc_true-rdc_flat
    rel_err = (rdc_true-rdc_flat)/np.abs(rdc_true)*1e2

    for r in range(4):
        fig = plt.figure()
        
        abs_err_ = abs_err[:,r]
        rel_err_ = rel_err[:,r]
        
        idx_rel_sorted = np.argmax(np.abs(rel_err_),axis=1)
        rows = np.arange(rel_err_.shape[0])
        rels = rel_err_[rows, idx_rel_sorted]
        abss = abs_err_[rows, idx_rel_sorted]
        
        idx_rel_sorted = np.flip(np.argsort(np.abs(rels)))
        relss = rels[idx_rel_sorted]
        absss = abss[idx_rel_sorted]
        
        
        plt.subplot(311)
        plt.plot(absss[:100],'o-')
        plt.title("absoulte error")
        plt.subplot(312)
        plt.plot(relss[:100],'o-')
        plt.ylim([-10, 10])
        plt.title("relative error")
        plt.subplot(313)
        plt.plot(relss[:100],'o-')
        plt.title("relative error")
        plt.tight_layout()
        plt.show()
        
        
    # abs_err = rdc_true-rdc_flat
    # rel_err = (rdc_true-rdc_flat)/np.abs(rdc_true)*1e2
    # idx_w = -1
    # for r in range(4):
    #     fig = plt.figure()
    #     rdc_true1 = rdc_true[:,r,idx_w]
    #     r_e = rel_err[:,r,idx_w]
    #     a_e = abs_err[:,r,idx_w]
    #     idx_rel_sorted = np.flip(np.argsort(np.abs(r_e)))
    #     r_e_sorted = r_e[idx_rel_sorted]
    #     a_e_sorted = a_e[idx_rel_sorted]

    #     plt.subplot(411)
    #     plt.plot(rdc_true1[:100],'o-')
    #     plt.title("absoulte value")
    #     plt.subplot(412)
    #     plt.plot(a_e_sorted[:100],'o-')
    #     plt.title("absoulte error")
    #     plt.subplot(413)
    #     plt.plot(r_e_sorted[:100],'o-')
    #     plt.ylim([-10, 10])
    #     plt.title("relative error")
    #     plt.subplot(414)
    #     plt.plot(r_e_sorted[:100],'o-')
    #     plt.title("relative error")
    #     plt.tight_layout()
        

# %%
