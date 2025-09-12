#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple
from loader_brg_seal import SealDONModel, SealLeakModel
import h5py
from solver_seal import main_seal_solver
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
# Reuse figure sizes and rcparams from the paper plotting module
from make_paper_figures import (
    _default_rcparams,
    figsize_F,
    figsize_SC,
    figsize_DC,
    figsize_DC_b,
)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


#%%

m_rdc = SealDONModel(device=device)
m_leak = SealLeakModel(device=device)

#%%
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
data_dir = 'dataset/data/tapered_seal'


    

fig, axs = plt.subplots(2, 2, figsize=figsize_F)
axs = axs.ravel()
    
    
for seal_idx, mat_file in enumerate(mat_files):
# for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    # print('current file: '+mat_file)
    
    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        input_ = np.array(mat.get('input'))
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:] # [C, c, K, k]
        leak = np.array(mat.get('Leak'))
        leak = leak[6,:].reshape(-1,1)

        n_para, n_data = input_.shape
        _, n_vel = w_vec.shape
        n_rdc_coeffs = rdc.shape[0]

        X_params = input_.T
        y_functions = rdc.transpose(2, 0, 1)
        
        w = w_vec.squeeze()
        rpm = w*30/np.pi

    train_idx = m_rdc.models[seal_idx+1]['train_idx']
    test_idx = m_rdc.models[seal_idx+1]['test_idx']
    
    X = input_.transpose()
    X_train = X[train_idx,:]
    X_test = X[test_idx,:]
    n_w = w.shape[0]
    
    leak_true = leak
    leak_true_train = leak_true[train_idx,:]
    leak_true_test = leak_true[test_idx,:]
    rdc_true = rdc.transpose([2,0,1])
    rdc_true_train = rdc_true[train_idx,:]
    rdc_true_test = rdc_true[test_idx,:]
    
    leak_pred_train = m_leak.predict(seal_idx+1, X_train)
    leak_pred_test = m_leak.predict(seal_idx+1, X_test)
    rdc_pred_train  = m_rdc.predict(seal_idx+1, X_train, w).reshape(X_train.shape[0], 1, 4, n_w).squeeze()
    rdc_pred_test  = m_rdc.predict(seal_idx+1, X_test, w).reshape(X_test.shape[0], 1, 4, n_w).squeeze()
    
    plot_idx = np.random.randint(1,10,1)
    p_rdc_train_true = rdc_true_train[plot_idx,:,:]
    p_rdc_test_true = rdc_true_test[plot_idx,:,:]
    p_rdc_train_pred = rdc_pred_train[plot_idx,:,:]
    p_rdc_test_pred = rdc_pred_test[plot_idx,:,:]
        
    # plot coefficient of determination (R2) of leakage for research paper
    # Compute train/test R^2 for leakage (scalar per sample)
    ytr = leak_true_train.ravel().astype(float)
    ypr = leak_pred_train.ravel().astype(float)
    yte = leak_true_test.ravel().astype(float)
    ype = leak_pred_test.ravel().astype(float)

    y_true = leak_true_train
    y_pred = leak_pred_train
    
    y_true = leak_true_test
    y_pred = leak_pred_test

    mn = np.nanmin([y_true, y_pred])
    mx = np.nanmax([y_true, y_pred])
    # axs[seal_idx].plot([mn, mx], [mn, mx], linestyle='--', linewidth=1, label=r'$y_{\mathrm{pred}}=y_{\mathrm{true}}$')
    axs[seal_idx].plot([mn, mx], [mn, mx], linestyle='--', linewidth=0.75, color='k',label=r'$y_{\mathrm{pred}}=y_{\mathrm{true}}$')

    # 산점도
    axs[seal_idx].scatter(y_true[:,-1], y_pred[:,-1], s=8, alpha=0.5, label='Data')

    # 선형 보정선 y = a x + b (가이드)
    a, b = np.polyfit(y_true[:,-1], y_pred[:,-1], 1)
    axs[seal_idx].plot([mn, mx], 
                       [a*mn + b, a*mx + b], 
                       linewidth=1.5, 
                       label='fit',
                       color="#E88B21")

    # axs[seal_idx].set_xlabel(r'$y_{\mathrm{true}}$')
    # axs[seal_idx].set_ylabel(r'$y_{\mathrm{pred}}$')
    axs[seal_idx].set_xlabel(r"Predicted $\dot{m}$")
    axs[seal_idx].set_ylabel(r"Real $\dot{m}$")
    # axs[seal_idx].set_title('Parity plot (Validation)')
    axs[seal_idx].legend(loc='best')
    axs[seal_idx].grid(True, linestyle=':', linewidth=0.6)
    
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    yrng = (y_true.max() - y_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12)))

    # 지표 텍스트(LaTeX)
    txt = '\n'.join([
        rf'$R^2={r2:.4f}$',
        rf'$\mathrm{{RMSE}}={rmse:.3g}$',
        rf'$\mathrm{{MAE}}={mae:.3g}$',
        rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
        rf'$\mathrm{{MAPE}}={100*mape:.2f}\%$'
    ])
    txt = '\n'.join([
        rf'$R^2={r2:.4f}$',
        rf'$\mathrm{{RMSE}}={rmse:.3g}$',
    ])
    axs[seal_idx].text(0.05, 0.95, txt, transform=axs[seal_idx].transAxes,
            va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
fig.tight_layout()
fig.savefig('valid_leak.png', dpi=600)
fig.show()
    



    # # --- 4) 잔차 히스토그램 (y_pred - y_true) ---
    # res = y_pred - y_true
    # mu, sigma = np.mean(res), np.std(res)

    # ax[1].hist(res, bins=30, alpha=0.8, edgecolor='k')
    # ax[1].axvline(0.0, color='k', linestyle='--', linewidth=1)
    # ax[1].axvline(mu, color='r', linestyle='-', linewidth=1, label=rf'$\mu={mu:.3g}$')
    # ax[1].axvline(mu+2*sigma, color='r', linestyle=':', linewidth=1, label=rf'$\mu\pm2\sigma$')
    # ax[1].axvline(mu-2*sigma, color='r', linestyle=':', linewidth=1)

    # ax[1].set_xlabel(r'$\mathrm{residual}=y_{\mathrm{pred}}-y_{\mathrm{true}}$')
    # ax[1].set_ylabel('count')
    # ax[1].set_title('Residual histogram (Validation)')
    # ax[1].legend(loc='best')
    # ax[1].grid(True, linestyle=':', linewidth=0.6)

    # plt.tight_layout()
    # plt.show()
    
            
    # _default_rcparams()
    # fig, axs = plt.subplots(2,2,figsize=figsize_SC)
    # axs = axs.ravel()
    # for i in range(4):
    #     axs[i].plot(rpm,p_rdc_train_true[:,i,:].transpose(),'-', label='Solver')
    #     axs[i].plot(rpm,p_rdc_train_pred[:,i,:].transpose(),':', label='DeepONet')
    #     axs[i].set_xlabel('Rotating speed (RPM)')
    #     axs[i].grid(True, alpha=0.3)
    # fig.tight_layout()
    # fig.show()

# %%
