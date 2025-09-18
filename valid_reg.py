#%%
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from loader_brg_seal import SealDONModel, SealLeakModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from time import time as tt
import math
import matplotlib.lines as mlines



device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# np.random.seed(42)
def _default_rcparams():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1
    })
figsize_F = (6.8,4.2)



m_rdc = SealDONModel(device=device)
m_leak = SealLeakModel(device=device)
data_dir = 'dataset/data/tapered_seal'
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
mat_files = ('20250908_T_203220',)
_default_rcparams()
for seal_idx, mat_file in enumerate(mat_files):
    seal_idx = 3
# for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    train_idx = m_rdc.models[seal_idx+1]['train_idx']
    test_idx = m_rdc.models[seal_idx+1]['test_idx']

    print('current file: '+mat_file)
    
    base_dir = 'net'
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, 'deeponet_seal_best_multihead_'+mat_file+'.pth')
    network_path = os.path.join(base_dir, 'deeponet_multihead_'+mat_file+'.pth')
    network_path_ts = os.path.join(base_dir, 'deeponet_multihead_'+mat_file+'.pt')
    loss_hist_path_png = os.path.join('deeponet_multihead_'+mat_file+'.png')
    loss_hist_path_eps = os.path.join('deeponet_multihead_'+mat_file+'.eps')

    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # inputNond: [nPara, nData] 형상 파라미터
        input_ = np.array(mat.get('input'))
        # wVec: [1, nVel] 회전 속도 벡터 (좌표 그리드)
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        Leak = np.array(mat.get('Leak'))[6,:].reshape(-1,1)
        # RDC: [6, nVel, nData] 동특성 계수 (타겟 함수)
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:] # no mass
        n_para, n_data = input_.shape
        _, n_w = w_vec.shape
        n_rdc_coeffs = rdc.shape[0] # 6 

        # 입력 데이터 (X): 형상 파라미터 [nData, nPara]
        X_params = input_.T

        # 출력 데이터 (y): 동특성 계수 함수 [nData, nVel, nRDC]
        y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]
        y_functions = y_functions[:,[2, 3, 0, 1],:] # C, c, K, k -> K, k, C, c

        # 회전 속도 그리드: [nVel, 1]
        w = w_vec.squeeze()                         # [n_vel]
        w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0 # normalization
        grid = w_norm[:, None] # [nVel, 1]
        # grid = w[:, None] # [nVel, 1]
        rpm = w*30/np.pi

        n_data, n_rdc, n_w = y_functions.shape
        # X_tr_case, X_te_case, y_tr_case, y_te_case = train_test_split(
        #     X_params, y_functions, test_size=0.2, random_state=42
        # )
        X_tr_case = X_params[train_idx,:]
        X_te_case = X_params[test_idx,:]
        y_tr_case = y_functions[train_idx,:]
        y_te_case = y_functions[test_idx,:]
        
        # train flatten
        n_tr = X_tr_case.shape[0]
        X_tr = np.repeat(X_tr_case, n_w, axis=0)
        w_tr = np.tile(w.reshape(-1,1), (n_tr,1))
        X_tr = np.hstack([X_tr, w_tr])
        y_tr = y_tr_case.transpose(0,2,1).reshape(-1, n_rdc)

        # test flatten
        n_te = X_te_case.shape[0]
        X_te = np.repeat(X_te_case, n_w, axis=0)
        w_te = np.tile(w.reshape(-1,1), (n_te,1))
        X_te = np.hstack([X_te, w_te])
        y_te = y_te_case.transpose(0,2,1).reshape(-1, n_rdc)
        
        plot_idx = np.random.randint(0,n_te,5)
        # plot_idx = [241,  34, 214,  89, 263]
        plot_idx = [276, 140, 89, 244, 35]
        
        rdc_pred_test  = m_rdc.predict(seal_idx+1, X_te_case, w).reshape(X_te_case.shape[0], 1, 4, n_w).squeeze()
        rdc_pred_test = rdc_pred_test[:,[2, 3, 0, 1],:] # C, c, K, k -> K, k, C, c
        rdc_pred_test_plot = rdc_pred_test[plot_idx,:,:]
        ylabels = ['K $(N/m)$', 'k $(N/m)$','C ($N \cdot s/m$)', 'c ($N \cdot s/m$)']

        # 계수별 타깃 벡터 생성 예: j=0 -> 'C'
        def train_eval(model):
            fig, axes = plt.subplots(2,2,figsize=figsize_F)
            axes = axes.ravel()
            for j in range(4):
                ax = axes[j]
                y_j = y_tr[:,j]
                y_te_ = y_te[:,j]
                y_te_i = y_te_.reshape(-1,n_w)
                model.fit(X_tr, y_j)
                t0 = tt()
                # X_te_ = np.repeat(X_te,axis=0,repeats=4)
                # _ = model.predict(X_te_).ravel()
                pred = model.predict(X_te).ravel()
                t1 = tt()
                print(f'elapse={t1-t0}')
                pred_ = pred.reshape(-1,n_w)
                
                ax.plot(rpm,y_te_i[plot_idx,:].T, 'o',
                    alpha=0.7, markerfacecolor='none', markersize=4)
                ax.set_prop_cycle(None)
                ax.plot(rpm,pred_[plot_idx,:].T,'*--',
                    alpha=0.7, markersize=4)
                ax.set_prop_cycle(None)
                ax.plot(rpm,rdc_pred_test_plot[:,j,:].T,'-')
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax.grid(True, linestyle=':', linewidth=0.6)
                ax.set_xlabel("Rotating speed (rpm)")
                ax.set_ylabel(ylabels[j])
                # ax.legend()
                # fig.tight_layout()

                y_true_ = y_te_.ravel()
                y_pred_ = rdc_pred_test[:,j,:].ravel()
                rmse = np.sqrt(mean_squared_error(y_true_, y_pred_))
                mae  = mean_absolute_error(y_true_, y_pred_)
                r2   = r2_score(y_true_, y_pred_)
                
                n = len(y_true_)
                num = np.sum(np.square(y_true_ - y_pred_)) / n  # update
                den = np.sum(np.square(y_pred_))
                squared_error = num/den
                rrmse = np.sqrt(squared_error)

                yrng = (y_true_.max() - y_true_.min())
                # rrmse = rmse / (yrng + 1e-12)
                mape = np.mean(np.abs((y_true_ - y_pred_) / (np.abs(y_true_) + 1e-12)))
                
                txt = '\n'.join([
                    rf'$R^2={r2:.4f}$',
                    # rf'$\mathrm{{RMSE}}={rmse:.3g}$',
                    # rf'$\mathrm{{MAE}}={mae:.3g}$',
                    rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
                    rf'$\mathrm{{MAPE}}={100*mape:.2f}\%$'
                ])
                
                print(f"DeepONet RMSE={rmse:.4g}  R^2={r2:.4f}  rRMSE={100*rrmse:.2f}  MAPE={100*mape:.2f}")
                
                # ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5), fontsize=8)
                    

                r2 = r2_score(y_te_, pred)

                mse = mean_squared_error(y_te_, pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_te_, pred)
                
                n = len(y_te_)
                num = np.sum(np.square(y_te_ - pred)) / n  # update
                den = np.sum(np.square(pred))
                squared_error = num/den
                rrmse = np.sqrt(squared_error)
                
                yrng = (y_te_.max() - y_te_.min())
                mape = np.mean(np.abs((y_te_ - pred) / (np.abs(y_te_) + 1e-12)))
                
                print(f"RF RMSE={rmse:.4g}  R^2={r2:.4f}  rRMSE={100*rrmse:.2f}  MAPE={100*mape:.2f}\n\n")
                
                # plt.figure()
                # plt.plot(y_te_,y_te_,'-k',label=r'$y_{true}=y_{pred}$')
                # plt.scatter(y_te_,pred,label='predicted')
                # plt.legend()
                # plt.show()
            true_handle = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markerfacecolor='none', label='True value', markersize=4)
            rf_handle = mlines.Line2D([], [], color='k', marker='*',linestyle='--',label='Random Forest')
            pred_handle = mlines.Line2D([], [], color='k', linestyle='-',label='DeepONet')
            fig.legend(handles=[true_handle, rf_handle, pred_handle],
                    loc='lower center', ncol=3,
                    bbox_to_anchor=(0.5, -0.06))   
            fig.tight_layout()
            # fig.savefig(f'valid_rdc_vs_rf_{seal_idx}.png', dpi=600, bbox_inches='tight')
            fig.savefig(f'valid_rdc_vs_ridge_{seal_idx}.png', dpi=600, bbox_inches='tight')
            fig.show()
            return rmse, r2, mae, rrmse, mape

        ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
        svr   = Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01))])
        krr   = Pipeline([("scaler", StandardScaler()), ("reg", KernelRidge(kernel="rbf", alpha=1e-2, gamma=None))])
        rf    = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=0)

        # for name, mdl in [("Ridge", ridge), ("SVR", svr), ("KRR", krr), ("RF", rf)]:
        # for name, mdl in [("SVR", svr)]:
        # for name, mdl in [("KRR", krr)]:
        # for name, mdl in [("RF", rf)]:
        for name, mdl in [("Ridge", ridge)]:
            print(f"{name:>5s}")
            rmse, r2, mae, rrmse, mape = train_eval(mdl)
            # print(f"RMSE={rmse:.4g}  R^2={r2:.4f}  rRMSE={100*rrmse:.2f}  MAPE={100*mape:.2f}\n\n")
            

            
            # from sklearn.gaussian_process import GaussianProcessRegressor
            # from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

            # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*(X_joint.shape[1]), length_scale_bounds=(1e-3, 1e3))
            # gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=0)
            # gpr.fit(X_tr, y_tr.ravel())
            # mu, std = gpr.predict(X_te, return_std=True)
            # print("GPR  RMSE=%.4g  R^2=%.4f" % (math.sqrt(mean_squared_error(y_te, mu)), r2_score(y_te, mu)))
# %%
