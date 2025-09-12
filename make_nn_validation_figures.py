#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple
from loader_brg_seal import SealDONModel, SealLeakModel
import h5py
from solver_seal import main_seal_solver


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

mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
data_dir = 'dataset/data/tapered_seal'



for seal_idx, mat_file in enumerate(mat_files):
# for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        input_nond = np.array(mat.get('inputNond'))
        input_ = np.array(mat.get('input'))
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:]
        leak = np.array(mat.get('Leak'))

        n_para, n_data = input_.shape
        _, n_vel = w_vec.shape
        n_rdc_coeffs = rdc.shape[0]

        X_params = input_.T
        y_functions = rdc.transpose(2, 0, 1)
        
        w = w_vec.squeeze()

    X = input_.transpose()
    pop = X.shape[0]
    n_w = w.shape[0]
    rdc_flat  = m_rdc.predict(seal_idx+1, X, w).reshape(pop, 1, 4, n_w).squeeze()
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
        


#%%

# try:
#     from solver_seal import main_seal_solver
# except Exception:
#     main_seal_solver = None  # type: ignore


# def _rpm(w: np.ndarray) -> np.ndarray:
#     return np.asarray(w) * 30.0 / np.pi


# def _paper_cases() -> List[Tuple[float, float, float]]:
#     """Return a small set of representative seal parameter cases in SI units.
#     Each tuple is (hIn [m], hOut [m], psr [-]).
#     """
#     um = 1e-6
#     return [
#         (150*um, 150*um, 0.0),     # parallel clearance, zero psr
#         (100*um, 180*um, -5.0),    # converging, negative psr
#         (300*um, 250*um, 7.5),     # diverging, positive psr
#     ]


# def _seal_env():
#     """Common geometry/fluid/operating settings used by the paper figures."""
#     Ds = 0.23
#     Ls = 0.15
#     NxSeal = 45
#     mu = 1.4e-3
#     rho = 850.0
#     dp = 16_000_000.0
#     return Ds, Ls, NxSeal, mu, rho, dp


# def _speed_vector(rpm_min=500, rpm_max=7000, n=11) -> np.ndarray:
#     w = np.linspace(rpm_min, rpm_max, int(n)) * np.pi / 30.0
#     return w


# def plot_leakage_validation(models: List[int], cases: List[Tuple[float, float, float]], out_path: str) -> None:
#     """Plot leakage validation: solver curves vs. NN horizontal predictions for several cases.

#     - One subplot per leakage NN model (2x2 grid for 4 models)
#     - Within each subplot, multiple cases overlaid with consistent colors
#     """
#     if main_seal_solver is None:
#         print('[warn] solver_seal.main_seal_solver not available; skipping leakage validation figure.')
#         return

#     _default_rcparams()
#     leak_net = SealLeakModel()
#     w_vec = _speed_vector()
#     rpm = _rpm(w_vec)

#     Ds, Ls, NxSeal, mu, rho, dp = _seal_env()

#     colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2', 'C3'])

#     fig, axs = plt.subplots(2, 2, figsize=figsize_F)
#     axs = axs.ravel()

#     for i, mid in enumerate(models):
#         ax = axs[i]
#         for j, (h_in, h_out, psr) in enumerate(cases):
#             # Solver leakage vs speed
#             geometry = {'hIn': h_in, 'hOut': h_out, 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
#             fluid = {'mu': mu, 'rho': rho}
#             op_conditions = {'dp': dp, 'psr': psr, 'w_vec': w_vec}
#             Leak, *_ = main_seal_solver(geometry, fluid, op_conditions)

#             # NN leakage (single scalar prediction)
#             x = np.array([[h_in, h_out, psr]], dtype=float)
#             y = float(leak_net.predict(mid, x)[0, 0])

#             ax.plot(rpm, Leak, '-', lw=1.5, color=colors[j % len(colors)], label=f'Case {j+1} solver')
#             ax.hlines(y, xmin=rpm.min(), xmax=rpm.max(), colors=colors[j % len(colors)], linestyles='--', lw=1.4,
#                       label=f'Case {j+1} NN')

#         ax.set_title(f'Leakage model {mid}')
#         ax.set_xlabel('Rotational speed (RPM)')
#         ax.set_ylabel('Leakage (kg/s)')
#         ax.grid(True, alpha=0.3)
#         if i == 0:
#             ax.legend(loc='best', ncol=2, fontsize=7, frameon=True)

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=600, bbox_inches='tight')
#     plt.close(fig)


# def plot_rdc_validation_per_model(model_id: int, cases: List[Tuple[float, float, float]], out_path: str) -> None:
#     """For a given RDC model, plot C,c,K,k vs speed for a few cases (solver vs NN)."""
#     if main_seal_solver is None:
#         print(f'[warn] solver not available; skipping RDC figure for model {model_id}.')
#         return

#     _default_rcparams()
#     rdc_net = SealDONModel()
#     w_vec = _speed_vector()
#     rpm = _rpm(w_vec)

#     Ds, Ls, NxSeal, mu, rho, dp = _seal_env()

#     labels = ['C', 'c', 'K', 'k']
#     colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2', 'C3'])

#     fig, axs = plt.subplots(2, 2, figsize=figsize_SC)
#     axs = axs.ravel()

#     for j, (h_in, h_out, psr) in enumerate(cases):
#         # Ground truth via solver
#         geometry = {'hIn': h_in, 'hOut': h_out, 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
#         fluid = {'mu': mu, 'rho': rho}
#         op_conditions = {'dp': dp, 'psr': psr, 'w_vec': w_vec}
#         Leak, RDC, *_ = main_seal_solver(geometry, fluid, op_conditions)
#         rdc_true = RDC[:, 2:]  # [n_w, 4] -> C,c,K,k

#         # NN prediction
#         x = np.array([[h_in, h_out, psr]], dtype=float)
#         rdc_pred = rdc_net.predict(model_id, x, w_vec)[0]  # [4, n_w]

#         for i_c in range(4):
#             ax = axs[i_c]
#             ax.plot(rpm, rdc_true[:, i_c], '-', lw=1.5, color=colors[j % len(colors)], label=f'Case {j+1} solver')
#             ax.plot(rpm, rdc_pred[i_c, :], '--', lw=1.4, color=colors[j % len(colors)], label=f'Case {j+1} NN')
#             ax.set_xlabel('Rotational speed (RPM)')
#             ax.set_ylabel(labels[i_c])
#             ax.grid(True, alpha=0.3)

#     # Put a single legend in the first subplot
#     axs[0].legend(loc='best', ncol=2, fontsize=7, frameon=True)
#     fig.suptitle(f'Seal RDC model {model_id}', y=0.995)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=600, bbox_inches='tight')
#     plt.close(fig)


# def plot_rdc_mape_summary(models: List[int], cases: List[Tuple[float, float, float]], out_path: str) -> None:
#     """Optional: summarize MAPE per coefficient across models for selected cases."""
#     if main_seal_solver is None:
#         print('[warn] solver not available; skipping RDC MAPE summary figure.')
#         return

#     _default_rcparams()
#     rdc_net = SealDONModel()
#     w_vec = _speed_vector()
#     Ds, Ls, NxSeal, mu, rho, dp = _seal_env()

#     mape = np.zeros((len(models), 4))
#     for im, mid in enumerate(models):
#         errs = []
#         vals = []
#         for (h_in, h_out, psr) in cases:
#             geometry = {'hIn': h_in, 'hOut': h_out, 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
#             fluid = {'mu': mu, 'rho': rho}
#             op_conditions = {'dp': dp, 'psr': psr, 'w_vec': w_vec}
#             _, RDC, *_ = main_seal_solver(geometry, fluid, op_conditions)
#             rdc_true = RDC[:, 2:]  # [n_w, 4]

#             x = np.array([[h_in, h_out, psr]], dtype=float)
#             rdc_pred = rdc_net.predict(mid, x, w_vec)[0].T  # [n_w, 4]

#             errs.append(np.abs(rdc_true - rdc_pred))
#             vals.append(np.abs(rdc_true))

#         E = np.concatenate(errs, axis=0)
#         V = np.concatenate(vals, axis=0)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             M = np.nanmean(np.where(V > 0, E / V, 0.0), axis=0)  # [4]
#         mape[im, :] = M

#     labels = ['C', 'c', 'K', 'k']
#     x = np.arange(len(labels))

#     fig, ax = plt.subplots(figsize=figsize_DC)
#     width = 0.18
#     for i, mid in enumerate(models):
#         ax.bar(x + (i - (len(models)-1)/2) * width, 100*mape[i], width=width, label=f'Model {mid}')

#     ax.set_ylabel('MAPE (%)')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.grid(True, axis='y', alpha=0.3)
#     ax.legend(fontsize=7, frameon=True, ncol=2)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=600, bbox_inches='tight')
#     plt.close(fig)


# def save_all_figures(out_dir: str = '.') -> None:
#     os.makedirs(out_dir, exist_ok=True)
#     models = [1, 2, 3, 4]
#     cases = _paper_cases()

#     # 1) Leakage validation (4 models in a 2x2 grid)
#     plot_leakage_validation(models, cases, out_path=os.path.join(out_dir, 'paper_nn_leak_validation.png'))

#     # 2) RDC validation per model (2x2 coefficients per figure)
#     for mid in models:
#         plot_rdc_validation_per_model(mid, cases, out_path=os.path.join(out_dir, f'paper_nn_rdc_validation_model{mid}.png'))

#     # 3) Extra: per-coefficient MAPE summary across the 4 RDC models
#     plot_rdc_mape_summary(models, cases, out_path=os.path.join(out_dir, 'paper_nn_rdc_mape_summary.png'))


# if __name__ == '__main__':
#     # By default, generate from dataset (as requested).
#     # Fallback: uncomment to also produce solver-based figures.
#     save_all_figures_from_dataset('.')
#     # save_all_figures('.')


# # =========================
# # Dataset-based comparisons
# # =========================

# # Map model id -> dataset directory (same ordering used in validation_network.py)
# _MODEL_DATASET_DIR = {
#     1: '20250908_T_182846',
#     2: '20250911_T_091324',
#     3: '20250908_T_183632',
#     4: '20250908_T_203220',
# }


# def _load_dataset_for_model(model_id: int):
#     base = 'dataset/data/tapered_seal'
#     dname = _MODEL_DATASET_DIR[model_id]
#     path = os.path.join(base, dname, 'dataset.mat')
#     with h5py.File(path, 'r') as f:
#         X = np.array(f['input']).T  # [nData, 3]; training scaling (um, um, psr*10)
#         w = np.array(f['params/wVec']).squeeze()  # [n_w] in rad/s
#         # RDC: [6, nVel, nData] -> use 2:6 (C,c,K,k) -> [4, nVel, nData] -> [nData, n_w, 4]
#         RDC = np.array(f['RDC'])
#         RDC = RDC[2:6, :, :].transpose(2, 1, 0)
#         # Leak: likely [1, nData] -> [nData]
#         Leak = np.array(f['Leak']).squeeze()
#     return X, w, RDC, Leak


# def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> str:
#     y_true = np.asarray(y_true).ravel()
#     y_pred = np.asarray(y_pred).ravel()
#     mse = np.mean((y_true - y_pred) ** 2)
#     rmse = np.sqrt(mse)
#     mae = np.mean(np.abs(y_true - y_pred))
#     # r2 (guard zero-variance)
#     vt = np.var(y_true)
#     r2 = 1.0 - (mse / vt) if vt > 0 else np.nan
#     # rRMSE
#     yr = np.nanmax(y_true) - np.nanmin(y_true)
#     rrmse = (rmse / yr * 100.0) if yr > 0 else np.nan
#     return f"RMSE={rmse:.3g}, MAE={mae:.3g}, R2={r2:.3f}, rRMSE={rrmse:.2f}%"


# def plot_leakage_parity_from_dataset(models: List[int], out_dir: str) -> None:
#     _default_rcparams()
#     leak_net = SealLeakModel()
#     for mid in models:
#         X, _, _, Leak_true = _load_dataset_for_model(mid)
#         Leak_pred = leak_net.predict(mid, X).squeeze()

#         fig, ax = plt.subplots(figsize=figsize_DC)
#         ax.scatter(Leak_true, Leak_pred, s=8, alpha=0.6, edgecolors='none')
#         lims = [min(Leak_true.min(), Leak_pred.min()), max(Leak_true.max(), Leak_pred.max())]
#         ax.plot(lims, lims, 'k--', lw=1)
#         ax.set_xlabel('Leakage (dataset) [kg/s]')
#         ax.set_ylabel('Leakage (NN) [kg/s]')
#         ax.set_title(f'Leakage parity (model {mid})')
#         ax.grid(True, alpha=0.3)
#         ax.text(0.05, 0.95, _metrics(Leak_true, Leak_pred), transform=ax.transAxes,
#                 va='top', ha='left', fontsize=7, bbox=dict(fc='white', ec='none', alpha=0.6))
#         fig.tight_layout()
#         fig.savefig(os.path.join(out_dir, f'paper_ds_leak_parity_model{mid}.png'), dpi=600, bbox_inches='tight')
#         plt.close(fig)


# def plot_rdc_parity_from_dataset_per_model(model_id: int, out_dir: str) -> None:
#     _default_rcparams()
#     rdc_net = SealDONModel()
#     X, w, RDC_true, _ = _load_dataset_for_model(model_id)
#     R_pred = rdc_net.predict(model_id, X, w)  # [nData, 4, n_w]
#     R_pred = R_pred.transpose(0, 2, 1)       # [nData, n_w, 4]

#     labels = ['C', 'c', 'K', 'k']
#     fig, axs = plt.subplots(2, 2, figsize=figsize_SC)
#     axs = axs.ravel()
#     for i in range(4):
#         t = RDC_true[:, :, i].ravel()
#         p = R_pred[:, :, i].ravel()
#         ax = axs[i]
#         ax.scatter(t, p, s=6, alpha=0.4, edgecolors='none')
#         lims = [min(np.min(t), np.min(p)), max(np.max(t), np.max(p))]
#         ax.plot(lims, lims, 'k--', lw=1)
#         ax.set_xlabel(f'{labels[i]} (dataset)')
#         ax.set_ylabel(f'{labels[i]} (NN)')
#         ax.grid(True, alpha=0.3)
#         ax.text(0.05, 0.95, _metrics(t, p), transform=ax.transAxes, va='top', ha='left', fontsize=7,
#                 bbox=dict(fc='white', ec='none', alpha=0.6))

#     fig.suptitle(f'RDC parity (model {model_id})', y=0.995)
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, f'paper_ds_rdc_parity_model{model_id}.png'), dpi=600, bbox_inches='tight')
#     plt.close(fig)


# def plot_rdc_speed_curves_from_dataset_per_model(model_id: int, out_dir: str, n_cases: int = 3) -> None:
#     _default_rcparams()
#     rdc_net = SealDONModel()
#     X, w, RDC_true, _ = _load_dataset_for_model(model_id)
#     rpm = _rpm(w)

#     # pick representative cases by spread of psr (3rd feature in X)
#     idx_sort = np.argsort(X[:, 2])
#     picks = np.linspace(0, len(X)-1, n_cases, dtype=int)
#     idx_sel = idx_sort[picks]

#     R_pred = rdc_net.predict(model_id, X[idx_sel], w)  # [n_cases, 4, n_w]

#     labels = ['C', 'c', 'K', 'k']
#     colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0', 'C1', 'C2'])
#     fig, axs = plt.subplots(2, 2, figsize=figsize_SC)
#     axs = axs.ravel()

#     for j, idx in enumerate(idx_sel):
#         T = RDC_true[idx]            # [n_w, 4]
#         P = R_pred[j].T              # [n_w, 4]
#         for i in range(4):
#             ax = axs[i]
#             ax.plot(rpm, T[:, i], '-', lw=1.5, color=colors[j % len(colors)], label=f'Case {j+1} dataset')
#             ax.plot(rpm, P[:, i], '--', lw=1.2, color=colors[j % len(colors)], label=f'Case {j+1} NN')
#             ax.set_xlabel('Rotational speed (RPM)')
#             ax.set_ylabel(labels[i])
#             ax.grid(True, alpha=0.3)

#     axs[0].legend(fontsize=7, ncol=2)
#     fig.suptitle(f'RDC curves vs speed (model {model_id})', y=0.995)
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, f'paper_ds_rdc_speed_model{model_id}.png'), dpi=600, bbox_inches='tight')
#     plt.close(fig)


# def save_all_figures_from_dataset(out_dir: str = '.') -> None:
#     os.makedirs(out_dir, exist_ok=True)
#     models = [1, 2, 3, 4]

#     # 1) Leakage parity per model
#     plot_leakage_parity_from_dataset(models, out_dir)

#     # 2) RDC parity + speed curves per model
#     for mid in models:
#         plot_rdc_parity_from_dataset_per_model(mid, out_dir)
#         plot_rdc_speed_curves_from_dataset_per_model(mid, out_dir, n_cases=3)
