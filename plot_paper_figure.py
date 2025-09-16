#%%
# --- for each input
# Campbell diagram
# Log-dec plot
# Unbalance response

# --- optimized
# parallel coordinate plot
# pareto front
# Radviz
# bearing history
# objective function history

import os
import numpy as np
import matplotlib.pyplot as plt

from import_data import rotor_import
from loader_brg_seal import BearingNondModel, SealDONModel, SealLeakModel
from solver_rotordyn import assemble_system_matrix, eig_batch, unbalance_response_batch_cpu_parallel
from scipy.signal import find_peaks
from itertools import cycle
from pymoo.visualization.pcp import PCP
from pymoo.visualization.radviz import Radviz
from typing import Optional

w_oper = 3500 * np.pi / 30

n_type_brg = 55 # brg types
LB_brg_idx = 1; UB_brg_idx = 55
LB_cr = 10;  UB_cr = 30   # Cr/D = 10/10000 ~ 30/10000
LB_h = 100; UB_h = 500 # seal clearance range
LB_psr = -10;  UB_psr = 10   # -> *0.1 해서 [0,1.0]

f_brg_dim = np.array([[1, 1e-4],[1, 1e-4]])
f_seal_dim = [1e-6, 1e-6, 1e-1]
rdc_signs = np.array([1, 1, -1, 1])

N_FWD_EVAL = 4 # forward 모드 n개만 평가
LOGDEC_MIN = 0.1 # 대수감쇠율 > 0
AF_MAX_ALLOW = 8.0
RATIO_MAX = 75.0



#%%

cm = 1/2.54  # centimeters in inches
figsize_F = (6.8, 4.2) # single column
figsize_SC = (6, 3.7) # single column
figsize_SC_two_third = (4, 3.7) # single column
figsize_SC_one_third = (2.1, 3.7) # single column
figsize_SC_two_third_s = (4.2, 2.4) # single column, short
figsize_SC_one_third_s = (2.4, 2.4) # single column, short
figsize_SC_two_third_ss = (4.2, 1.8) # single column, shorter
figsize_SC_one_third_ss = (2.4, 1.8) # single column, shorter
figsize_SC_s = (6, 2.8) # single column, short
figsize_SC_ss = (6, 2.4) # single column, shorter
figsize_SC_sss = (6, 2) # single column, shorter
figsize_DC = (3.3, 2.06) # double column
# figsize_DC_b = (3.3, 2.06) # double column, big
figsize_DC_b = (3.5, 2.18) # double column, big
figsize_DC_tall = (3.3, 3.3) # double column

def _default_rcparams():
    plt.rcParams.update({
        "figure.figsize": figsize_DC,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1
    })


def build_rotor_and_models(bs_params):
    data_dir = 'dataset/data'
    rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
    rotor_sheet = "RDOPT"

    (n_ele, n_node, n_dof, n_add, n_brg, n_seal,
    rotor_elements, rotor_nodal_props, added_elements, added_props,
    mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb,
    brgs, seals, support_dofs) = rotor_import(file_path=rotor_file, sheet_name=rotor_sheet, bs_params=bs_params)

    unique_seals = {s.SealNet for s in seals}
    n_type_seal = len(unique_seals)
    
    model_brg = BearingNondModel()
    model_seal = SealDONModel()
    model_seal_leak = SealLeakModel()

    return {
        'n_ele': n_ele,
        'n_node': n_node,
        'n_dof': n_dof,
        'n_brg': n_brg,
        'n_seal': n_seal,
        'n_type_seal': n_type_seal,
        'rotor_elements': rotor_elements,
        'mat_M': mat_M,
        'mat_K_r': mat_K_r,
        'mat_C_g': mat_C_g,
        'unb': unb,
        'brgs': brgs,
        'seals': seals,
        'support_dofs': support_dofs,
        'model_brg': model_brg,
        'model_seal': model_seal,
        'model_seal_leak': model_seal_leak,
        'added_elements': added_elements,
    }

def calc_KC_for_design(X, ctx, w_vec):
    n_brg = ctx['n_brg']
    n_seal = ctx['n_seal']
    brgs = ctx['brgs']
    seals = ctx['seals']
    model_brg = ctx['model_brg']
    model_seal = ctx['model_seal']
    n_type_seal = ctx['n_type_seal']
    model_seal_leak = ctx['model_seal_leak']

    # Parameter scalings
    f_brg_dim = np.array([[1, 1e-4], [1, 1e-4]])
    f_seal_dim = [1e-6, 1e-6, 1e-1]
    rdc_signs = np.array([1, 1, -1, 1])

    pop = X.shape[0]
    X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
    X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

    # Bearings
    x_brg = X_brg * f_brg_dim
    K_brg, C_brg, loss_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

    # Group seals by type
    groups = {}
    for i, s in enumerate(seals):
        groups.setdefault(s.SealNet, []).append(i)
    idx_seal = [np.array(groups.get(t+1, []), dtype=int) for t in range(n_type_seal)]

    # Seals
    seal_leak = np.zeros((pop, n_seal), dtype=float)
    seal_rdc = np.zeros((pop, n_seal, 4, w_vec.shape[0]), dtype=float)
    for t in range(n_type_seal):
        idx = idx_seal[t]
        if idx.size == 0:
            continue
        params_t = X_seal[:, idx]
        x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
        leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))   
        rdc_flat = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, w_vec.shape[0])
        seal_rdc[:, idx] = rdc_flat
        seal_leak[:, idx] = leak_flat

    K_seal = seal_rdc[:, :, [2, 3, 3, 2], :] * rdc_signs[None, None, :, None]
    C_seal = seal_rdc[:, :, [0, 1, 1, 0], :] * rdc_signs[None, None, :, None]

    K_vals = np.concatenate([K_brg, K_seal], axis=1)
    C_vals = np.concatenate([C_brg, C_seal], axis=1)
    
    leak = seal_leak.sum(axis=1)

    return K_vals, C_vals, loss_brg, leak


def analyze_design(X, ctx, w_vec):
    mat_M = ctx['mat_M']
    mat_K_r = ctx['mat_K_r']
    mat_C_g = ctx['mat_C_g']
    support_dofs = ctx['support_dofs']
    n_dof = ctx['n_dof']
    unb = ctx['unb']
    n_brg = ctx['n_brg']
    n_seal = ctx['n_seal']
    brgs = ctx['brgs']
    seals = ctx['seals']
    model_brg = ctx['model_brg']
    model_seal = ctx['model_seal']
    
    pop = X.shape[0]
    X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
    X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

    rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
    C_struct = np.zeros_like(mat_K_r)

    K_vals, C_vals, loss_brg_full, leak = calc_KC_for_design(X, ctx, w_vec)
    
    K_all, Ceff_all = assemble_system_matrix(mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals)

    eigvals, _ = eig_batch(M=mat_M, K_all=K_all, Ceff_all=Ceff_all, track=True)

    harmonic = unbalance_response_batch_cpu_parallel(
        M=mat_M, unb=unb, K_all=K_all, Ceff_all=Ceff_all, w_vec=w_vec
    )

    # amplitude [pop, n_w, n_node]
    n_node = ctx['n_node']
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2
    Ux = harmonic[:, :, idx_x, 0]
    Uy = harmonic[:, :, idx_y, 0]
    amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)
    
    F = np.zeros((pop, 6), dtype=float) 
    idx_op = int(np.argmin(np.abs(w_vec - w_oper)))
    F[:, 0] = leak
    loss_brg = loss_brg_full[:,:,:,idx_op].sum(axis=1).squeeze()
    F[:, 1] = loss_brg

    assert n_dof == 4 * n_node
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2
    Ux = harmonic[:, :, idx_x, 0]  # [pop, n_w, n_node]
    Uy = harmonic[:, :, idx_y, 0]
    amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)  # [pop, n_w, n_node]

    brg_nodes = np.array([b.node for b in brgs], dtype=int)
    seal_nodes = np.array([s.node for s in seals], dtype=int)
    cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))
    
    
    Bx = np.stack([Ux.real, -Ux.imag], axis=-1)      # (..., 2)
    By = np.stack([Uy.real, -Uy.imag], axis=-1)      # (..., 2)
    B  = np.stack([Bx, By], axis=-2)                 # (..., 2, 2)

    U, S, Vt = np.linalg.svd(B)                      # batched SVD
    amp = S[..., 0]                                    # 진짜 최대 진폭 (장반경)
    b_amp = S[..., 1]                                    # 최소 진폭 (단반경)

    # 장축 방향과 최대가 생기는 위상(원하면)
    psi   = np.arctan2(U[..., 1, 0], U[..., 0, 0])   # 장축 각
    tstar = np.arctan2(Vt[..., 0, 1], Vt[..., 0, 0]) # 피크 위상
    alpha = -tstar
    
    # t = np.linspace(0,2*np.pi,12)
    
    # hx = harmonic[:, :, idx_x, 0]  # [pop, n_w, n_node]
    # hy = harmonic[:, :, idx_y, 0]
    
    # qx = np.real(hx.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    # qy = np.real(hy.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    
    # q = np.sqrt(qx**2 + qy**2)
    # i_max = np.argmax(np.max(q,axis=1))
    # i_psi = np.argmax(q[i_max,:])
    # alpha = np.atan(qy[i_max,i_psi]/qx[i_max,i_psi])
    
    # hx_r = hx * np.exp(1j * alpha)
    # hy_r = hy * np.exp(1j * alpha)
    # theta = 0.0
    # amp = np.abs(np.real(hx_r * np.cos(theta) + hy_r * np.sin(theta)))

    eps = 1e-18
    AF_max = np.zeros(pop, dtype=float)
    # For separation margin: collect peak centers and AF per peak for nearest peak logic
    peak_centers_list = [[] for _ in range(pop)]  # angular speed of peaks
    peak_af_list = [[] for _ in range(pop)]       # AF at those peaks
    w = w_vec
    for p in range(pop):
        af_p = 0.0
        A = amp[p, :, cal_nodes]  # [n_cal, n_w] due to fancy indexing
        for c in range(len(cal_nodes)):
            y = A[c, :]
            if y.size < 3:
                continue
            pk, _ = find_peaks(y)
            if pk.size == 0:
                continue
            for j in pk:
                Ac = y[j]
                if Ac <= eps:
                    continue
                yhpp = Ac / np.sqrt(2.0)
                # Left half-power crossing
                if j == 0:
                    N1 = w[0]
                else:
                    l_idx = np.where(y[:j] <= yhpp)[0]
                    if l_idx.size == 0:
                        N1 = w[0]
                    else:
                        i0 = l_idx[-1]
                        x0, y0 = w[i0], y[i0]
                        N1 = x0 + (yhpp - y0) / max(Ac - y0, eps) * (w[j] - x0)
                # Right half-power crossing
                if j >= y.size - 1:
                    N2 = w[-1]
                else:
                    r_idx_rel = np.where(y[j+1:] <= yhpp)[0]
                    if r_idx_rel.size == 0:
                        N2 = w[-1]
                    else:
                        i1 = j + 1 + r_idx_rel[0]
                        x1, y1 = w[i1], y[i1]
                        N2 = x1 + (yhpp - y1) / max(Ac - y1, eps) * (w[j] - x1)
                af_peak = w[j] / max(N2 - N1, eps)
                # record peak center and AF for separation margin evaluation
                peak_centers_list[p].append(w[j])
                peak_af_list[p].append(af_peak)
                if af_peak > af_p:
                    af_p = af_peak
        AF_max[p] = af_p
    F[:, 2] = AF_max

    k_use = min(N_FWD_EVAL, eigvals.shape[2])
    alpha = np.real(eigvals[:,:,:k_use])  # [pop, n_w, 2n]
    beta  = np.imag(eigvals[:,:,:k_use])

    logdec = -2 * np.pi * alpha / np.sqrt(alpha**2 + beta**2)
    min_logdec = np.min(logdec, axis=(1, 2)) # [pop]
    F[:, 3] = -min_logdec

    # logdec2 = -2 * np.pi * alpha8 / beta8 # almost same
    # wn = np.sqrt(np.maximum(alpha8**2 + beta8**2, 1e-30))
    # zeta = np.clip(-alpha8 / (wn + 1e-30), a_min=1e-12, a_max = 0.999)
    # logdec3 = 2.0 * np.pi * zeta / np.sqrt(1.0 - zeta**2)
    # logdec_masked = np.where(valid, logdec, np.inf)
    # min_logdec = np.min(logdec_masked, axis=(1, 2))  # [pop]

    brg_nodes = np.array([b.node for b in brgs], dtype=int)
    if brg_nodes.size > 0:
        brg_ids = X_brg[:, :, 0].astype(int)                         # [pop, n_brg]
        cr_ratio = (X_brg[:, :, 1] * f_brg_dim[0, 1]).astype(float)  # scaled
        Db_all = np.array([b.Db for b in brgs], dtype=float)[None, :]
        Cr = Db_all * cr_ratio                                       # [pop, n_brg]

        max_id = int(np.max(brg_ids)) if brg_ids.size else 0
        mp_lookup = np.zeros(max(UB_brg_idx + 1, max_id + 1), dtype=float)
        for bid in range(1, mp_lookup.size):
            try:
                mp_lookup[bid] = float(model_brg.get_bearing_by_id(bid)['Mp'])
            except Exception:
                mp_lookup[bid] = 0.0
        Mp = mp_lookup[brg_ids]
        Cp = Cr / np.clip(1.0 - Mp, 1e-9, None)                     # [pop, n_brg]
        amp_brg = amp[:, :, brg_nodes]                               # [pop, n_w, n_brg]
        amp_ratio_brg = (amp_brg / (Cp[:, None, :] + eps)) * 100.0
        F[:, 4] = amp_ratio_brg.reshape(pop, -1).max(axis=1)
    else:
        F[:, 4] = 0.0

    if seal_nodes.size > 0:
        h_in  = (X_seal[:, :, 0].astype(float) * f_seal_dim[0])   # [pop, n_seal]
        h_out = (X_seal[:, :, 1].astype(float) * f_seal_dim[1])   # [pop, n_seal]
        h_min = np.minimum(h_in, h_out)
        amp_seal = amp[:, :, seal_nodes]                          # [pop, n_w, n_seal]
        amp_ratio_seal = (amp_seal / (h_min[:, None, :] + eps)) * 100.0
        F[:, 5] = amp_ratio_seal.reshape(pop, -1).max(axis=1)
    else:
        F[:, 5] = 0.0

    return eigvals, amp, loss_brg, loss_brg_full, leak, F, peak_af_list, peak_centers_list, logdec, amp_ratio_brg, amp_ratio_seal, harmonic


#%%

def plot_campbell(eigvals, w_vec, out_path, title=None, ylim_rpm=(0, 7000), figsize=figsize_DC):
    _default_rcparams()
    E = np.array(eigvals)[0]  # [n_w, k]
    beta = np.abs(E.imag) * 30.0 / np.pi / 60 # in Hz
    rpm = w_vec * 30.0 / np.pi
    
    # pick lowest n modes by average frequency
    n_pick = min(4, beta.shape[1]) if beta.ndim == 2 else 0
    if n_pick == 0:
        return
    idx = np.argsort(beta.mean(axis=0))[:n_pick]
    
    fig, ax = plt.subplots(figsize=figsize)
    # excitation (1x) line in Hz
    exc = rpm / 60.0
    
    ax.plot(rpm, exc, 'k-', lw=1.0, label='1x')
    
    for k in idx:
        if k == 0:
            suffix='st'
        elif k == 1:
            suffix='nd'
        elif k == 2:
            suffix='rd'
        else:
            suffix='st'
        line, = ax.plot(rpm, beta[:, k], '-', label=f'{k}{suffix} F')
        
    for k in idx:
        cross_rpm = []
        f = beta[:, k] - exc
        for i in range(len(f) - 1):
            fi, fj = f[i], f[i+1]
            if not np.isfinite(fi) or not np.isfinite(fj):
                continue
            if fi == 0:
                cross_rpm.append(rpm[i])
            elif fj == 0:
                cross_rpm.append(rpm[i+1])
            elif fi * fj < 0:  
                t = fi / (fi - fj)
                r_c = rpm[i] + t * (rpm[i+1] - rpm[i])
                cross_rpm.append(r_c)
        if cross_rpm:
            cross_rpm = np.array(cross_rpm, dtype=float)
            ax.plot(cross_rpm, cross_rpm/60.0, linestyle='None', marker='o',
                    markersize=6.0, markerfacecolor='none', markeredgecolor='black',
                    markeredgewidth=1.1)
            
    # plot 1x line last so it stays visible bu under markers
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel('Frequency (Hz)')
    if title:
        ax.set_title(title)
    ax.set_ylim([0, ylim_rpm[1]/60])
    ax.set_xlim([0, rpm.max()])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.grid(True, alpha=0.3)
    # ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')




def plot_logdec(logdec_arr, w_vec, out_path, ylim=(0.0, 3.0), figsize=figsize_DC):
    _default_rcparams()
    from scipy.interpolate import CubicSpline
    if logdec_arr.shape[0] == 1:
        logdec_arr = logdec_arr.squeeze()
    n_w, n_mode = logdec_arr.shape
    
    x_rpm = w_vec * 30.0 / np.pi
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(n_mode):
        if k == 0:
            suffix='st'
        elif k == 1:
            suffix='nd'
        elif k == 2:
            suffix='rd'
        else:
            suffix='st'
        
        if n_w >= 3:
            w_use = np.linspace(w_vec.min(), w_vec.max(), max(200, n_w*5))
            cs = CubicSpline(w_vec, logdec_arr[:,k], bc_type='natural')
            y_plot = cs(w_use)
            ax.plot(w_use* 30.0 / np.pi, y_plot, label=f'{k+1}{suffix}')
        else:
            ax.plot(x_rpm, logdec_arr[:, k], label=f'{k+1}{suffix}')
        
    # for k in range(n_mode):
    #     ax.plot(x_rpm, logdec_arr[:, k], label=f"Mode {k+1}")
    
    ax.axhline(0.1, color='r', lw=1.2, linestyle='--')
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'$\delta$')
    ax.set_xlim([0, x_rpm.max()])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.legend(loc=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')





def plot_logdec_lowest(logdec_arr, eigvals, w_vec, out_path, n=4, ylim=(0.0, 2.0), figsize=figsize_DC):
    _default_rcparams()
    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))
    
    L = np.array(logdec_arr)  # [n_w, m]
    EV = np.array(eigvals)[0]  # [n_w, k]
    m = min(L.shape[1], EV.shape[1])
    if m == 0:
        return
    # select lowest frequency modes at first speed
    alpha0 = EV[0, :m].real
    beta0 = EV[0, :m].imag
    wn0 = np.sqrt(np.maximum(alpha0*alpha0 + beta0*beta0, 0.0))
    order = np.argsort(wn0)
    sel = order[:max(1, int(n))]
    
    x_rpm = w_vec * 30.0 / np.pi
    fig, ax = plt.subplots(figsize=figsize)
    for i, k in enumerate(sel):
        if k == 0:
            suffix='st'
        elif k == 1:
            suffix='nd'
        elif k == 2:
            suffix='rd'
        else:
            suffix='st'
        ax.plot(x_rpm, L[:, k], label=f'{k}{suffix} F')
        
    # for i, k in enumerate(sel):
    #     ax.plot(x_rpm, L[:, k], label=f"Mode {i+1}")
    # ax.axhline(0.1, color='r', lw=1.2, linestyle='--')
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'$\delta$')
    # ax.set_ylim(ylim)
    ax.set_xlim([0, x_rpm.max()])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')




def plot_bearing_id_hist(X_pop, X_par, n_brg, idx=None, figsize=figsize_SC_s):
    import matplotlib as mpl
    plt.rc('axes', prop_cycle=mpl.rcParamsDefault['axes.prop_cycle'])
    ids_pop = X_pop[:, :2*n_brg].reshape(-1, n_brg, 2)[:,:,0].ravel()
    ids_par = X_par[:, :2*n_brg].reshape(-1, n_brg, 2)[:,:,0].ravel()
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(ids_pop, bins=np.arange(1,55)+1, alpha=0.4, rwidth=0.8, align='mid',label='Cand.')
    ax.hist(ids_par, bins=np.arange(1,55)+1, alpha=0.8, rwidth=0.8, align='mid',label='Pareto')
    
    # boundaries = [4, 5, 9, 15, 19, 23, 39, 55]   # 각 타입 끝 ID
    # labels = ["Axial grooved", "Pressure dam", "Partial arc", "2-Lobe", "3-Lobe", "4-Lobe", "4-Pad tilting", "5-Pad tilting"]
    boundaries = [1, 4, 5, 9]   # 각 타입 끝 ID
    labels = ["", "Axial grooved", "Pressure dam", "Partial arc"]
    for b in boundaries:
        # ax.axvline(b + 0.5, color='k', linestyle='--', lw=1)
        ax.axvline(b, color='k', linestyle='--', lw=1)

    ylevels_ = [1.2, 1.02, 1.11] 
    # 타입 레이블 표시
    for i, b in enumerate(boundaries):
        if i == 0:
            x0 = 1
        else:
            x0 = boundaries[i-1]+1
        x1 = b
        xpos = (x0 + x1) / 2
        
        ylevels = ylevels_  # ylim 대비 비율
        ypos = ax.get_ylim()[1] * ylevels[i % 3]
        
        ax.text(xpos-0.5, ypos, labels[i],
                ha='center', va='bottom', fontsize=8, rotation=0)
        
    boundaries = [15, 19, 23, 39, 55]   # 각 타입 끝 ID
    labels = ["2-Lobe", "3-Lobe", "4-Lobe", "4-Pad tilting", "5-Pad tilting"]
    for b in boundaries[:-1]:
        # ax.axvline(b + 0.5, color='k', linestyle='--', lw=1)
        ax.axvline(b, color='k', linestyle='--', lw=1)

    # 타입 레이블 표시
    for i, b in enumerate(boundaries):
        if i == 0:
            x0 = 9
        else:
            x0 = boundaries[i-1]+1
        x1 = b
        xpos = (x0 + x1) / 2
        
        ylevels = ylevels_[2] # ylim 대비 비율
        ypos = ax.get_ylim()[1] * ylevels
        
        ax.text(xpos-0.5, ypos, labels[i],
                ha='center', va='bottom', fontsize=8, rotation=0)
        
    boundaries = [31, 39, 47, 55]   # 각 타입 끝 ID
    labels = ["LBP","LOP","LBP","LOP"]
    # 타입 레이블 표시
    for i, b in enumerate(boundaries):
        if i%2 == 0:
            ax.axvline(b, color='b', linestyle='--', lw=1)
            
        if i == 0:
            x0 = 23
        else:
            x0 = boundaries[i-1]+1
        x1 = b
        xpos = (x0 + x1) / 2
        
        ylevels = ylevels_[1]  # ylim 대비 비율
        ypos = ax.get_ylim()[1] * ylevels
        
        ax.text(xpos-0.5, ypos, labels[i],
                ha='center', va='bottom', fontsize=8, rotation=0)

    boundaries = [4, 5, 9, 15, 19, 23, 31, 39, 47, 55]
    ax.set_xlim(1, 55)
    ax.set_xticks(boundaries)
    plt.xlabel('Bearing ID'); plt.ylabel('Count'); plt.legend(); plt.tight_layout(); 
    fig.savefig("bearing_hist.png", dpi=600, bbox_inches="tight")
    plt.show()








def plot_unbalance_response_rpm(amp, w_vec, ctx, out_path, nodes='key', smooth_spline=True, figsize=figsize_DC, show_lgd=False):
    _default_rcparams()
    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))
    
    A = np.array(amp)[0]  # [n_w, n_node]
    n_w, n_node = A.shape
    brgs = ctx['brgs']
    seals = ctx['seals']
    unb = ctx['unb']

    bearing_nodes = np.array([b.node for b in brgs], dtype=int) if len(brgs) else np.array([], dtype=int)
    seal_nodes = np.array([s.node for s in seals], dtype=int) if len(seals) else np.array([], dtype=int)
    unb_nodes = np.unique(np.array([n for c in unb.cases for n in c.node], dtype=int)) if getattr(unb, 'cases', None) else np.array([], dtype=int)

    if nodes == 'all':
        sel = np.arange(n_node)
    elif nodes == 'brg':
        sel = bearing_nodes
    elif nodes == 'seal':
        sel = seal_nodes
    elif nodes == 'unb':
        sel = unb_nodes
    elif nodes == 'key':
        sel = np.unique(np.concatenate([bearing_nodes, seal_nodes, unb_nodes]))
    elif isinstance(nodes, (list, tuple, np.ndarray)):
        sel = np.array(nodes, dtype=int)
    else:
        sel = np.unique(np.concatenate([bearing_nodes, seal_nodes, unb_nodes]))
    sel = sel.astype(int)
    sel = sel[(sel >= 0) & (sel < n_node)]
    if sel.size == 0:
        sel = np.arange(min(n_node, 6))
        
    set_brg = set(bearing_nodes.tolist())
    set_seal = set(seal_nodes.tolist())
    set_unb = set(unb_nodes.tolist())
    
    def category(n):
        if n in set_brg:
            return 'brg'
        if n in set_seal:
            return 'seal'
        if n in set_unb:
            return 'unb'
        return 'other'

    from scipy.interpolate import CubicSpline
    rpm = w_vec * 30.0 / np.pi
    fig, ax = plt.subplots(figsize=figsize)
    
    seen_labels = set()
    w_use = w_vec
    # for n, sty in zip(sel, cycle(sty_cycle)):
    for iu, (n, sty) in enumerate(zip(sel, cycle(sty_cycle))):
        y = A[:, n]
        cat = category(n)
        label = f"{cat.title()} {n}"
        label = f"{'node'.title()} {n}, {cat.title()} {iu}"
        label = f"{'node'.title()} {n}"
        lbl = None if cat in seen_labels else label
        if lbl is not None:
            seen_labels.add(label)
        if smooth_spline and n_w >= 3:
            w_use = np.linspace(w_vec.min(), w_vec.max(), max(200, n_w*5))
            cs = CubicSpline(w_vec, y, bc_type='natural')
            y_plot = cs(w_use)
            ax.plot(w_use * 30.0 / np.pi, y_plot * 1e6, lw=1.5, label=lbl, **sty,)
        else:
            ax.plot(rpm, y * 1e6, lw=1.5, label=lbl, **sty,)
    
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'Unbalanced response $(\mu m)$')
    _, ylim_max = ax.get_ylim()
    ax.set_ylim(0, ylim_max)
    ax.set_xlim(0, rpm.max())
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.grid(True, alpha=0.3)
    if show_lgd:
        ax.legend()
        # long legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                borderaxespad=0., ncol=2,
                handlelength=1.2)

    fig.tight_layout()
    
    fig.savefig(out_path, dpi=600, bbox_inches='tight')







def plot_unbalance_response_rpm_phase(harmonic_resp, w_vec, ctx, out_path, nodes='key', smooth_spline=True, figsize=figsize_DC, show_lgd=False):
    _default_rcparams()
    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))
    
    
    
    A = np.array(amp)[0]  # [n_w, n_node]
    n_w, n_node = A.shape
    brgs = ctx['brgs']
    seals = ctx['seals']
    unb = ctx['unb']

    bearing_nodes = np.array([b.node for b in brgs], dtype=int) if len(brgs) else np.array([], dtype=int)
    seal_nodes = np.array([s.node for s in seals], dtype=int) if len(seals) else np.array([], dtype=int)
    unb_nodes = np.unique(np.array([n for c in unb.cases for n in c.node], dtype=int)) if getattr(unb, 'cases', None) else np.array([], dtype=int)

    if nodes == 'all':
        sel = np.arange(n_node)
    elif nodes == 'brg':
        sel = bearing_nodes
    elif nodes == 'seal':
        sel = seal_nodes
    elif nodes == 'unb':
        sel = unb_nodes
    elif nodes == 'key':
        sel = np.unique(np.concatenate([bearing_nodes, seal_nodes, unb_nodes]))
    elif isinstance(nodes, (list, tuple, np.ndarray)):
        sel = np.array(nodes, dtype=int)
    else:
        sel = np.unique(np.concatenate([bearing_nodes, seal_nodes, unb_nodes]))
    sel = sel.astype(int)
    sel = sel[(sel >= 0) & (sel < n_node)]
    if sel.size == 0:
        sel = np.arange(min(n_node, 6))
        
    set_brg = set(bearing_nodes.tolist())
    set_seal = set(seal_nodes.tolist())
    set_unb = set(unb_nodes.tolist())
    
    def category(n):
        if n in set_brg:
            return 'brg'
        if n in set_seal:
            return 'seal'
        if n in set_unb:
            return 'unb'
        return 'other'

    from scipy.interpolate import CubicSpline
    rpm = w_vec * 30.0 / np.pi
    fig, ax = plt.subplots(figsize=figsize)
    
    seen_labels = set()
    w_use = w_vec
    # for n, sty in zip(sel, cycle(sty_cycle)):
    for iu, (n, sty) in enumerate(zip(sel, cycle(sty_cycle))):
        y = A[:, n]
        cat = category(n)
        label = f"{cat.title()} {n}"
        label = f"{'node'.title()} {n}, {cat.title()} {iu}"
        label = f"{'node'.title()} {n}"
        lbl = None if cat in seen_labels else label
        if lbl is not None:
            seen_labels.add(label)
        if smooth_spline and n_w >= 3:
            w_use = np.linspace(w_vec.min(), w_vec.max(), max(200, n_w*5))
            cs = CubicSpline(w_vec, y, bc_type='natural')
            y_plot = cs(w_use)
            ax.plot(w_use * 30.0 / np.pi, y_plot * 1e6, lw=1.5, label=lbl, **sty,)
        else:
            ax.plot(rpm, y * 1e6, lw=1.5, label=lbl, **sty,)
    
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'Unbalanced response $(\mu m)$')
    _, ylim_max = ax.get_ylim()
    ax.set_ylim(0, ylim_max)
    ax.set_xlim(0, rpm.max())
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.grid(True, alpha=0.3)
    if show_lgd:
        ax.legend()
        # long legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                borderaxespad=0., ncol=2,
                handlelength=1.2)

    fig.tight_layout()
    
    fig.savefig(out_path, dpi=600, bbox_inches='tight')







def plot_unbalance_response_rated_speed(x_nodes, harmonic_resp, out_path, figsize=figsize_SC_sss, plot_rotor=False,
                                        rotor_elements=None, added_elements=None, brgs=None, seals=None, colors=None, ylim=None, unb=None):
    _default_rcparams()
    lw = 1
    n_dof = x_nodes.shape[0]*4
    h = harmonic_resp.squeeze()
    hx = h[np.arange(0,n_dof,4)]
    hy = h[np.arange(2,n_dof,4)]

    t = np.linspace(0,2*np.pi,12)
    qx = np.real(hx.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    qy = np.real(hy.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))

    q = np.sqrt(qx**2 + qy**2)
    i_max = np.argmax(np.max(q,axis=1))
    i_psi = np.argmax(q[i_max,:])
    alpha = np.atan(qy[i_max,i_psi]/qx[i_max,i_psi])

    hx_r = hx * np.exp(1j * alpha)
    hy_r = hy * np.exp(1j * alpha)
    theta = 0.0
    A_signed = np.real(hx_r * np.cos(theta) + hy_r * np.sin(theta))

    # Use response as the Y-axis scale (in micrometers)
    y_scale_resp = 1e6  # meters -> micrometers
    A_plot = A_signed * y_scale_resp
    if np.sign(A_plot[0]) == -1:
        A_plot *=-1
    
    if ylim is None:
        ylim=np.ceil(np.max(np.abs(A_plot))/5+1)*5

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(x_nodes,A_plot,'-', linewidth=2, zorder=5)
    ax1.plot(x_nodes,-A_plot,'-', linewidth=2, zorder=5)
    ax1.set_ylabel('Ampliture (um)')
    ax1.set_xlabel('Axial location (m)')
    ax1.set_ylim(-ylim, ylim)

    if plot_rotor:
        from matplotlib.patches import Rectangle, Circle

        if colors is None:
            colors = {
                'outer': "#2133BB",
                'outer': "#737976",
                'inner': '0.35',
                'added': 'tab:purple',
                'spring_brg': 'tab:red',
                'damper_brg': 'tab:green',
                'spring_seal': 'tab:orange',
                'damper_seal': 'tab:blue',
                'center': "#A7A7A7",
            }
        
        alpha = 1.0
        vec_L = np.array([e.L for e in rotor_elements], dtype=float)
        x_nodes = np.concatenate([[0.0], np.cumsum(vec_L)])
        # Create axes if needed
        ax = ax1.twinx()
        created_ax = True

        ax.axhline(0, color=colors['center'], zorder=1, linestyle='--')

        rmax = 0.0
        for i, e in enumerate(rotor_elements):
            x0, L = x_nodes[i], float(e.L)
            ro = 0.5 * float(e.Od)
            ri = 0.5 * float(e.Id)
            rmax = max(rmax, ro)

            if ro > 0 and L > 0:
                rect = Rectangle((x0, -ro), width=L, height=2*ro,
                                fill=False, edgecolor=colors['outer'], zorder=3,alpha=alpha)
                ax.add_patch(rect)
        
        def draw_disk(ax, x0, h, r, label=None):
            arm_x = [x0, x0]
            arm_y = [-h/2, h/2]
            color = '#0D7441'
            color = '#737976'
            ax.plot(arm_x, arm_y, color=color, lw=lw, label=label, alpha=alpha)
            
            circ = Circle((x0,h/2+r),r , color=color, fill=False, lw=lw, alpha=alpha)
            ax.add_patch(circ)
            circ = Circle((x0,-h/2-r),r , color=color, fill=False, lw=lw, alpha=alpha)
            ax.add_patch(circ)

        r_disk = 0.025
        added_D_max = 0.0
        if added_elements:
            for a in added_elements: 
                added_D_max = max(added_D_max,a.Od)
            added_D_max *= 1.2
            for idx, a in enumerate(added_elements): 
                x0 = x_nodes[int(a.node)]
                added_D_max = max(added_D_max,a.Od)
                label = None
                if idx == 0:
                    label = "Disk"
                draw_disk(ax, x0=x0, h=added_D_max, r=r_disk, label=label)
        rmax = max(rmax, added_D_max/2.0)

        n_ele = len(rotor_elements)
        ro_nodes = np.zeros_like(x_nodes)
        for i in range(n_ele + 1):
            left_ro = 0.5 * rotor_elements[i-1].Od if i > 0 else 0.5 * rotor_elements[0].Od
            right_ro = 0.5 * rotor_elements[i].Od if i < n_ele else 0.5 * rotor_elements[-1].Od
            ro_nodes[i] = max(left_ro, right_ro)
        if added_elements:
            for a in added_elements:
                ro_nodes[int(a.node)] = max(ro_nodes[int(a.node)], 0.5*float(a.Od))

                
        def draw_xbox(ax, x0, w, h, label=None):
            rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
                            fill=True, facecolor='#737976', edgecolor="#737976", lw=lw, label=label, alpha=alpha)
            # rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
            #                 fill=True, facecolor='#B8BEBB', edgecolor="#737976", lw=lw, label=label, alpha=alpha)
            ax.add_patch(rect)
            # xx = [x0-w/2, x0+w/2]
            # yx = [-h/2, h/2]
            # ax.plot(xx, yx, color="#737976", lw=0.9*lw, alpha=alpha)
            # xx = [x0-w/2, x0+w/2]
            # yx = [h/2, -h/2]
            # ax.plot(xx, yx, color='#737976', lw=0.9*lw, alpha=alpha)
            
        def draw_seal(ax, x0, w, ri, h, c, color=None,label=None):
            
            if color is None:
                color = "#4F60BE"
            
            # rect = Rectangle((x0-w/2, -ro-c), width=w, height=h,
            #                 fill=True, facecolor="#4F60BE", edgecolor="#4F60BE", lw=lw, label=label)
            rect = Rectangle((x0-w/2, -ri-c-h), width=w, height=h,
                            fill=True, facecolor=color, edgecolor=color, lw=lw, label=label, alpha=alpha)
            ax.add_patch(rect)
            rect = Rectangle((x0-w/2, ri+c), width=w, height=h,
                            fill=True, facecolor=color, edgecolor=color, lw=lw, alpha=alpha)
            ax.add_patch(rect)
            
        if brgs:
            for idx, b in enumerate(brgs):
                j = int(b.node)
                xb = x_nodes[j]
                label = None
                if idx == 0:
                    label = "Bearing"
                draw_xbox(ax, x0=xb, w=0.1, h=ro_nodes[j]*2.4, label=label)
        seal_colors = {0: "#4F60BE",
                    1: "#E79D1C",
                    2: "#9806EC",
                    3: "#2789EC"}
        seal_colors = {0: "#A7A6A6",
                    1: "#A7A6A6",
                    2: "#A7A6A6",
                    3: "#A7A6A6"}
        
        seal_types = {0: "1st Imp.",
                    1: "Stage bush",
                    2: "Series Imp.",
                    3: "Balance piston"}
        
        if seals:
            unique_seals = {s.SealNet for s in seals}
            n_type_seal = len(unique_seals)
            ploted_seal = np.zeros(n_type_seal, dtype=bool)
            for idx, s in enumerate(seals):
                j = int(s.node)
                xs = x_nodes[j]
                color_ = seal_colors[int(s.SealNet-1)]
                c = 0.005
                label = None
                if not ploted_seal[int(s.SealNet-1)]:
                    ploted_seal[int(s.SealNet-1)] = True
                    label = seal_types[int(s.SealNet-1)]
                    
                draw_seal(ax, x0=xs, w=max(s.Ls,0.045), ri=s.Ds/2, h=0.04, c=c, label=label,color=color_)
        xmin, xmax = x_nodes.min(), x_nodes.max()
        ax.set_xlim(xmin - 0.02*(xmax-xmin), xmax + 0.02*(xmax-xmin))
        ax.set_xlabel('Axial location (m)')
        ax.set_ylim(-2*rmax, 2*rmax)
        # ax.set_ylabel('Shaft radius (m)')
        ax.xaxis.grid(True, zorder=1)
        ax.set_yticklabels([])
        ax.set_yticks([])

        # return ax
    
    fig.tight_layout()
    # ax1.plot(x_nodes,A_plot,'-', linewidth=2, zorder=5)
    # ax1.plot(x_nodes,-A_plot,'-', linewidth=2, zorder=5)
    # ax1.set_ylabel('Ampliture (um)')
    # ax1.set_xlabel('Axial location (m)')
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    # fig.savefig('fuck.png', dpi=600, bbox_inches="tight")
    # fig.show()
    return fig, ax1





def plot_unbalance_response_rated_speed_all(x_nodes, harmonic_resp, out_path, figsize=figsize_SC_sss,
                                            plot_rotor=False,rotor_elements=None, added_elements=None, 
                                            brgs=None, seals=None, colors=None, ylim=None,
                                            labels=None,unb=None,scale=None):
    _default_rcparams()
    lw = 1
    n_node = x_nodes.shape[0]
    n_dof = n_node*4
    pop, _, _, _ = harmonic_resp.shape
    h = harmonic_resp.squeeze()
    hx = h[:,np.arange(0,n_dof,4)]
    hy = h[:,np.arange(2,n_dof,4)]
    

    t = np.linspace(0,2*np.pi,12)
    qx = np.real(hx.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    qy = np.real(hy.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    
    qx = qx.reshape(pop,n_node,-1)
    qy = qy.reshape(pop,n_node,-1)

    # qx[:,0].reshape(4,-1) = hx
    
    q = np.sqrt(qx**2 + qy**2)

    A_plot = []

    for ip in range(pop):
        i_max = np.argmax(np.max(q[ip],axis=1))
        i_psi = np.argmax(q[ip,i_max,:])
        alpha = np.atan(qy[ip, i_max, i_psi] / qx[ip, i_max, i_psi])

        hx_r = hx[ip] * np.exp(1j * alpha)
        hy_r = hy[ip] * np.exp(1j * alpha)
        theta = 0.0
        A_signed = np.real(hx_r * np.cos(theta) + hy_r * np.sin(theta))

        y_scale_resp = 1e6  # meters -> micrometers
        A_plot.append(A_signed * y_scale_resp)
        if np.sign(A_signed[0]) == -1:
            A_plot[ip] *=-1
    
    if ylim is None:
        ylim=np.ceil(np.max(np.abs(A_plot))/5+1)*5

    fig, ax1 = plt.subplots(figsize=figsize)
    for ip in range(pop):
        ax1.plot(x_nodes,A_plot[ip],'-', linewidth=1)
        # fig, ax2 = plt.subplots()
    if unb is not None:
        unb_ = unb.cases[0]
        unb_node = unb_.node[0]
        unb_mag = unb_.mag[0]
        ax1.plot(x_nodes[unb_node],0,'-', linewidth=2, color='#FF0000')
        # ax2.quiver([x_nodes[unb_node], np.zeros_like(unb_node)], 
        #            [np.zeros_like(unb_mag), np.zeros_like(unb_mag)],
        #            [np.zeros_like(unb_mag),unb_mag])
        # ax2.quiver([0, x_nodes[unb_node]], [0, -unb_mag], 
        #            pivot='tip',scale=ylim*0.8, color='#FF0000')
        ax1.quiver([x_nodes[unb_node]], [0], [0], [-unb_mag], 
                    angles='xy',
                    pivot='tip',color='#FF0000',
                    scale=scale,linewidth=0.5)
        labels.append('Unb.')
        # ax1.plot(x_nodes,-A_plot[ip],'-', linewidth=2, zorder=5)
        
    ax1.set_ylabel('Ampliture (um)')
    ax1.set_xlabel('Axial location (m)')
    ax1.set_ylim(-ylim, ylim)
    
    if labels is not None:
        ax1.legend(labels)

    if plot_rotor:
        from matplotlib.patches import Rectangle, Circle

        if colors is None:
            colors = {
                'outer': "#2133BB",
                'outer': "#8D8D8D",
                'inner': '0.35',
                'added': 'tab:purple',
                'spring_brg': 'tab:red',
                'damper_brg': 'tab:green',
                'spring_seal': 'tab:orange',
                'damper_seal': 'tab:blue',
                'center': "#A7A7A7",
            }
        
        alpha = 1.0
        vec_L = np.array([e.L for e in rotor_elements], dtype=float)
        x_nodes = np.concatenate([[0.0], np.cumsum(vec_L)])
        # Create axes if needed
        ax = ax1.twinx()
        created_ax = True

        ax.axhline(0, color=colors['center'], zorder=1, linestyle='--')

        rmax = 0.0
        for i, e in enumerate(rotor_elements):
            x0, L = x_nodes[i], float(e.L)
            ro = 0.5 * float(e.Od)
            ri = 0.5 * float(e.Id)
            rmax = max(rmax, ro)

            if ro > 0 and L > 0:
                rect = Rectangle((x0, -ro), width=L, height=2*ro,
                                fill=False, edgecolor=colors['outer'], zorder=3,alpha=alpha)
                ax.add_patch(rect)
        
        def draw_disk(ax, x0, h, r, label=None):
            arm_x = [x0, x0]
            arm_y = [-h/2, h/2]
            color = '#0D7441'
            color = '#737976'
            color = colors['outer']
            ax.plot(arm_x, arm_y, color=color, lw=lw, label=label, alpha=alpha)
            
            circ = Circle((x0,h/2+r),r , color=color, fill=False, lw=lw, alpha=alpha)
            ax.add_patch(circ)
            circ = Circle((x0,-h/2-r),r , color=color, fill=False, lw=lw, alpha=alpha)
            ax.add_patch(circ)

        r_disk = 0.025
        added_D_max = 0.0
        if added_elements:
            for a in added_elements: 
                added_D_max = max(added_D_max,a.Od)
            added_D_max *= 1.2
            for idx, a in enumerate(added_elements): 
                x0 = x_nodes[int(a.node)]
                added_D_max = max(added_D_max,a.Od)
                label = None
                if idx == 0:
                    label = "Disk"
                draw_disk(ax, x0=x0, h=added_D_max, r=r_disk, label=label)
        rmax = max(rmax, added_D_max/2.0)

        n_ele = len(rotor_elements)
        ro_nodes = np.zeros_like(x_nodes)
        for i in range(n_ele + 1):
            left_ro = 0.5 * rotor_elements[i-1].Od if i > 0 else 0.5 * rotor_elements[0].Od
            right_ro = 0.5 * rotor_elements[i].Od if i < n_ele else 0.5 * rotor_elements[-1].Od
            ro_nodes[i] = max(left_ro, right_ro)
        if added_elements:
            for a in added_elements:
                ro_nodes[int(a.node)] = max(ro_nodes[int(a.node)], 0.5*float(a.Od))

                
        def draw_xbox(ax, x0, w, h, label=None):
            rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
                            fill=True, facecolor=colors['outer'], edgecolor=colors['outer'], lw=lw, label=label, alpha=alpha)
            # rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
            #                 fill=True, facecolor='#B8BEBB', edgecolor="#737976", lw=lw, label=label, alpha=alpha)
            ax.add_patch(rect)
            # xx = [x0-w/2, x0+w/2]
            # yx = [-h/2, h/2]
            # ax.plot(xx, yx, color="#737976", lw=0.9*lw, alpha=alpha)
            # xx = [x0-w/2, x0+w/2]
            # yx = [h/2, -h/2]
            # ax.plot(xx, yx, color='#737976', lw=0.9*lw, alpha=alpha)
            
        def draw_seal(ax, x0, w, ri, h, c, color=None,label=None):
            
            if color is None:
                color = "#4F60BE"
            
            # rect = Rectangle((x0-w/2, -ro-c), width=w, height=h,
            #                 fill=True, facecolor="#4F60BE", edgecolor="#4F60BE", lw=lw, label=label)
            rect = Rectangle((x0-w/2, -ri-c-h), width=w, height=h,
                            fill=True, facecolor=color, edgecolor=color, lw=lw, label=label, alpha=alpha)
            ax.add_patch(rect)
            rect = Rectangle((x0-w/2, ri+c), width=w, height=h,
                            fill=True, facecolor=color, edgecolor=color, lw=lw, alpha=alpha)
            ax.add_patch(rect)
            
        if brgs:
            for idx, b in enumerate(brgs):
                j = int(b.node)
                xb = x_nodes[j]
                label = None
                if idx == 0:
                    label = "Bearing"
                draw_xbox(ax, x0=xb, w=0.1, h=ro_nodes[j]*2.4, label=label)
        seal_colors = {0: "#4F60BE",
                    1: "#E79D1C",
                    2: "#9806EC",
                    3: "#2789EC"}
        seal_colors = {0: colors['outer'],
                    1: colors['outer'],
                    2: colors['outer'],
                    3: colors['outer']}
        
        seal_types = {0: "1st Imp.",
                    1: "Stage bush",
                    2: "Series Imp.",
                    3: "Balance piston"}
        
        if seals:
            unique_seals = {s.SealNet for s in seals}
            n_type_seal = len(unique_seals)
            ploted_seal = np.zeros(n_type_seal, dtype=bool)
            for idx, s in enumerate(seals):
                j = int(s.node)
                xs = x_nodes[j]
                color_ = seal_colors[int(s.SealNet-1)]
                c = 0.005
                label = None
                if not ploted_seal[int(s.SealNet-1)]:
                    ploted_seal[int(s.SealNet-1)] = True
                    label = seal_types[int(s.SealNet-1)]
                    
                draw_seal(ax, x0=xs, w=max(s.Ls,0.045), ri=s.Ds/2, h=0.04, c=c, label=label,color=color_)
        xmin, xmax = x_nodes.min(), x_nodes.max()
        ax.set_xlim(xmin - 0.02*(xmax-xmin), xmax + 0.02*(xmax-xmin))
        ax.set_xlabel('Axial location (m)')
        ax.set_ylim(-2*rmax, 2*rmax)
        # ax.set_ylabel('Shaft radius (m)')
        ax.xaxis.grid(True, zorder=1)
        ax.set_yticklabels([])
        ax.set_yticks([])

        # return ax
    
    ax2 = ax.twinx()
    for ip in range(pop):
        # ax2.plot(x_nodes,A_plot[ip],'-', linewidth=2, marker='o', markersize=3)
        ax2.plot(x_nodes,A_plot[ip],'-', linewidth=2)
        # ax1.plot(x_nodes,-A_plot[ip],'-', linewidth=2, zorder=5)
    # ax1.set_ylabel('Ampliture (um)')
    # ax1.set_xlabel('Axial location (m)')
    ax2.set_ylim(-ylim, ylim)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    
    # fig, ax2 = plt.subplots()
    if unb is not None:
        unb_ = unb.cases[0]
        unb_node = unb_.node[0]
        unb_mag = unb_.mag[0]
        # ax2.quiver([x_nodes[unb_node], np.zeros_like(unb_node)], 
        #            [np.zeros_like(unb_mag), np.zeros_like(unb_mag)],
        #            [np.zeros_like(unb_mag),unb_mag])
        # ax2.quiver([0, x_nodes[unb_node]], [0, -unb_mag], 
        #            pivot='tip',scale=ylim*0.8, color='#FF0000')
        ax2.quiver([x_nodes[unb_node]], [0], [0], [-unb_mag], 
                    angles='xy',
                    pivot='tip',color='#FF0000',
                    scale=scale)
    
    # ax1.plot(x_nodes,-A_plot,'-', linewidth=2, zorder=5)
    # ax1.set_ylabel('Ampliture (um)')
    # ax1.set_xlabel('Axial location (m)')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    # fig.savefig('fuck.png', dpi=600, bbox_inches="tight")
    # fig.show()
    return fig, ax1








def pareto_plot_PCP(F, names, idx: Optional[np.ndarray] = None, out_path="pareto_PCP.png", figsize=figsize_DC):
    _default_rcparams()
    # import matplotlib.pyplot as plt

    plot = PCP(
        # title=("Pareto Front (Objectives)", {'pad': 30}),
        labels=names,
        figsize=figsize
        )
    plot.set_axis_style(color="grey", alpha=0.5)
    plot.add(F, color="grey", alpha=0.3)

    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    colors = default_cycle.by_key().get('color')
    
    # Normalize idx into indices, allow None/empty
    indices = np.array([], dtype=int)
    if idx is not None:
        if isinstance(idx, (int, np.integer)):
            indices = np.array([int(idx)])
        else:
            idx_arr = np.asarray(idx)
            if idx_arr.dtype == bool:
                indices = np.flatnonzero(idx_arr)
            elif idx_arr.size > 0:
                indices = idx_arr.astype(int).ravel()
    for i, idc in enumerate(indices):
        plot.add(np.atleast_2d(F[idc]), linewidth=3, linestyle='-',
                color=colors[i % len(colors)])

    plot.save(out_path, dpi=600, bbox_inches="tight")

    plot.show()
    return plot







def pareto_plot_RAD(F, names, idx: Optional[np.ndarray] = None, figsize=figsize_DC_tall,scale=True):
    if scale:
        from sklearn.preprocessing import MinMaxScaler
        F_scaled = F
        for of in range(F.shape[1]):
            scaler = MinMaxScaler()
            scaler.fit(F_scaled[:,of].reshape(-1,1))
            F_scaled[:,of] = scaler.transform(F_scaled[:,of].reshape(-1,1)).squeeze()
        F = F_scaled
    
    plot = Radviz(
        # title="Optimization",
        # legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
        labels=names,
        endpoint_style={"s": 60, "color": "green"},
        figsize=figsize
        )
    plot.set_axis_style(color="black", alpha=1.0)
    plot.add(F, color="grey", s=10)
    
    # Normalize idx into indices, allow None/empty
    indices = np.array([], dtype=int)
    if idx is not None:
        if isinstance(idx, (int, np.integer)):
            indices = np.array([int(idx)])
        else:
            idx_arr = np.asarray(idx)
            if idx_arr.dtype == bool:
                indices = np.flatnonzero(idx_arr)
            elif idx_arr.size > 0:
                indices = idx_arr.astype(int).ravel()

    
    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    colors = default_cycle.by_key().get('color')

    for i, idc in enumerate(indices):
        color = colors[(i - 3) % len(colors)]
        plot.add(np.atleast_2d(F[idc]), linewidth=2,marker='*')
    plot.save("pareto_RAD.png", dpi=600, bbox_inches="tight")
    plot.show()
    return plot


def plot_2factor(F_pop,F_par,ip,idx=None,figsize=figsize_SC_one_third_s, xlim=None, ylim=None):
    plt.figure(figsize=figsize)
    plt.scatter(F_pop[:, ip[0]], F_pop[:, ip[1]], s=10, alpha=0.5,facecolors='none', edgecolors='grey', label="Cand.")
    if idx is not None:
        idx_not_sel = np.setdiff1d(np.arange(0,F_par.shape[0]),idx)
        plt.scatter(F_par[idx_not_sel, ip[0]], F_par[idx_not_sel, ip[1]], s=15, facecolors="#FB7AE1", edgecolors="#FB7AE1", label="Pareto")
        for i, sel in enumerate(idx):
            print(F_par[sel, ip[0]])
            plt.scatter(F_par[sel, ip[0]], F_par[sel, ip[1]], marker='*', s=60, label=f'Design {i+1}')
    else:
        plt.scatter(F_par[:, ip[0]], F_par[:, ip[1]], s=15, facecolors="#FB7AE1", edgecolors="#FB7AE1", label="Pareto")
    
    plt.xlabel('Seal leakage (kg/s)')
    plt.ylabel('Bearing power loss (W)')
    plt.legend(loc=1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig("pareto_2factor.png", dpi=600, bbox_inches="tight")
    plt.show()


#%%

w_range = np.array([500, 7000]) * np.pi / 30
n_w = 11
w_vec = np.linspace(w_range[0], w_range[1], n_w)
w_oper = 3500 * np.pi / 30

bs_params = {
    'mu_brg': 0.025,
    'mu_seal': 1.4e-3,
    'rho_seal': 850,
}

ctx = build_rotor_and_models(bs_params)

rotor_elements = ctx['rotor_elements']
added_elements = ctx['added_elements']
brgs = ctx['brgs']
seals = ctx['seals']
unb = ctx['unb']

vec_L = np.array([e.L for e in rotor_elements], dtype=float)
x_nodes = np.concatenate([[0.0], np.cumsum(vec_L)])

# --- Initial/reference design (programmatic, matches n_brg/n_seal) ---
def build_initial_X(n_brg):
    X_brg = np.tile([2, 15], (n_brg, 1))
    X_seal = np.concatenate([np.tile([285, 285, 6.5], 2),
                            np.tile([300, 300, 0, 300, 300, 7.5], 6),
                            [285, 285, 6.5]]
                        )

    return np.concatenate([X_brg.ravel(), X_seal.ravel()], axis=0)[None, :]

X_init = build_initial_X(ctx['n_brg']).astype(float)

eig_init, amp_init, loss_brg_init, loss_brg_full_init, leak_init, F_init, peak_af_list_init, peak_centers_list_init, logdec_init, amp_ratio_brg_init, amp_ratio_seal_init, harmonic_init = analyze_design(X_init, ctx, w_vec)

#%%
plot_campbell(eig_init, w_vec, out_path='ini_campbell.png')
plot_logdec(logdec_init, w_vec, out_path='ini_logdec.png')
plot_unbalance_response_rpm(amp_init, w_vec, ctx, out_path='ini_unb.png', nodes='key', figsize=figsize_SC_sss, show_lgd=True)

#%%

w_op = np.array([w_oper])

eig_init_oper, amp_init_oper, loss_brg_init_oper, loss_brg_full_init_oper, leak_init_oper, F_init_oper, peak_af_list_init_oper, peak_centers_list_init_oper, logdec_init_oper, amp_ratio_brg_init_oper, amp_ratio_seal_init_oper, harmonic_init_oper = analyze_design(X_init, ctx, w_op)

plot_unbalance_response_rated_speed(x_nodes, harmonic_resp=harmonic_init_oper, figsize=figsize_SC_ss, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals, out_path="ini_unb_shaft.png",ylim=25)









#%%
d = np.load('checkpoints/latest.npz')
X_pop, F_pop = d['pop_X'], d['pop_F']
X_pareto, F_pareto = d['opt_X'], d['opt_F']

n_brg = ctx['n_brg']

obj_names = [r'$\dot{\mathrm{m}}_{total}$', r'$\mathrm{PL}_{brg}$', r'$\mathrm{AF}_{\max}$', r'$\delta_{\min}$', r'$\mathrm{Ampl.}_{brg}$', r'$\mathrm{Ampl.}_{seal}$']
pareto_signs = np.array([1, 1, 1, -1, 1, 1])
F_pareto_corrected = F_pareto * pareto_signs
sel0 = np.argsort(F_pareto[:, 0])
sel1 = np.argsort(F_pareto[:, 1])

sel = sel0[:5]
sel = sel[[2,1,3]]
# F_pareto[sel,:]

pareto_plot_PCP(F=F_pareto_corrected, names=obj_names, idx=sel, figsize=figsize_SC_two_third_ss)
pareto_plot_RAD(F=F_pareto_corrected, names=obj_names, idx=sel)
plot_2factor(F_pop=F_pop,F_par=F_pareto_corrected,ip=[0, 1],idx=sel,figsize=figsize_SC_one_third_ss, xlim=[22,30], ylim=[1000,4000])






#%%
for i, case in enumerate(sel):
    X = X_pareto[case,:].reshape(1,-1)
    eig, amp, loss_brg, loss_brg_full, leak, F_init, peak_af_list, peak_centers_list, logdec, amp_ratio_brg, amp_ratio_seal, _ = analyze_design(X, ctx, w_vec)
    
    plot_campbell(eig, w_vec, out_path=f'opt_cb_{i}.png')
    plot_logdec(logdec, w_vec, out_path=f'opt_logdec_{i}.png')
    plot_unbalance_response_rpm(amp, w_vec, ctx, out_path=f'opt_unb_{i}.png', nodes='key', figsize=figsize_SC_sss, show_lgd=True)
    
    _, _, _, _, _, _, _, _, _, _, _, harmonic_oper= analyze_design(X, ctx, w_op)
    plot_unbalance_response_rated_speed(x_nodes, harmonic_resp=harmonic_oper, out_path=f'opt_unb_shaft{i}.png', figsize=figsize_SC_sss, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals,ylim=25)


#%%

# choose by minimum total leakage as an example
idx = int(np.argmin(F_pareto[:, 0])) if len(F_pareto) else 0
X_opt = X_pareto[idx:idx+1, :]

plot_bearing_id_hist(X_pop, X_pareto, n_brg, figsize=figsize_SC_ss)



X = np.concatenate((X_init,X_pareto[sel,:]))

_, _, _, _, _, _, _, _, _, _, _, harmonic_oper_sel= analyze_design(X, ctx, w_op)

# np.argmax(F_pareto[:,1])
# F_pareto[45,:]
# X_pareto[45,:]


labels = ['Ini.','1','2','3']
plot_unbalance_response_rated_speed_all(x_nodes, harmonic_resp=harmonic_oper_sel, out_path=f'opt_unb_shaft_all.png', figsize=figsize_SC_s, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals,ylim=25,labels=labels,unb=unb,scale=70000)




gens, bests = [], []
# check the gen in latest.npz and read data up to that generation only (faster than scanning all files)
try:
    with np.load('checkpoints/latest.npz') as ld:
        latest_gen = int(ld['gen']) if 'gen' in ld.files else None
except Exception:
    latest_gen = None

if latest_gen is not None and latest_gen >= 0:
    for gnum in range(latest_gen + 1):
        f = f'checkpoints/gen_{gnum:04d}.npz'
        if not os.path.exists(f):
            continue
        with np.load(f) as nd:
            if 'pop_F' in nd.files:
                pop_F_signed = nd['pop_F'] * pareto_signs
                best = np.min(pop_F_signed, axis=0)
            elif 'opt_F' in nd.files:
                pop_F_signed = nd['opt_F'] * pareto_signs
                best = np.min(pop_F_signed, axis=0)
            else:
                continue
        gens.append(gnum)
        bests.append(best)

i_obj = [0, 2, 3, 4, 5, 1]

if gens:
    order = np.argsort(gens)
    ug = np.asarray(gens, dtype=int)[order]
    bv = np.asarray(bests, dtype=float)[order]
    fig, ax = plt.subplots(figsize=figsize_SC_ss)
    ax.plot(ug, bv[:,[0,2,3,4,5]], '-', ms=3)
    ax.plot(ug, bv[:,[1]], '-', color="#FB7AE1")
    ax.set_ylim(0,47)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best objective')
    ax2 = ax.twinx()
    ax2.plot(ug,bv[:,[0,2,3,4,5]], '-')
    ax2.plot(ug,bv[:,1], '-', color="#FB7AE1",zorder=0)
    ax2.set_ylim(1300,1800)
    ax2.legend(np.array(obj_names)[i_obj],framealpha=1,loc=1,ncols=2)
    fig.tight_layout()
    fig.savefig('obj_history.png', dpi=600, bbox_inches='tight')
    fig.show()
