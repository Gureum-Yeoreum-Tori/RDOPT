#%%
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
figsize_DC_one_third = (2.4, 1.8) # double column, big
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

def calc_ellip(Ux, Uy):
    # amplitude [pop, n_w, n_node]
    # n_node = ctx['n_node']
    # idx_x = np.arange(n_node) * 4
    # idx_y = idx_x + 2
    # Ux = harmonic[:, :, idx_x, 0]
    # Uy = harmonic[:, :, idx_y, 0]

    xr = np.real(Ux)
    xi = np.imag(Ux)
    yr = np.real(Uy)
    yi = np.imag(Uy)

    ra1 = np.sqrt( (xr**2 + xi**2 +yr**2 + yi**2)/2 + np.sqrt((xr**2 - xi**2 +yr**2 - yi**2)**2/4 + (xr*xi + yr*yi)**2) )
    rb1 = np.sqrt( (xr**2 + xi**2 +yr**2 + yi**2)/2 - np.sqrt((xr**2 - xi**2 +yr**2 - yi**2)**2/4 + (xr*xi + yr*yi)**2) )
    alpha1 = 1/2 * np.arctan( 2*(xr*yr+xi*yi) / (xr**2+xi**2-yr**2-yi**2))
    
    Bx = np.stack([Ux.real, -Ux.imag], axis=-1)      # (..., 2)
    By = np.stack([Uy.real, -Uy.imag], axis=-1)      # (..., 2)
    B  = np.stack([Bx, By], axis=-2)                 # (..., 2, 2)

    U, S, Vt = np.linalg.svd(B)                      # batched SVD
    ra2 = S[..., 0]                                    # 진짜 최대 진폭 (장반경)
    rb2 = S[..., 1]                                    # 최소 진폭 (단반경)
    
    psi   = np.arctan2(U[..., 1, 0], U[..., 0, 0])   # 장축 각
    tstar = np.arctan2(Vt[..., 0, 1], Vt[..., 0, 0]) # 피크 위상
    alpha2 = -tstar
    
    return ra1, rb1, alpha1, ra2, rb2, alpha2


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

    eigvals, eigvecs = eig_batch(M=mat_M, K_all=K_all, Ceff_all=Ceff_all, track=True)

    harmonic = unbalance_response_batch_cpu_parallel(
        M=mat_M, unb=unb, K_all=K_all, Ceff_all=Ceff_all, w_vec=w_vec
    )
    
    F = np.zeros((pop, 6), dtype=float) 
    idx_op = int(np.argmin(np.abs(w_vec - w_oper)))
    F[:, 0] = leak
    loss_brg = loss_brg_full[:,:,:,idx_op].sum(axis=1).squeeze()
    F[:, 1] = loss_brg
    
    # amplitude [pop, n_w, n_node]
    n_node = ctx['n_node']
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2
    Ux = harmonic[:, :, idx_x, 0]
    Uy = harmonic[:, :, idx_y, 0]
    
    amp, _, _, _, _, _ = calc_ellip(Ux, Uy)

    # xr = np.real(Ux)
    # xi = np.imag(Ux)
    # yr = np.real(Uy)
    # yi = np.imag(Uy)

    # amp = np.sqrt( (xr**2 + xi**2 +yr**2 + yi**2)/2 + np.sqrt((xr**2 - xi**2 +yr**2 - yi**2)**2/4 + (xr*xi + yr*yi)**2) )
    # b = np.sqrt( (xr**2 + xi**2 +yr**2 + yi**2)/2 - np.sqrt((xr**2 - xi**2 +yr**2 - yi**2)**2/4 + (xr*xi + yr*yi)**2) )
    # alpha = 1/2 * np.arctan( 2*(xr*yr+xi*yi) / (xr**2+xi**2-yr**2-yi**2))
    
    # Bx = np.stack([Ux.real, -Ux.imag], axis=-1)      # (..., 2)
    # By = np.stack([Uy.real, -Uy.imag], axis=-1)      # (..., 2)
    # B  = np.stack([Bx, By], axis=-2)                 # (..., 2, 2)

    # U, S, Vt = np.linalg.svd(B)                      # batched SVD
    # amp = S[..., 0]                                    # 진짜 최대 진폭 (장반경)
    # b_amp = S[..., 1]                                    # 최소 진폭 (단반경)
    

    # # 장축 방향과 최대가 생기는 위상(원하면)
    # psi   = np.arctan2(U[..., 1, 0], U[..., 0, 0])   # 장축 각
    # tstar = np.arctan2(Vt[..., 0, 1], Vt[..., 0, 0]) # 피크 위상
    # alpha2 = -tstar
    
    eps = 1e-18
    AF_max = np.zeros(pop, dtype=float)
    # For separation margin: collect peak centers and AF per peak for nearest peak logic
    peak_centers_list = [[] for _ in range(pop)]  # angular speed of peaks
    peak_af_list = [[] for _ in range(pop)]       # AF at those peaks
    w = w_vec
    brg_nodes = np.array([b.node for b in brgs], dtype=int) 
    seal_nodes = np.array([s.node for s in seals], dtype=int)
    cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))
    
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
    alpha_eig = np.real(eigvals[:,:,:k_use])  # [pop, n_w, 2n]
    beta  = np.imag(eigvals[:,:,:k_use])

    logdec = -2 * np.pi * alpha_eig / np.sqrt(alpha_eig**2 + beta**2)
    min_logdec = np.min(logdec, axis=(1, 2)) # [pop]
    F[:, 3] = -min_logdec

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

    return eigvals, amp, loss_brg, loss_brg_full, leak, F, peak_af_list, peak_centers_list, logdec, amp_ratio_brg, amp_ratio_seal, harmonic, eigvecs



def plot_unbalance_response_rpm(har_resp, w_vec, ctx, out_path, nodes='key', smooth_spline=True, figsize=figsize_DC, show_lgd=False):
    _default_rcparams()
    import matplotlib as mlp
    from cycler import cycler
    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))
    
    n_node = ctx['n_node']
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2
    Ux = har_resp[:, :, idx_x, 0]
    Uy = har_resp[:, :, idx_y, 0]
    
    amp, _, _, _, _, _ = calc_ellip(Ux, Uy)
    
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
    ax.set_ylim(0, 50)
    ax.set_xlim(0, rpm.max())
    start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    ax.grid(True, alpha=0.3)
    if show_lgd:
        ax.legend()
        # long legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                borderaxespad=0., ncol=2,
                handlelength=1.2)

    fig.tight_layout()
    
    fig.savefig(out_path, dpi=600, bbox_inches='tight')



def plot_unbalance_response_rated_speed(x_nodes, har_resp, out_path, figsize=figsize_SC_sss, plot_rotor=False,
                                        rotor_elements=None, added_elements=None, brgs=None, seals=None, colors=None, ylim=None, unb=None):
    _default_rcparams()
    lw = 1
    n_dof = x_nodes.shape[0]*4
    h = har_resp.squeeze() # pop 없음 
    Ux = h[np.arange(0,n_dof,4)]
    Uy = h[np.arange(2,n_dof,4)]
    
    amp, _, _, _, _, _ = calc_ellip(Ux, Uy)
    
    t = np.linspace(0,2*np.pi,12)
    qx = np.real(Ux.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    qy = np.real(Uy.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))

    q = np.sqrt(qx**2 + qy**2)
    i_max = np.argmax(np.max(q,axis=1))
    i_psi = np.argmax(q[i_max,:])
    alpha = np.atan(qy[i_max,i_psi]/qx[i_max,i_psi])

    hx_r = Ux * np.exp(1j * alpha)
    hy_r = Uy * np.exp(1j * alpha)
    theta = 0.0
    A_signed = np.real(hx_r * np.cos(theta) + hy_r * np.sin(theta))
    
    A_signed = amp * np.sign(A_signed)
    A_signed = amp

    # Use response as the Y-axis scale (in micrometers)
    y_scale_resp = 1e6  # meters -> micrometers
    A_plot = A_signed * y_scale_resp
    if np.sign(A_plot[0]) == -1:
        A_plot *=-1
    
    if ylim is None:
        ylim=np.ceil(np.max(np.abs(A_plot))/5+1)*5

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(x_nodes,A_plot,'-o', linewidth=2, zorder=5)
    # ax1.plot(x_nodes,-A_plot,'-o', linewidth=2, zorder=5)
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
    ax2 = ax.twinx()
    ax2.plot(x_nodes,A_plot,'-o', linewidth=2)
    ax2.set_ylim(-ylim, ylim)
    # ax1.set_yticklabels([])
    # ax1.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    # ax1.set_ylabel('')
    
    # ax2.plot(x_nodes,-A_plot,'-', linewidth=2)
    fig.tight_layout()
    # ax1.plot(x_nodes,A_plot,'-', linewidth=2, zorder=5)
    # ax1.plot(x_nodes,-A_plot,'-', linewidth=2, zorder=5)
    # ax1.set_ylabel('Ampliture (um)')
    # ax1.set_xlabel('Axial location (m)')
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    # fig.savefig('fuck.png', dpi=600, bbox_inches="tight")
    # fig.show()
    return A_plot, fig, ax1


def plot_unbalance_response_rated_speed_all(x_nodes, har_resp, out_path, figsize=figsize_SC_sss,
                                            plot_rotor=False,rotor_elements=None, added_elements=None, 
                                            brgs=None, seals=None, colors=None, ylim=None,
                                            labels=None,unb=None,scale=None):
    _default_rcparams()
    lw = 1
    n_node = x_nodes.shape[0]
    n_dof = n_node*4
    pop, _, _, _ = har_resp.shape
    h = har_resp.squeeze()

    Ux = h[:,np.arange(0,n_dof,4)]
    Uy = h[:,np.arange(2,n_dof,4)]

    amp, _, _, _, _, _ = calc_ellip(Ux, Uy)
    
    
    A_plot = []
    y_scale_resp = 1e6  # meters -> micrometers
    
    for ip in range(pop):
        amp_ = amp[ip]
        U = np.stack([Ux[ip], Uy[ip]])                  # (2, n)
        j0 = np.argmax(amp_)                 # 진폭 최대 인덱스
        c = np.sum(np.conjugate(U[:, j0, None]) * U, axis=0)   # U_j^H U_j0
        dphi = np.angle(c).squeeze()
        signed_amp = amp_ * np.sign(np.cos(dphi))
        signed_amp[np.abs(np.cos(dphi)) < 0.001] = 0.0
        A_plot.append(signed_amp * y_scale_resp)
        if np.sign(signed_amp[7]) == 1:
            A_plot[ip] *=-1


    # t = np.linspace(0,2*np.pi,12)
    # qx = np.real(Ux.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))
    # qy = np.real(Uy.reshape(-1,1) * np.exp(1j*t.reshape(1,-1)))

    # qx = qx.reshape(pop,n_node,-1)
    # qy = qy.reshape(pop,n_node,-1)

    # # qx[:,0].reshape(4,-1) = hx

    # q = np.sqrt(qx**2 + qy**2)

    # A_plot = []

    # for ip in range(pop):
    #     i_max = np.argmax(np.max(q[ip],axis=1))
    #     i_psi = np.argmax(q[ip,i_max,:])
    #     alpha = np.atan(qy[ip, i_max, i_psi] / qx[ip, i_max, i_psi])

    #     hx_r = Ux[ip] * np.exp(1j * alpha)
    #     hy_r = Uy[ip] * np.exp(1j * alpha)
    #     theta = 0.0
    #     A_signed = np.real(hx_r * np.cos(theta) + hy_r * np.sin(theta))
        
    #     # A_signed = amp[ip] * np.sign(A_signed)

    #     y_scale_resp = 1e6  # meters -> micrometers
    #     A_plot.append(A_signed * y_scale_resp)
    #     if np.sign(A_signed[0]) == -1:
    #         A_plot[ip] *=-1
    
    if ylim is None:
        ylim=np.ceil(np.max(np.abs(A_plot))/5+1)*5
        
    A_plot = np.array(A_plot)

    fig, ax1 = plt.subplots(figsize=figsize)
    label_plot = labels
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
        label_plot.append('Unb.')
        # ax1.plot(x_nodes,-A_plot[ip],'-', linewidth=2, zorder=5)
        
    ax1.set_ylabel('Ampliture (um)')
    ax1.set_xlabel('Axial location (m)')
    ax1.set_ylim(-ylim, ylim)
    
    if label_plot is not None:
        ax1.legend(label_plot)

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


#%%

w_range = np.array([500, 7000]) * np.pi / 30
n_w = 14
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

eig_init, amp_init, loss_brg_init, loss_brg_full_init, leak_init, F_init, peak_af_list_init, peak_centers_list_init, logdec_init, amp_ratio_brg_init, amp_ratio_seal_init, harmonic_init, eigvecs_init = analyze_design(X_init, ctx, w_vec)

plot_unbalance_response_rpm(harmonic_init, w_vec, ctx, out_path='ini_unb.png', nodes='key', figsize=figsize_SC_sss, show_lgd=True)


#%%

w_op = np.array([w_oper])

eig_init_oper, amp_init_oper, loss_brg_init_oper, loss_brg_full_init_oper, leak_init_oper, F_init_oper, peak_af_list_init_oper, peak_centers_list_init_oper, logdec_init_oper, amp_ratio_brg_init_oper, amp_ratio_seal_init_oper, harmonic_init_oper, _ = analyze_design(X_init, ctx, w_op)


plot_unbalance_response_rated_speed(x_nodes, har_resp=harmonic_init_oper, figsize=figsize_SC_ss, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals, out_path="ini_unb_shaft.png",ylim=25)



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

# sel = sel0[:5]
sel = sel0[[0,1,4]]


X = np.concatenate((X_init,X_pareto[sel,:]))

_, _, _, _, _, F, _, _, _, _, _, harmonic_oper_sel, _= analyze_design(X, ctx, w_op)


labels = ['Ini.','1','2','3']
plot_unbalance_response_rated_speed_all(x_nodes, har_resp=harmonic_oper_sel, out_path=f'opt_unb_shaft_all.png', figsize=figsize_SC_s, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals,ylim=60,labels=labels,unb=unb,scale=50000)


eig, amp, loss_brg, loss_brg_full, leak, F, peak_af_list, peak_centers_list, logdec, amp_ratio_brg, amp_ratio_seal, harmonic, eigvecs = analyze_design(X, ctx, w_vec)
for i, case in enumerate(sel):
    plot_unbalance_response_rpm(np.expand_dims(harmonic[i,:,:],axis=0), w_vec, ctx, out_path=f'opt_unb_{i}.png', nodes='key', figsize=figsize_DC_one_third, show_lgd=False)


#%%
# n_node = ctx['n_node']
# idx_x = np.arange(n_node) * 4
# idx_y = idx_x + 2
# for i in range(4):
#     Ux = harmonic_oper_sel[i, :, idx_x, 0]
#     Uy = harmonic_oper_sel[i, :, idx_y, 0]
#     amp, _, _, _, _, _ = calc_ellip(Ux, Uy)
#     plt.figure()
#     # plt.plot(x_nodes,amp*1e6)
    
#     plt.plot(x_nodes,amp,'r-',x_nodes,np.abs(Ux),x_nodes,np.abs(Uy))
#     plt.show()
    
#     # plot_unbalance_response_rated_speed(x_nodes, har_resp=harmonic_oper_sel[i,:,:,:], figsize=figsize_SC_ss, plot_rotor=True, rotor_elements=rotor_elements,added_elements=added_elements,brgs=brgs,seals=seals, out_path="ini_unb_test.png",ylim=50)

# amp, _, alpha, _, _, alpha2 = calc_ellip(Ux, Uy)

# amp = amp.squeeze()
# U = np.stack([Ux, Uy])                   # (2, n)
# j0 = np.argmax(amp)                 # 진폭 최대 인덱스
# c = np.sum(np.conjugate(U[:, j0, None]) * U, axis=0)   # U_j^H U_j0
# dphi = np.angle(c).squeeze()
# signed_amp = amp * np.sign(np.cos(dphi))
# # 선택: 거의 90°이면 0 처리
# signed_amp[np.abs(np.cos(dphi)) < 0.1] = 0.0

# plt.plot(x_nodes,signed_amp,x_nodes,amp)

#%%

#%%

import matplotlib.pyplot as plt
from cycler import cycler
def plot_unbalance_legend(ctx, out_path='unb_legend.png', figsize=figsize_DC, ncol=2):
    import matplotlib as mlp
    from cycler import cycler
    import matplotlib.pyplot as plt
    import numpy as np

    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))

    brgs = ctx['brgs']
    seals = ctx['seals']
    unb = ctx['unb']

    bearing_nodes = np.array([b.node for b in brgs], dtype=int) if len(brgs) else np.array([], dtype=int)
    seal_nodes = np.array([s.node for s in seals], dtype=int) if len(seals) else np.array([], dtype=int)
    unb_nodes = np.unique(np.array([n for c in unb.cases for n in c.node], dtype=int)) if getattr(unb, 'cases', None) else np.array([], dtype=int)
    sel = np.unique(np.concatenate([bearing_nodes, seal_nodes, unb_nodes]))

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

    fig, ax = plt.subplots(figsize=figsize)
    for iu, (n, sty) in enumerate(zip(sel, cycle(sty_cycle))):
        lbl = f"Node {n}, {category(n)}"
        ax.plot([], [], label=lbl, **sty)

    ax.legend(bbox_to_anchor=(0.5,0.5), loc='center',
            borderaxespad=0., ncol=6, frameon=False)
    ax.axis("off")
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

plot_unbalance_legend(ctx,figsize=(5,0.48))

#%%






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
    
    ax.axhline(0.1, color='r', lw=1.2, linestyle='--')
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'$\delta$')
    ax.set_xlim(0, x_rpm.max())
    ax.set_ylim(-0.5, 4.5)
    start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 1000))
    # ax.legend(loc=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches='tight')




for i, case in enumerate(sel):
    plot_logdec(logdec[i], w_vec, out_path=f'opt_logdec_{i}.png',figsize=figsize_DC_one_third)


# %%

import matplotlib.pyplot as plt
from cycler import cycler
def plot_logdec_legend(ctx, out_path='logdec_legend.png', figsize=figsize_DC,n_mode=4):
    import matplotlib as mlp
    from cycler import cycler
    import matplotlib.pyplot as plt
    import numpy as np

    default_cycle = mlp.rcParams.get('axes.prop_cycle')    
    sty_cycle = cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', default_cycle.by_key().get('color'))

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(n_mode):
        if k == 0:
            suffix='st'
        elif k == 1:
            suffix='nd'
        elif k == 2:
            suffix='rd'
        else:
            suffix='th'
        
        if n_w >= 3:
            ax.plot([], [], label=f'{k+1}{suffix} Mode')
        else:
            ax.plot([], [], label=f'{k+1}{suffix} Mode')

    ax.legend(bbox_to_anchor=(0.5,0.5), loc='center',
            borderaxespad=0., ncol =4, frameon=False)
    ax.axis("off")
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig)

plot_logdec_legend(ctx,figsize=(5,0.16))
# %%
