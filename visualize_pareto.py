#%%
import numpy as np
import matplotlib.pyplot as plt
from pymoo.visualization.pcp import PCP
from pymoo.visualization.radviz import Radviz
from typing import Optional
import os
from import_data import rotor_import
from cycler import cycler
from itertools import cycle

data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"
bs_params = { 'mu_brg': 0.01, 'mu_seal': 1.4e-3, 'rho_seal': 850 }

n_ele, n_node, n_dof, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb, brgs, seals, support_dofs = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

# from import_data import plot_rotor_2d
# plot_rotor_2d(rotor_elements, added_elements=added_elements, brgs=brgs, seals=seals, lw=1.6, save_img=True)

#%%
d = np.load('checkpoints/latest.npz')
X_pop, F_pop = d['pop_X'], d['pop_F']
X_pareto, F_pareto = d['opt_X'], d['opt_F']

def plot_PCP(F, names, idx: Optional[np.ndarray] = None):
    import matplotlib.pyplot as plt
    # plt.rcParams.update({
    #     # "figure.figsize": (6, 4),   # 단일 column에 맞춤
    #     "font.size": 11,                 # 본문 글자 크기와 유사
    #     "axes.labelsize": 10,            # 축 라벨은 약간 크게
    #     "xtick.labelsize": 10,
    #     "ytick.labelsize": 10,
    #     "legend.fontsize": 10,
    #     "lines.linewidth": 1.0
    # })
    # fig, ax = plt.subplots()
    plot = PCP(
        # title=("Pareto Front (Objectives)", {'pad': 30}),
        labels=names,
        figsize=(6, 4)
        )
    plot.set_axis_style(color="grey", alpha=0.5)
    plot.add(F, color="grey", alpha=0.3)

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
    
    first_colors = ["red", "blue", "green"]
    default_cycle = plt.rcParams.get('axes.prop_cycle')
    cycle_colors = (default_cycle.by_key().get('color',
                    ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])) if default_cycle else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i, idc in enumerate(indices):
        if i < 3:
            color = first_colors[i]
        else:
            color = cycle_colors[(i - 3) % len(cycle_colors)]
        plot.add(np.atleast_2d(F[idc]), linewidth=5, color=color, linestyle='-')
    plot.save("pareto_PCP.png", dpi=300, bbox_inches="tight")

    plot.show()
    return plot

def plot_RAD(F, names, idx: Optional[np.ndarray] = None):
    plot = Radviz(
        # title="Optimization",
        # legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
        labels=names,
        endpoint_style={"s": 70, "color": "green"},
        figsize=(6, 4)
        )
    plot.set_axis_style(color="black", alpha=1.0)
    plot.add(F, color="grey", s=20)
    
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

    first_colors = ["red", "blue", "green"]
    default_cycle = plt.rcParams.get('axes.prop_cycle')
    cycle_colors = (default_cycle.by_key().get('color',
                    ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])) if default_cycle else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i, idc in enumerate(indices):
        if i < 3:
            color = first_colors[i]
        else:
            color = cycle_colors[(i - 3) % len(cycle_colors)]
        plot.add(np.atleast_2d(F[idc]), linewidth=3, color=color)
    plot.save("pareto_RAD.png", dpi=300, bbox_inches="tight")
    plot.show()
    return plot

def plot_pairwise(F_pop, F_par, names):
    import matplotlib as mpl
    plt.rc('axes', prop_cycle=mpl.rcParamsDefault['axes.prop_cycle'])
    m = F_pop.shape[1]
    fig, axes = plt.subplots(m, m, figsize=(2.6*m, 2.6*m))
    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            if i == j:
                ax.hist(F_pop[:, i], bins=30, alpha=0.5, color='gray')
                ax.hist(F_par[:, i], bins=30, alpha=0.8, color='tab:blue')
            else:
                ax.scatter(F_pop[:, j], F_pop[:, i], s=6, alpha=0.25, color='gray')
                ax.scatter(F_par[:, j], F_par[:, i], s=10, alpha=0.8, color='tab:blue')
            if i == m-1: ax.set_xlabel(names[j])
            if j == 0:    ax.set_ylabel(names[i])
    fig.tight_layout(); 
    plt.show()


def plot_3d(F, idx=(0,1,2), c_idx=3, names=None):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    c = F[:, c_idx]
    sc = ax.scatter(F[:, idx[0]], F[:, idx[1]], F[:, idx[2]],
                    c=c, cmap='viridis', s=12, alpha=0.8)
    if names:
        ax.set_xlabel(names[idx[0]]); ax.set_ylabel(names[idx[1]]); ax.set_zlabel(names[idx[2]])
    cb = fig.colorbar(sc, pad=0.1); cb.set_label(names[c_idx] if names else 'color')
    plt.tight_layout(); plt.show()

from glob import glob
def summarize_generations(dir='checkpoints'):
    import matplotlib as mpl
    plt.rc('axes', prop_cycle=mpl.rcParamsDefault['axes.prop_cycle'])
    files = sorted(glob(f'{dir}/gen_*.npz'))
    nds_cnt, best_vals = [], []
    for f in files:
        d = np.load(f)
        F = d['opt_F'] if 'opt_F' in d else None
        if F is not None and len(F):
            nds_cnt.append(F.shape[0])
            best_vals.append(F.min(axis=0))
    nds_cnt = np.array(nds_cnt)
    best_vals = np.vstack(best_vals) if best_vals else None
    if best_vals is not None:
        plt.figure(figsize=(7,4))
        for i,name in enumerate(obj_names):
            plt.plot(best_vals[:, i], label=name)
        plt.title('Best per objective'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return nds_cnt, best_vals

def plot_bearing_id_hist(X_pop, X_par, n_brg):
    import matplotlib as mpl
    plt.rc('axes', prop_cycle=mpl.rcParamsDefault['axes.prop_cycle'])
    ids_pop = X_pop[:, :2*n_brg].reshape(-1, n_brg, 2)[:,:,0].ravel()
    ids_par = X_par[:, :2*n_brg].reshape(-1, n_brg, 2)[:,:,0].ravel()
    fig, ax = plt.subplots(figsize=(7,3))
    ax.hist(ids_pop, bins=np.arange(1,55)-0.5, alpha=0.4, label='pop')
    ax.hist(ids_par, bins=np.arange(1,55)-0.5, alpha=0.8, label='pareto')
    plt.xlabel('Bearing ID'); plt.ylabel('Count'); plt.legend(); plt.tight_layout(); 
    fig.savefig("bearing_hist.png", dpi=300, bbox_inches="tight")
    plt.show()

#%%
sorted_idx = np.argsort(F_pareto[:,0])
idx = sorted_idx[[0, 1, -1]]
# obj_signs = np.array([1, 1, 1, -1, 1, 1])
# F_pareto = F_pareto * obj_signs[None, None, None, :, None, None]
# F_pop = F_pop * obj_signs[None, None, None, :, None, None]

obj_names = [r'$\dot{\text{m}}_{total}$', r'$\text{PL}_{brg}$', r'$\text{AF}_{\max}$', r'$\delta{}_{\min}$', r'$\text{Ampl.}_{brg}$', r'$\text{Ampl.}_{seal}$']

# FF = []
# for idx in np.arange(5,405,5,dtype=int):
#     d = np.load(f'checkpoints/gen_{idx:04d}.npz')
#     F_ = d.get('opt_F')
#     if F_ is None or F_.size == 0 or F_.shape[1] != 6: 
#         continue
#     FF.append(F_)
# F_all = np.vstack(FF)
plot_PCP(F=F_pareto, names=obj_names, idx=idx)
#%%
# plot_RAD(F=F_pareto, names=obj_names, idx=idx)
#%%

# plot_pairwise(F_pop, F_pareto, obj_names)
# plot_3d(F_pareto, idx=(0,1,2), c_idx=3, names=obj_names)
# nds_cnt, best_vals= summarize_generations()
plot_bearing_id_hist(X_pop, X_pareto, n_brg)

# print(X_pareto[sorted_idx[0],4:-1])
# print(X_pareto[sorted_idx[-1],4:-1])

# #%%
from sklearn.preprocessing import MinMaxScaler
F_scaled = F_pareto
for of in range(6):
    scaler = MinMaxScaler()
    scaler.fit(F_scaled[:,of].reshape(-1,1))
    F_scaled[:,of] = scaler.transform(F_scaled[:,of].reshape(-1,1)).squeeze()
plot_RAD(F=F_scaled, names=obj_names, idx=idx)

#%%
# Visualize unbalanced response and logarithmic decrement for selected Pareto Xs
from collections import defaultdict
from loader_brg_seal import BearingNondModel, SealDONModel
from solver_rotordyn import assemble_system_matrix
from solver_rotordyn import eig_batch as eig
from solver_rotordyn import unbalance_response_batch_cpu_parallel as unbalanced_response

# Setup analysis grid (keep small for speed)
w_range = np.array([500, 8000]) * np.pi / 30
n_w = 7
w_vec = np.linspace(w_range[0], w_range[1], n_w)

# Models
model_brg = BearingNondModel()
model_seal = SealDONModel()

rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
C_struct = np.zeros_like(mat_K_r)

# Parameter scalings and groupings
f_brg_dim = np.array([[1, 1e-4],[1, 1e-4]])
f_seal_dim = [1e-6, 1e-6, 1e-1]
rdc_signs = np.array([1, 1, -1, 1])

groups = defaultdict(list)
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
idx_seal = [np.array(groups[t+1], dtype=int) for t in range(3)]

# Select a few representative Pareto points (fast)
# sel = sorted_idx[[0, -1]] if len(sorted_idx) >= 4 else np.arange(min(4, len(sorted_idx)))
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']

amp_curves = []   # max amplitude over supports vs speed
logd_curves = []  # min logarithmic decrement (first few forward modes) vs speed
labels = []

#%%

def plot_harmonic(w, amp, nodes='key', fontsize=None, smooth=None):
    from scipy.interpolate import CubicSpline
    """
    Plot unbalanced response for selected nodes.

    Parameters
    - w:            ndarray [n_w] in rad/s
    - amp:          ndarray [n_w, n_node] or [pop, n_w, n_node]
    - nodes:        selection of nodes to plot
                    - 'key' (default): bearings + seals + unbalance nodes
                    - 'all': all nodes
                    - 'brg', 'seal', 'unb': specific node categories
                    - int or list/array of ints: specific node indices (0-based)
    - fontsize:     label font size
    - smooth:       optional smoothing for curves along speed axis
                    - None or 0/1: no smoothing
                    - 'spline'/'cubic' or ('spline', n_pts): cubic spline
    """
    plt.rcParams.update({
        "figure.figsize": (6, 4),   # 단일 column에 맞춤
        "font.size": 8,                 # 본문 글자 크기와 유사
        "axes.labelsize": 9,            # 축 라벨은 약간 크게
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0
    })
        
    fig, ax = plt.subplots()

    # Normalize amp shape to [n_w, n_node]
    A = np.array(amp)
    if A.ndim == 3:
        A = A[0]  # use first population by default
    elif A.ndim != 2:
        raise ValueError("amp must be [n_w, n_node] or [pop, n_w, n_node]")

    n_w, n_node = A.shape

    # Build category node sets from globals if available
    bearing_nodes = np.array([b.node for b in brgs], dtype=int) if 'brgs' in globals() and len(brgs) else np.array([], dtype=int)
    seal_nodes_g  = np.array([s.node for s in seals], dtype=int) if 'seals' in globals() and len(seals) else np.array([], dtype=int)
    if 'unb' in globals() and hasattr(unb, 'cases') and len(unb.cases):
        unb_nodes = np.unique(np.array([n for c in unb.cases for n in c.node], dtype=int))
    else:
        unb_nodes = np.array([], dtype=int)

    # Determine which nodes to plot
    sel: np.ndarray
    if isinstance(nodes, str):
        key = nodes.lower()
        if key == 'key' or nodes is None:
            sel = np.unique(np.concatenate([bearing_nodes, seal_nodes_g, unb_nodes]))
        elif key in ('brg', 'bearing'):
            sel = bearing_nodes
        elif key in ('seal', 'seals'):
            sel = seal_nodes_g
        elif key in ('unb', 'unbalance', 'unbalanced'):
            sel = unb_nodes
        elif key == 'all':
            sel = np.arange(n_node)
        else:
            # Unknown key string -> empty selection
            sel = np.array([], dtype=int)
    else:
        sel = np.atleast_1d(nodes).astype(int)
        sel = sel[(sel >= 0) & (sel < n_node)]

    # Assign category for each selected node (priority: bearing > seal > unbalance > other)
    set_brg = set(bearing_nodes.tolist())
    set_seal = set(seal_nodes_g.tolist())
    set_unb = set(unb_nodes.tolist())

    def category(n):
        if n in set_brg:
            return 'bearing'
        if n in set_seal:
            return 'seal'
        if n in set_unb:
            return 'unbalance'
        return 'other'

    # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                            # cycler('linestyle', ['-', '--', ':', '-.'])))
    sty_cycle = cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) * cycler('linestyle', ['-', '--', ':', '-.'])

    # Optional smoothing
    A_plot = A
    spline_mode = False
    w_use = w
    n_pts = max(200, n_w * 5) if 'n_w' in globals() else max(200, A.shape[0] * 5)
    if isinstance(smooth, str) and smooth.lower() in ('spline', 'cubic', 'cubic-spline'):
        spline_mode = True
        w_use = np.linspace(w.min(), w.max(), int(n_pts))
    elif isinstance(smooth, (tuple, list)) and len(smooth) >= 2 and str(smooth[0]).lower() in ('spline', 'cubic', 'cubic-spline'):
        spline_mode = True
        try:
            n_pts = max(10, int(smooth[1]))
        except Exception:
            n_pts = max(200, A.shape[0] * 5)
        w_use = np.linspace(w.min(), w.max(), int(n_pts))

    w_rpm = w_use * 30.0 / np.pi


    # Plot each chosen node with category-specific linestyle and category-only legend
    seen_labels = set()
    
    for n, sty in zip(sel, cycle(sty_cycle)):
        cat = category(n)
        # Only label once per category to keep legend concise
        label = f"{cat.title()} {n}"
        lbl = None if cat in seen_labels else label
        if lbl is not None:
            seen_labels.add(label)
        if spline_mode:
            cs = CubicSpline(w, A[:, n], bc_type='natural')
            y_plot = cs(w_use)
        else:
            y_plot = A_plot[:, n]
        ax.plot(
            w_rpm,
            y_plot * 1e6,
            label=lbl,
            lw=1.8,
            **sty,
        )

    xlabel = "Rotor speed (rpm)"
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ylabel = "Unbalanced response " + r'$(\mu{}\text{m})$'
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlim([0, 7000])
    if len(seen_labels):
        ax.legend()
    fig.tight_layout()
    fig.savefig("unb_resp.png", dpi=300, bbox_inches="tight")
    # plt.show()

def plot_logdec_lowest(logdec_arr, eigvals_arr, w, n=4, pop_idx=0, fontsize=None):
    """
    Plot log decrement for the n modes with the lowest natural frequency
    (at the first speed), in ascending order.

    Parameters
    - logdec_arr: ndarray [pop, n_w, m] or [n_w, m]
    - eigvals_arr: ndarray [pop, n_w, k] (complex); uses first m columns
    - w: ndarray [n_w] in rad/s
    - n: number of lowest-frequency modes to plot
    - pop_idx: which population index to use (if pop dimension present)
    - fontsize: label font size
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    plt.rc('axes', prop_cycle=mpl.rcParamsDefault['axes.prop_cycle'])
    L = np.array(logdec_arr)
    EV = np.array(eigvals_arr)
    
    plt.rcParams.update({
        "figure.figsize": (6, 4),   # 단일 column에 맞춤
        "font.size": 8,                 # 본문 글자 크기와 유사
        "axes.labelsize": 9,            # 축 라벨은 약간 크게
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0
    })
        
    fig, ax = plt.subplots()

    # Normalize shapes
    if L.ndim == 2:
        # assume [n_w, m]
        Lp = L
        EVp = EV[pop_idx]
    elif L.ndim == 3:
        # [pop, n_w, m]
        Lp = L[pop_idx]
        EVp = EV[pop_idx]
    else:
        raise ValueError("logdec_arr must be [n_w, m] or [pop, n_w, m]")

    n_w, m = Lp.shape
    # Use the corresponding first m modes of eigvals for wn
    EVm = EVp[:, :m]
    # Natural frequencies at first speed index
    alpha0 = np.real(EVm[0])
    beta0 = np.imag(EVm[0])
    wn0 = np.sqrt(np.maximum(alpha0*alpha0 + beta0*beta0, 0.0))
    # Sort indices by ascending natural frequency
    order = np.argsort(wn0)
    sel = order[:max(1, int(n))]

    x_rpm = w * 30.0 / np.pi
    for i, idx in enumerate(sel):
        ax.plot(x_rpm, Lp[:, idx], label=f"Mode {i+1}", lw=1.8)

    ax.axhline(0.1, color='red', lw=1.5, zorder=1, linestyle='-')
    ax.set_xlabel("Rotor speed (rpm)", fontsize=fontsize)
    ax.set_ylabel(r'$\delta{}$', fontsize=fontsize)
    ax.set_ylim([-0.1, 2.0])
    ax.set_xlim([0, 7000])
    ax.legend()
    fig.tight_layout()
    fig.savefig("log_dec.png", dpi=300, bbox_inches="tight")
#%%

for i, pi in enumerate(idx):
    X = X_pareto[pi]
    X = X.reshape(1, -1)  # [1, n_var]

    # Split and scale params
    X_brg = X[:, :2*n_brg].reshape(1, n_brg, 2)
    X_seal = X[:, 2*n_brg:].reshape(1, n_seal, 3)

    x_brg = X_brg * f_brg_dim
    K_brg, C_brg, _ = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

    seal_rdc = np.zeros((1, n_seal, 4, n_w), dtype=float)
    for t in range(3):
        idx_s = idx_seal[t]
        if idx_s.size == 0:
            continue
        params_t = X_seal[:, idx_s]
        x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
        rdc_flat = model_seal.predict(t+1, x_seal, w_vec).reshape(1, len(idx_s), 4, n_w)
        seal_rdc[:, idx_s] = rdc_flat

    K_seal = seal_rdc[:,:,[2, 3, 3, 2],:] * rdc_signs[None, None, :, None]
    C_seal = seal_rdc[:,:,[0, 1, 1, 0],:] * rdc_signs[None, None, :, None]

    K_vals = np.concatenate([K_brg, K_seal], axis=1)
    C_vals = np.concatenate([C_brg, C_seal], axis=1)

    K_all, Ceff_all = assemble_system_matrix(
        mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals
    )

    # Unbalanced response
    harmonic = unbalanced_response(
        M=mat_M, unb=unb, K_all=K_all, Ceff_all=Ceff_all, w_vec=w_vec
    )  # [1, n_w, n_dof, 1]
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2
    Ux = harmonic[0, :, idx_x, 0]
    Uy = harmonic[0, :, idx_y, 0]
    amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)  # [n_w, n_node]
    brg_nodes = np.array([b.node for b in brgs], dtype=int)
    seal_nodes = np.array([s.node for s in seals], dtype=int)
    cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))
    # Guard against any out-of-range indices using actual amplitude shape
    valid = (cal_nodes >= 0) & (cal_nodes < amp.shape[1])
    cal = cal_nodes[valid]
    amp_curves.append(amp[:, cal].max(axis=1) if cal.size else amp.max(axis=1))

    # Logarithmic decrement from forward modes
    eigvals, _ = eig(M=mat_M, K_all=K_all, Ceff_all=Ceff_all, track=True)
    alpha = np.real(eigvals[0])  # [n_w, k]
    beta  = np.imag(eigvals[0])
    k_use = min(4, alpha.shape[1])
    if k_use == 0:
        logd_curves.append(np.zeros(n_w))
    else:
        a = alpha[:, :k_use]
        b = beta[:, :k_use]
        logdec = -2.0 * np.pi * a / np.sqrt(np.maximum(a*a + b*b, 1e-30))  # [n_w, k_use]
        logd_curves.append(np.min(logdec, axis=1))
    labels.append(f"idx {int(pi)}")
    
    plot_harmonic(w=w_vec, amp=amp.transpose(), smooth='spline', nodes='seal')
    plot_logdec_lowest(logdec_arr=logdec, eigvals_arr=eigvals, w=w_vec, n=4, pop_idx=0)
    
    # plt.figure()
    # plt.plot(w_vec*30/np.pi,amp.transpose()*1e6)
    # plt.show()
    
    # plt.figure()
    # plt.plot(w_vec,logdec)
    # plt.ylim([0, 3])
    # plt.show()

# # Plot unbalanced response (max over supports)
# plt.figure(figsize=(7, 4))
# for c, y, lb in zip(colors, amp_curves, labels):
#     plt.plot(w_vec*30/np.pi, y, label=lb, color=c)
# plt.xlabel('Speed (rpm)'); plt.ylabel('Max |U| at supports (m)')
# plt.title('Unbalanced Response (selected Pareto)')
# plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# # Plot logarithmic decrement (min over first modes)
# plt.figure(figsize=(7, 4))
# for c, y, lb in zip(colors, logd_curves, labels):
#     plt.plot(w_vec*30/np.pi, y, label=lb, color=c)
# plt.xlabel('Speed (rpm)'); plt.ylabel('Min logarithmic decrement')
# plt.title('Logarithmic Decrement (selected Pareto)')
# plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()


# %%
