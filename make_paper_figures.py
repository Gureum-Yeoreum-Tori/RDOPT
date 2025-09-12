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

# Optional: reuse existing pareto plotting helpers if available
# try:
    # from visualize_pareto import plot_PCP as pareto_plot_PCP
    # from visualize_pareto import plot_RAD as pareto_plot_RAD
# from visualize_pareto import plot_bearing_id_hist
# except Exception:
    # pareto_plot_PCP = None
    # pareto_plot_RAD = None
    # plot_bearing_id_hist = None

try:
    from solver_seal import main_seal_solver
except Exception:
    main_seal_solver = None

cm = 1/2.54  # centimeters in inches

figsize_F = (6.8, 4.2) # single column
figsize_SC = (6, 3.7) # single column
figsize_SC_two_third = (4, 3.7) # single column
figsize_SC_one_third = (2.1, 3.7) # single column
figsize_SC_two_third_s = (4.2, 2.4) # single column, short
figsize_SC_one_third_s = (2.4, 2.4) # single column, short
figsize_SC_s = (6, 2.8) # single column, short
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

    model_brg = BearingNondModel()
    model_seal = SealDONModel()
    model_seal_leak = SealLeakModel()

    return {
        'n_ele': n_ele,
        'n_node': n_node,
        'n_dof': n_dof,
        'n_brg': n_brg,
        'n_seal': n_seal,
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
    }


def calc_KC_for_design(X, ctx, w_vec):
    n_brg = ctx['n_brg']
    n_seal = ctx['n_seal']
    brgs = ctx['brgs']
    seals = ctx['seals']
    model_brg = ctx['model_brg']
    model_seal = ctx['model_seal']

    # Parameter scalings
    f_brg_dim = np.array([[1, 1e-4], [1, 1e-4]])
    f_seal_dim = [1e-6, 1e-6, 1e-1]
    rdc_signs = np.array([1, 1, -1, 1])

    pop = X.shape[0]
    X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
    X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

    # Bearings
    x_brg = X_brg * f_brg_dim
    K_brg, C_brg, _loss_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

    # Group seals by type
    n_type_seal = 3
    groups = {}
    for i, s in enumerate(seals):
        groups.setdefault(s.SealNet, []).append(i)
    idx_seal = [np.array(groups.get(t+1, []), dtype=int) for t in range(n_type_seal)]

    # Seals
    seal_rdc = np.zeros((pop, n_seal, 4, w_vec.shape[0]), dtype=float)
    for t in range(n_type_seal):
        idx = idx_seal[t]
        if idx.size == 0:
            continue
        params_t = X_seal[:, idx]
        x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
        rdc_flat = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, w_vec.shape[0])
        seal_rdc[:, idx] = rdc_flat

    K_seal = seal_rdc[:, :, [2, 3, 3, 2], :] * rdc_signs[None, None, :, None]
    C_seal = seal_rdc[:, :, [0, 1, 1, 0], :] * rdc_signs[None, None, :, None]

    K_vals = np.concatenate([K_brg, K_seal], axis=1)
    C_vals = np.concatenate([C_brg, C_seal], axis=1)

    return K_vals, C_vals


def analyze_design(X, ctx, w_vec):
    mat_M = ctx['mat_M']
    mat_K_r = ctx['mat_K_r']
    mat_C_g = ctx['mat_C_g']
    support_dofs = ctx['support_dofs']
    unb = ctx['unb']

    rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
    C_struct = np.zeros_like(mat_K_r)

    K_vals, C_vals = calc_KC_for_design(X, ctx, w_vec)
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

    return eigvals, amp


def plot_campbell(eigvals, w_vec, out_path, title=None, ylim_rpm=(0, 7000)):
    _default_rcparams()
    E = np.array(eigvals)[0]  # [n_w, k]
    beta = np.abs(E.imag) * 30.0 / np.pi / 60 # in Hz
    rpm = w_vec * 30.0 / np.pi

    # pick lowest n modes by average frequency
    n_pick = min(8, beta.shape[1]) if beta.ndim == 2 else 0
    if n_pick == 0:
        return
    idx = np.argsort(beta.mean(axis=0))[:n_pick]

    fig, ax = plt.subplots()
    # excitation (1x) line in Hz
    exc = rpm / 60.0

    for k in idx:
        line, = ax.plot(rpm, beta[:, k], '-')

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

    # plot 1x line last so it stays visible but under markers
    ax.plot(rpm, exc, 'k-', lw=1.0)
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel('Frequency (Hz)')
    if title:
        ax.set_title(title)
    ax.set_ylim([0, ylim_rpm[1]/60])
    ax.set_xlim([0, rpm.max()])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1400))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=600, bbox_inches='tight')
    fig.savefig(out_path, dpi=600)


def compute_logdec(eigvals):
    EV = np.array(eigvals)[0]  # [n_w, k]
    alpha = EV.real
    beta = EV.imag
    return -2 * np.pi * alpha / np.sqrt(np.maximum(alpha*alpha + beta*beta, 1e-30))


def plot_logdec_lowest(logdec_arr, eigvals, w_vec, out_path, n=4, ylim=(0.0, 2.0)):
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
    fig, ax = plt.subplots()
    for i, k in enumerate(sel):
        ax.plot(x_rpm, L[:, k], label=f"Mode {i+1}")
    ax.axhline(0.1, color='r', lw=1.2, linestyle='--')
    ax.set_xlabel('Rotational speed (RPM)')
    ax.set_ylabel(r'$\delta$')
    # ax.set_ylim(ylim)
    ax.set_xlim([0, x_rpm.max()])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1400))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # fig.savefig(out_path, dpi=600, bbox_inches='tight')
    fig.savefig(out_path, dpi=600)


def plot_unbalance_response(amp, w_vec, ctx, out_path, nodes='key', smooth_spline=True, figsize=figsize_DC, show_lgd=False):
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

    # optionally spline along speed
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
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 1400))
    ax.grid(True, alpha=0.3)
    if show_lgd:
        ax.legend()
        # long legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                borderaxespad=0., ncol=2,
                handlelength=1.2)

    fig.tight_layout()
    
    fig.savefig(out_path, dpi=600, bbox_inches='tight')


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

        plot.add(np.atleast_2d(F[idc]), linewidth=3, linestyle='-')
    # plot.save(out_path, dpi=600, bbox_inches="tight")
    plot.save(out_path, dpi=600, bbox_inches="tight")

    plot.show()
    return plot

def pareto_plot_RAD(F, names, idx: Optional[np.ndarray] = None, figsize=figsize_DC,scale=True):
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
        endpoint_style={"s": 40, "color": "green"},
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

    first_colors = ["red", "blue", "green"]
    default_cycle = plt.rcParams.get('axes.prop_cycle')
    cycle_colors = (default_cycle.by_key().get('color',
                    ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])) if default_cycle else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i, idc in enumerate(indices):
        # if i < 3:
        #     color = first_colors[i]
        # else:
        #     color = cycle_colors[(i - 3) % len(cycle_colors)]
        color = cycle_colors[(i - 3) % len(cycle_colors)]
        plot.add(np.atleast_2d(F[idc]), linewidth=2)
    plot.save("pareto_RAD.png", dpi=600, bbox_inches="tight")
    plot.show()
    return plot

def save_optimization_plots(checkpoint_path, n_brg):
    if not os.path.exists(checkpoint_path):
        print(f"[warn] checkpoint not found: {checkpoint_path}")
        return
    d = np.load(checkpoint_path)
    X_pop, F_pop = d['pop_X'], d['pop_F']
    X_pareto, F_pareto = d['opt_X'], d['opt_F']

    # obj_names = [r'$\dot{\mathrm{m}}_{total}$', r'$\mathrm{PL}_{brg}$', r'$\mathrm{AF}_{\max}$', r'$-\delta_{\min}$', r'$\mathrm{Ampl.}_{brg}$', r'$\mathrm{Ampl.}_{seal}$']
    obj_names = [r'$\dot{\mathrm{m}}_{total}$', r'$\mathrm{PL}_{brg}$', r'$\mathrm{AF}_{\max}$', r'$\delta_{\min}$', r'$\mathrm{Ampl.}_{brg}$', r'$\mathrm{Ampl.}_{seal}$']
    pareto_signs = np.array([1, 1, 1, -1, 1, 1])
    F_pareto_corrected = F_pareto * pareto_signs
    # Select a few reference pareto points by first objective
    sel = np.argsort(F_pareto[:, 0])
    # sel = sel[[0, 1, min(len(sel)-1, len(sel)-1)]] if len(sel) >= 3 else sel
    sel = sel[[0, -1]]

    pareto_plot_PCP(F=F_pareto_corrected, names=obj_names, idx=sel,
                    figsize=figsize_SC_two_third_s)
    
    # if pareto_plot_PCP is not None:
    #     # pareto_plot_PCP(F=F_pareto_corrected[:,:2], names=obj_names[:2],
    #     #                 idx=sel,
    #     #                 out_path="pareto_2factor")
    # else:
    #     print("[warn] pareto PCP helper not available; skipping PCP figure.")
        

    # pareto_plot_RAD(F=F_pareto_corrected, 
    #                 names=obj_names, 
    #                 idx=sel,
    #                 figsize=figsize_DC_b)
    # if pareto_plot_RAD is not None:
    #     pareto_plot_RAD(F=F_pareto_corrected, names=obj_names, idx=sel)
    # else:
    #     print("[warn] pareto Radviz helper not available; skipping Radviz figure.")

    # plot_bearing_id_hist(X_pop, X_pareto, n_brg)
    # if plot_bearing_id_hist is not None:
        # plot_bearing_id_hist(X_pop, X_pareto, n_brg)
    # else:
    #     print("[warn] bearing histogram helper not available; skipping bearing history figure.")


# def save_nn_validation(w_vec, out_dir='.'):
#     if main_seal_solver is None:
#         print('[warn] solver_seal.main_seal_solver not available; skipping NN validation figures.')
#         return
#     model_seal = SealDONModel()
#     model_leak = SealLeakModel()

#     # Example tapered seal geometry (similar to visualize_optimal.py)
#     # Units already in SI
#     Ds = 0.23
#     Ls = 0.15
#     NxSeal = 45
#     mu = 1.4e-3
#     rho = 850
#     dp = 16_000_000
#     # geometry params: [h_in (um), h_out (um), psr*10]
#     x_seal_um = np.array([100, 188, -5.0])
#     x_seal = np.array([x_seal_um[0]*1e-6, x_seal_um[1]*1e-6, x_seal_um[2]*1e-1])[None, :]

#     # Predict by networks (use model_id=3 by default; adjust if needed)
#     rdc_pred = model_seal.predict(3, x_seal, w_vec)[0]  # [4, n_w] → C,c,K,k
#     leak_pred = model_leak.predict(3, x_seal)[0, 0]

#     geometry = {'hIn': x_seal[0, 0], 'hOut': x_seal[0, 1], 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
#     fluid = {'mu': mu, 'rho': rho}
#     op_conditions = {'dp': dp, 'psr': x_seal[0, 2], 'w_vec': w_vec}
#     Leak, RDC, *_ = main_seal_solver(geometry, fluid, op_conditions)
#     rdc_true = RDC[:, 2:]  # C,c,K,k

#     rpm = w_vec * 30.0 / np.pi

#     # 1) Leakage: solver vs. network (network gives single value → horizontal)
#     _default_rcparams()
#     fig, ax = plt.subplots()
#     ax.plot(rpm, Leak, 'k-', lw=1.6, label='Solver')
#     ax.hlines(leak_pred, xmin=rpm.min(), xmax=rpm.max(), colors='tab:blue', linestyles='--', lw=1.6, label='Network')
#     ax.set_xlabel('Speed (RPM)')
#     ax.set_ylabel('Leakage flow rate [kg/s]')
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, 'nn_leak_validation.png'), dpi=600, bbox_inches='tight')

#     # 2) RDC curves: each channel C,c,K,k vs speed
#     labels = ['C', 'c', 'K', 'k']
#     _default_rcparams()
#     fig, axs = plt.subplots(2, 2, figsize=(7, 5))
#     axs = axs.ravel()
#     for i in range(4):
#         axs[i].plot(rpm, rdc_true[:, i], 'k-', lw=1.6, label='Solver')
#         axs[i].plot(rpm, rdc_pred[i, :], 'tab:blue', lw=1.6, linestyle='--', label='Network')
#         axs[i].set_xlabel('Speed (RPM)')
#         axs[i].set_ylabel(labels[i])
#         axs[i].grid(True, alpha=0.3)
#         if i == 0:
#             axs[i].legend(loc='best')
#     fig.tight_layout()
#     fig.savefig(os.path.join(out_dir, 'nn_rdc_validation.png'), dpi=600, bbox_inches='tight')


# def run():
#     # --- global settings ---
#     w_range = np.array([500, 7000]) * np.pi / 30
#     n_w = 14
#     w_vec = np.linspace(w_range[0], w_range[1], n_w)
#     w_oper = 3500 * np.pi / 30

#     bs_params = {
#         'mu_brg': 0.01,
#         'mu_seal': 1.4e-3,
#         'rho_seal': 850,
#     }

#     # --- load rotor + models ---
#     ctx = build_rotor_and_models(bs_params)

#     # --- Initial/reference design (programmatic, matches n_brg/n_seal) ---
#     def build_initial_X(n_brg, n_seal):
#         # Bearings: [id, clearance_ratio]
#         X_brg = np.tile([2, 15], (n_brg, 1))
#         # Seals: [h_in(um), h_out(um), psr*10]
#         # X_seal = np.tile([300, 300, 0], (n_seal, 1))
        
#         X_seal = np.concatenate([np.tile([285, 285, 6.5], 2),
#                                 np.tile([300, 300, 0, 300, 300, 7.5], 6),
#                                 [285, 285, 6.5]]
#                             )

#         return np.concatenate([X_brg.ravel(), X_seal.ravel()], axis=0)[None, :]

#     X_init = build_initial_X(ctx['n_brg'], ctx['n_seal']).astype(float)

#     eig_init, amp_init = analyze_design(X_init, ctx, w_vec)
#     L_init = compute_logdec(eig_init)

#     plot_campbell(eig_init, w_vec, out_path='initial_campbell.png')
#     plot_logdec_lowest(L_init, eig_init, w_vec, out_path='initial_logdec.png', n=4)
#     plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance.png', nodes='key')

#     # --- Optimization results (Pareto visuals) ---
#     save_optimization_plots('checkpoints/latest.npz', ctx['n_brg'])

#     # --- Optimized design: pick one pareto solution ---
#     try:
#         d = np.load('checkpoints/latest.npz')
#         X_pareto, F_pareto = d['opt_X'], d['opt_F']
#         # choose by minimum total leakage as an example
#         idx = int(np.argmin(F_pareto[:, 0])) if len(F_pareto) else 0
#         X_opt = X_pareto[idx:idx+1, :]
#     except Exception:
#         # fallback to initial if checkpoints not present
#         X_opt = X_init.copy()

#     eig_opt, amp_opt = analyze_design(X_opt, ctx, w_vec)
#     L_opt = compute_logdec(eig_opt)

#     plot_campbell(eig_opt, w_vec, out_path='optimized_campbell.png')
#     plot_logdec_lowest(L_opt, eig_opt, w_vec, out_path='optimized_logdec.png', n=4)
#     plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance.png', nodes='key')

#     # --- Neural network model validation (leakage + RDC)
#     save_nn_validation(w_vec, out_dir='.')

#     print('Saved figures:')
#     for f in [
#         'initial_campbell.png', 'initial_logdec.png', 'initial_unbalance.png',
#         'pareto_PCP.png', 'pareto_RAD.png', 'bearing_hist.png',
#         'optimized_campbell.png', 'optimized_logdec.png', 'optimized_unbalance.png',
#         'nn_leak_validation.png', 'nn_rdc_validation.png']:
#         if os.path.exists(f):
#             print(' -', f)

#%%

# --- global settings ---
w_range = np.array([500, 7000]) * np.pi / 30
n_w = 11
w_vec = np.linspace(w_range[0], w_range[1], n_w)
w_oper = 3500 * np.pi / 30

bs_params = {
    'mu_brg': 0.025,
    'mu_seal': 1.4e-3,
    'rho_seal': 850,
}

# --- load rotor + models ---
ctx = build_rotor_and_models(bs_params)

# --- Initial/reference design (programmatic, matches n_brg/n_seal) ---
def build_initial_X(n_brg, n_seal):
    # Bearings: [id, clearance_ratio]
    X_brg = np.tile([2, 15], (n_brg, 1))
    # Seals: [h_in(um), h_out(um), psr*10]
    # X_seal = np.tile([300, 300, 0], (n_seal, 1))
    
    X_seal = np.concatenate([np.tile([285, 285, 6.5], 2),
                            np.tile([300, 300, 0, 300, 300, 7.5], 6),
                            [285, 285, 6.5]]
                        )

    return np.concatenate([X_brg.ravel(), X_seal.ravel()], axis=0)[None, :]

X_init = build_initial_X(ctx['n_brg'], ctx['n_seal']).astype(float)

eig_init, amp_init = analyze_design(X_init, ctx, w_vec)
L_init = compute_logdec(eig_init)

# --- Optimized design: pick one pareto solution ---
try:
    d = np.load('checkpoints/latest.npz')
    X_pop, F_pop = d['pop_X'], d['pop_F']
    X_pareto, F_pareto = d['opt_X'], d['opt_F']
    # choose by minimum total leakage as an example
    idx = int(np.argmin(F_pareto[:, 0])) if len(F_pareto) else 0
    X_opt = X_pareto[idx:idx+1, :]
except Exception:
    # fallback to initial if checkpoints not present
    X_opt = X_init.copy()


#%%
# plot_campbell(eig_init, w_vec, out_path='initial_campbell.png')
# plot_logdec_lowest(L_init, eig_init, w_vec, out_path='initial_logdec.png', n=4)

# plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance_brg.png', nodes='brg')
# plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance_seal.png', nodes='seal', figsize=figsize_DC_tall)
# plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance.png', nodes='key', figsize=figsize_SC_s)
# plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance.png', nodes='key', figsize=(6, 2.35))







#%%
plt.figure(figsize=figsize_SC_one_third_s)
plt.scatter(F_pop[:, 0], F_pop[:, 1], s=10, alpha=0.5,facecolors='none', edgecolors='grey', label="Solutions")
plt.scatter(F_pareto[:, 0], F_pareto[:, 1], s=30, facecolors="#FB7AE1", edgecolors='#FB7AE1', label="Pareto")
plt.xlabel('Seal leakage (kg/s)')
plt.ylabel('Bearing power loss (W)')
plt.legend()
plt.savefig("pareto_2factor.png", dpi=600, bbox_inches="tight")
plt.show()



#%%
# --- Optimization results (Pareto visuals) ---
save_optimization_plots('checkpoints/latest.npz', ctx['n_brg'])







# #%%

# eig_opt, amp_opt = analyze_design(X_opt, ctx, w_vec)
# L_opt = compute_logdec(eig_opt)

# plot_campbell(eig_opt, w_vec, out_path='optimized_campbell_0.png')
# plot_logdec_lowest(L_opt, eig_opt, w_vec, out_path='optimized_logdec_0.png', n=4)
# # plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance_0.png', nodes='key',figsize=figsize_SC_s)
# plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance_0.png', nodes='key')


# plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance_brg.png', nodes='brg')
# plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance_seal.png', nodes='seal')


#%%

# --- Neural network model validation (leakage + RDC)
# save_nn_validation(w_vec, out_dir='.')


#%%
# print('Saved figures:')
# for f in [
#     'initial_campbell.png', 'initial_logdec.png', 'initial_unbalance.png',
#     'pareto_PCP.png', 'pareto_RAD.png', 'bearing_hist.png',
#     'optimized_campbell.png', 'optimized_logdec.png', 'optimized_unbalance.png',
#     'nn_leak_validation.png', 'nn_rdc_validation.png']:
#     if os.path.exists(f):
#         print(' -', f)

# %%
