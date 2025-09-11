import os
import numpy as np
import matplotlib.pyplot as plt

from import_data import rotor_import
from loader_brg_seal import BearingNondModel, SealDONModel, SealLeakModel
from solver_rotordyn import assemble_system_matrix, eig_batch, unbalance_response_batch_cpu_parallel
from scipy.signal import find_peaks

# Optional: reuse existing pareto plotting helpers if available
try:
    from visualize_pareto import plot_PCP as pareto_plot_PCP
    from visualize_pareto import plot_RAD as pareto_plot_RAD
    from visualize_pareto import plot_bearing_id_hist
except Exception:
    pareto_plot_PCP = None
    pareto_plot_RAD = None
    plot_bearing_id_hist = None

try:
    from solver_seal import main_seal_solver
except Exception:
    main_seal_solver = None


def _default_rcparams():
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2
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
    beta = np.abs(E.imag) * 30.0 / np.pi
    rpm = w_vec * 30.0 / np.pi

    # pick lowest n modes by average frequency
    n_pick = min(8, beta.shape[1]) if beta.ndim == 2 else 0
    if n_pick == 0:
        return
    idx = np.argsort(beta.mean(axis=0))[:n_pick]

    fig, ax = plt.subplots()
    for k in idx:
        ax.plot(rpm, beta[:, k], '-')
    ax.plot(rpm, rpm, 'k-', lw=1.0)
    ax.set_xlabel('Speed (RPM)')
    ax.set_ylabel('Modal frequency (RPM)')
    if title:
        ax.set_title(title)
    ax.set_ylim(ylim_rpm)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')


def compute_logdec(eigvals):
    EV = np.array(eigvals)[0]  # [n_w, k]
    alpha = EV.real
    beta = EV.imag
    return -2 * np.pi * alpha / np.sqrt(np.maximum(alpha*alpha + beta*beta, 1e-30))


def plot_logdec_lowest(logdec_arr, eigvals, w_vec, out_path, n=4, ylim=(0.0, 2.0)):
    _default_rcparams()
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
    ax.axhline(0.1, color='r', lw=1.2)
    ax.set_xlabel('Speed (RPM)')
    ax.set_ylabel(r'$\delta$')
    ax.set_ylim(ylim)
    ax.set_xlim([x_rpm.min(), x_rpm.max()])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')


def plot_unbalance_response(amp, w_vec, ctx, out_path, nodes='key', smooth_spline=True):
    _default_rcparams()
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

    # optionally spline along speed
    from scipy.interpolate import CubicSpline
    rpm = w_vec * 30.0 / np.pi
    fig, ax = plt.subplots()
    for n in sel:
        y = A[:, n]
        if smooth_spline and n_w >= 3:
            w_use = np.linspace(w_vec.min(), w_vec.max(), max(200, n_w*5))
            cs = CubicSpline(w_vec, y, bc_type='natural')
            y_plot = cs(w_use)
            ax.plot(w_use * 30.0 / np.pi, y_plot * 1e6, lw=1.6)
        else:
            ax.plot(rpm, y * 1e6, lw=1.6)
    ax.set_xlabel('Rotor speed (RPM)')
    ax.set_ylabel(r'Unbalanced response $(\mu m)$')
    ax.set_xlim([rpm.min(), rpm.max()])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')


def save_optimization_plots(checkpoint_path, n_brg):
    if not os.path.exists(checkpoint_path):
        print(f"[warn] checkpoint not found: {checkpoint_path}")
        return
    d = np.load(checkpoint_path)
    X_pop, F_pop = d['pop_X'], d['pop_F']
    X_pareto, F_pareto = d['opt_X'], d['opt_F']

    obj_names = [r'$\dot{\mathrm{m}}_{total}$', r'$\mathrm{PL}_{brg}$', r'$\mathrm{AF}_{\max}$', r'$-\delta_{\min}$', r'$\mathrm{Ampl.}_{brg}$', r'$\mathrm{Ampl.}_{seal}$']

    # Select a few reference pareto points by first objective
    sel = np.argsort(F_pareto[:, 0])
    sel = sel[[0, 1, min(len(sel)-1, len(sel)-1)]] if len(sel) >= 3 else sel

    if pareto_plot_PCP is not None:
        pareto_plot_PCP(F=F_pareto, names=obj_names, idx=sel)
    else:
        print("[warn] pareto PCP helper not available; skipping PCP figure.")

    if pareto_plot_RAD is not None:
        pareto_plot_RAD(F=F_pareto, names=obj_names, idx=sel)
    else:
        print("[warn] pareto Radviz helper not available; skipping Radviz figure.")

    if plot_bearing_id_hist is not None:
        plot_bearing_id_hist(X_pop, X_pareto, n_brg)
    else:
        print("[warn] bearing histogram helper not available; skipping bearing history figure.")


def save_nn_validation(w_vec, out_dir='.'):
    if main_seal_solver is None:
        print('[warn] solver_seal.main_seal_solver not available; skipping NN validation figures.')
        return
    model_seal = SealDONModel()
    model_leak = SealLeakModel()

    # Example tapered seal geometry (similar to visualize_optimal.py)
    # Units already in SI
    Ds = 0.23
    Ls = 0.15
    NxSeal = 45
    mu = 1.4e-3
    rho = 850
    dp = 16_000_000
    # geometry params: [h_in (um), h_out (um), psr*10]
    x_seal_um = np.array([100, 188, -5.0])
    x_seal = np.array([x_seal_um[0]*1e-6, x_seal_um[1]*1e-6, x_seal_um[2]*1e-1])[None, :]

    # Predict by networks (use model_id=3 by default; adjust if needed)
    rdc_pred = model_seal.predict(3, x_seal, w_vec)[0]  # [4, n_w] → C,c,K,k
    leak_pred = model_leak.predict(3, x_seal)[0, 0]

    geometry = {'hIn': x_seal[0, 0], 'hOut': x_seal[0, 1], 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
    fluid = {'mu': mu, 'rho': rho}
    op_conditions = {'dp': dp, 'psr': x_seal[0, 2], 'w_vec': w_vec}
    Leak, RDC, *_ = main_seal_solver(geometry, fluid, op_conditions)
    rdc_true = RDC[:, 2:]  # C,c,K,k

    rpm = w_vec * 30.0 / np.pi

    # 1) Leakage: solver vs. network (network gives single value → horizontal)
    _default_rcparams()
    fig, ax = plt.subplots()
    ax.plot(rpm, Leak, 'k-', lw=1.6, label='Solver')
    ax.hlines(leak_pred, xmin=rpm.min(), xmax=rpm.max(), colors='tab:blue', linestyles='--', lw=1.6, label='Network')
    ax.set_xlabel('Speed (RPM)')
    ax.set_ylabel('Leakage flow rate [kg/s]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'nn_leak_validation.png'), dpi=300, bbox_inches='tight')

    # 2) RDC curves: each channel C,c,K,k vs speed
    labels = ['C', 'c', 'K', 'k']
    _default_rcparams()
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    axs = axs.ravel()
    for i in range(4):
        axs[i].plot(rpm, rdc_true[:, i], 'k-', lw=1.6, label='Solver')
        axs[i].plot(rpm, rdc_pred[i, :], 'tab:blue', lw=1.6, linestyle='--', label='Network')
        axs[i].set_xlabel('Speed (RPM)')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'nn_rdc_validation.png'), dpi=300, bbox_inches='tight')


def run():
    # --- global settings ---
    w_range = np.array([500, 7000]) * np.pi / 30
    n_w = 14
    w_vec = np.linspace(w_range[0], w_range[1], n_w)
    w_oper = 3500 * np.pi / 30

    bs_params = {
        'mu_brg': 0.01,
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

    plot_campbell(eig_init, w_vec, out_path='initial_campbell.png', title='Campbell (initial)')
    plot_logdec_lowest(L_init, eig_init, w_vec, out_path='initial_logdec.png', n=4)
    plot_unbalance_response(amp_init, w_vec, ctx, out_path='initial_unbalance.png', nodes='key')

    # --- Optimization results (Pareto visuals) ---
    save_optimization_plots('checkpoints/latest.npz', ctx['n_brg'])

    # --- Optimized design: pick one pareto solution ---
    try:
        d = np.load('checkpoints/latest.npz')
        X_pareto, F_pareto = d['opt_X'], d['opt_F']
        # choose by minimum total leakage as an example
        idx = int(np.argmin(F_pareto[:, 0])) if len(F_pareto) else 0
        X_opt = X_pareto[idx:idx+1, :]
    except Exception:
        # fallback to initial if checkpoints not present
        X_opt = X_init.copy()

    eig_opt, amp_opt = analyze_design(X_opt, ctx, w_vec)
    L_opt = compute_logdec(eig_opt)

    plot_campbell(eig_opt, w_vec, out_path='optimized_campbell.png', title='Campbell (optimized)')
    plot_logdec_lowest(L_opt, eig_opt, w_vec, out_path='optimized_logdec.png', n=4)
    plot_unbalance_response(amp_opt, w_vec, ctx, out_path='optimized_unbalance.png', nodes='key')

    # --- Neural network model validation (leakage + RDC)
    save_nn_validation(w_vec, out_dir='.')

    print('Saved figures:')
    for f in [
        'initial_campbell.png', 'initial_logdec.png', 'initial_unbalance.png',
        'pareto_PCP.png', 'pareto_RAD.png', 'bearing_hist.png',
        'optimized_campbell.png', 'optimized_logdec.png', 'optimized_unbalance.png',
        'nn_leak_validation.png', 'nn_rdc_validation.png']:
        if os.path.exists(f):
            print(' -', f)


if __name__ == '__main__':
    run()
