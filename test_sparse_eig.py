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
from typing import Optional, Tuple
from time import time as tc

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
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
    
    unique_seals = {s.SealNet for s in seals}
    n_type_seal = len(unique_seals)

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
    }


# --- load rotor + models ---
ctx = build_rotor_and_models(bs_params)

n_brg = ctx['n_brg']
n_seal = ctx['n_seal']
n_type_seal = ctx['n_type_seal']
brgs = ctx['brgs']
seals = ctx['seals']
model_brg = ctx['model_brg']
model_seal = ctx['model_seal']
model_seal_leak = ctx['model_seal_leak']

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
#%%

n_type_brg = 55 # brg types
LB_brg_idx = 1; UB_brg_idx = 55
LB_cr = 10;  UB_cr = 30   # Cr/D = 10/10000 ~ 30/10000

LB_h = 100; UB_h = 500 # seal clearance range
LB_psr = -10;  UB_psr = 10   # -> *0.1 해서 [0,1.0]


lb = []
for _ in range(n_brg):
    lb += [LB_brg_idx, LB_cr]
for _, s in enumerate(seals):
    lb += [LB_h, LB_h, LB_psr - LB_psr*s.is_bush]
ub = []
for _ in range(n_brg):
    ub += [UB_brg_idx, UB_cr]
for _, s in enumerate(seals):
    ub += [UB_h, UB_h, UB_psr - UB_psr*s.is_bush]
    
def generate_population(lb, ub, n_pop):
    lb = np.array(lb)
    ub = np.array(ub)
    dim = len(lb)

    pop = np.random.uniform(lb, ub, size=(n_pop, dim))
    pop = np.round(pop).astype(int)
    return pop

n_pop = 50
X = generate_population(lb, ub, n_pop)

#%%
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

#%%
# Seals
seal_rdc = np.zeros((pop, n_seal, 4, w_vec.shape[0]), dtype=float)
seal_leak = np.zeros((pop, n_seal), dtype=float)
for t in range(n_type_seal):
    idx = idx_seal[t]
    if idx.size == 0:
        continue
    params_t = X_seal[:, idx]
    x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    rdc_flat = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, w_vec.shape[0])
    leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))
    seal_rdc[:, idx] = rdc_flat
    seal_leak[:, idx] = leak_flat

leak = seal_leak.sum(axis=1)
K_seal = seal_rdc[:, :, [2, 3, 3, 2], :] * rdc_signs[None, None, :, None]
C_seal = seal_rdc[:, :, [0, 1, 1, 0], :] * rdc_signs[None, None, :, None]

K_vals = np.concatenate([K_brg, K_seal], axis=1)
C_vals = np.concatenate([C_brg, C_seal], axis=1)

mat_M = ctx['mat_M']
mat_K_r = ctx['mat_K_r']
mat_C_g = ctx['mat_C_g']
support_dofs = ctx['support_dofs']
unb = ctx['unb']

rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
C_struct = np.zeros_like(mat_K_r)

K_all, Ceff_all = assemble_system_matrix(mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals)

M=mat_M
K_all=K_all
Ceff_all=Ceff_all
P, W, n, _ = K_all.shape
I = np.eye(n, dtype=M.dtype)
Z = np.zeros((n, n), dtype=M.dtype)
idx_x = np.arange(0, n, 4)
idx_y = idx_x + 2

from scipy.linalg import lu_factor, lu_solve
lu, piv = lu_factor(M)

def _build_state_space_blocks(Kp: np.ndarray, Ceff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    MinvK = lu_solve((lu, piv), Kp)
    MinvC = lu_solve((lu, piv), Ceff)
    A_top = np.concatenate([Z, I], axis=1)
    A_bot = np.concatenate([-MinvK, -MinvC], axis=1)
    return A_top, A_bot

import scipy


def eig_cpu(iw):
    with threadpool_limits(limits=1):
        w_fwd  = np.empty((P, n), dtype=np.complex128)
        phi_fwd= np.empty((P, n, n), dtype=np.complex128)
        for ip in range(P):
            Kp   = K_all[ip, iw]
            Ceff = Ceff_all[ip, iw]
            A_top, A_bot = _build_state_space_blocks(Kp, Ceff)
            A = np.concatenate([A_top, A_bot], axis=0)
            # w, v = np.linalg.eig(A)
            w, v = scipy.linalg.eig(A)
            idx = np.argsort(w.imag)
            w_fwd[ip]  = w[idx][n:]
            phi_fwd[ip]= v[:n, idx][:, n:]
        return iw, w_fwd, phi_fwd

t_s = tc()
n_jobs: int = -1
results = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(eig_cpu)(iw) for iw in range(W)
)

t_e = tc()
print(f't_eig={t_e-t_s}')
# %% Sparse eig in parallel (robust k-mode extraction)
# - Extract up to k forward modes (imag >= 0), sorted by |lambda|
# - Avoids broadcasting by allocating outputs for k modes, not n

from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import eigs

I_n = csr_matrix(np.eye(n, dtype=float))
B_sparse = bmat([[I_n, None], [None, csr_matrix(M)]], format='csr')
I_n_sparse = I_n  # already csr from above
k = int(12)
# Number of forward modes to keep consistently
k_sparse = int(min(max(1, k), n))

def _select_forward_modes(w, v, k_keep):
    """Select up to k_keep forward (imag>=0) modes sorted by |lambda|.
    Returns (lam_sel [k_keep], phi_sel [n,k_keep]) possibly padded with nan.
    """
    # Filter forward half-plane
    fwd_mask = (w.imag >= 0)
    if not np.any(fwd_mask):
        # No forward modes found; return nans
        lam_out = np.full((k_keep,), np.nan+1j*np.nan, dtype=np.complex128)
        phi_out = np.full((n, k_keep), np.nan+1j*np.nan, dtype=np.complex128)
        return lam_out, phi_out

    w_fwd = w[fwd_mask]
    v_fwd = v[:, fwd_mask]
    # Sort by magnitude (closest to origin first)
    order = np.argsort(np.abs(w_fwd))
    w_sorted = w_fwd[order]
    v_sorted = v_fwd[:, order]
    m = min(k_keep, w_sorted.shape[0])

    lam_out = np.full((k_keep,), np.nan+1j*np.nan, dtype=np.complex128)
    phi_out = np.full((n, k_keep), np.nan+1j*np.nan, dtype=np.complex128)
    lam_out[:m] = w_sorted[:m]
    phi_out[:, :m] = v_sorted[:n, :m]
    return lam_out, phi_out

def eig_cpu_sparse(iw):
    with threadpool_limits(limits=1):
        # Store only k_sparse modes to avoid shape mismatch
        w_fwd  = np.empty((P, k_sparse), dtype=np.complex128)
        phi_fwd= np.empty((P, n, k_sparse), dtype=np.complex128)
        for ip in range(P):
            Kp   = K_all[ip, iw]
            Ceff = Ceff_all[ip, iw]
            A_sparse = bmat([[csr_matrix((n, n)), I_n_sparse],
                            [-csr_matrix(Kp), -csr_matrix(Ceff)]], format='csr')

            # Request a bit more than needed to have enough forward modes
            N = 2 * n
            # ARPACK requirement: 1 < k_req < N
            k_req = max(2, min(2 * k_sparse, N - 2))
            
            w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse, sigma=0.0,
                                        which='LI', maxiter=2000)
            # try:
            #     w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse, sigma=0.0,
            #                             which='LM', maxiter=2000)
            # except Exception:
            #     # Fallback without shift-invert
            #     w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse,
            #                             which='SM', maxiter=4000)

            # Select forward modes and store
            lam_sel, phi_sel = _select_forward_modes(w_sparse, v_sparse, k_sparse)
            w_fwd[ip] = lam_sel
            phi_fwd[ip] = phi_sel
        return iw, w_fwd, phi_fwd

t_s = tc()
n_jobs: int = -1
results_sparse = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(eig_cpu_sparse)(iw) for iw in range(W)
)

t_e = tc()
print(f't_eig={t_e-t_s}')

#%%
# Compare eigenvalues and eigenvectors (dense vs sparse)
ic = 0
r = results[ic]
r1 = results_sparse[ic]

# r, r1 are tuples: (iw, w_fwd, phi_fwd)
iw_dense = r[0]
iw_sparse = r1[0]
w_dense_all = r[1]        # shape: (P, n)
phi_dense_all = r[2]      # shape: (P, n, n)
w_sparse_all = r1[1]      # shape: (P, k_sparse)
phi_sparse_all = r1[2]    # shape: (P, n, k_sparse)

rel_errs = []
vec_sims = []
for ip in range(P):
# for ip in range(1):
    # Dense forward modes already extracted; sort by |lambda|
    w_d_fwd = w_dense_all[ip]                 # (n,)
    phi_d_fwd = phi_dense_all[ip]             # (n, n) columns correspond to w_d_fwd
    order_d = np.argsort(np.abs(w_d_fwd))
    take = min(k_sparse, w_d_fwd.shape[0])
    cols_d = order_d[:take]
    w_d = w_d_fwd[cols_d]
    phi_d = phi_d_fwd[:, cols_d]

    # Sparse forward modes were already filtered and sorted by |lambda| in _select_forward_modes
    w_s = w_sparse_all[ip]
    phi_s = phi_sparse_all[ip]
    valid = np.isfinite(w_s.real) & np.isfinite(w_s.imag)
    if not np.any(valid):
        continue
    w_s = w_s[valid]
    phi_s = phi_s[:, valid]

    m = min(w_s.shape[0], w_d.shape[0])
    if m == 0:
        continue
    w_s = w_s[:m]
    w_d = w_d[:m]
    phi_s = phi_s[:, :m]
    phi_d = phi_d[:, :m]

    # Eigenvalue relative errors
    rel = np.abs(w_s - w_d) / np.maximum(1.0, np.abs(w_d))
    rel_errs.append(rel)

    # Eigenvector similarity, invariant to complex scaling: |<x,y>|/(||x||·||y||)
    # Normalize columns
    denom_d = np.linalg.norm(phi_d, axis=0, keepdims=True)
    denom_s = np.linalg.norm(phi_s, axis=0, keepdims=True)
    # Guard against zeros
    denom_d[denom_d == 0] = 1.0
    denom_s[denom_s == 0] = 1.0
    pd = phi_d / denom_d
    ps = phi_s / denom_s
    sim = np.abs(np.sum(np.conj(pd) * ps, axis=0))
    vec_sims.append(sim)

if rel_errs:
    rel_errs = np.concatenate(rel_errs)
    vec_sims = np.concatenate(vec_sims)
    print(f"eig compare iw(dense)={iw_dense}, iw(sparse)={iw_sparse}")
    print(f"rel_err_mean={float(np.mean(rel_errs)):.3e}, rel_err_max={float(np.max(rel_errs)):.3e}")
    print(f"vec_sim_mean={float(np.mean(vec_sims)):.3e}, vec_sim_min={float(np.min(vec_sims)):.3e}")
else:
    print("No valid forward modes to compare.")

for im in range(k):
    m1 = pd[:,im]
    m2 = ps[:,im]

    mx1 = np.abs(m1[np.arange(0,n,4)])
    my1 = np.abs(m1[np.arange(2,n,4)])

    mx2 = np.abs(m2[np.arange(0,n,4)])
    my2 = np.abs(m2[np.arange(2,n,4)])

    ma1 = np.sqrt(mx1**2+my1**2)
    ma2 = np.sqrt(mx2**2+my2**2)

    x = np.arange(0,n//4)
    plt.plot(x,ma1)
    plt.plot(x,ma2)
    plt.show()

#%% 
# MAC-based mode tracking and Campbell diagram comparison (dense vs sparse)
def _normalize_modes(phi: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(phi, axis=0, keepdims=True)
    d[d == 0] = 1.0
    return phi / d

def _mac_matrix(phi_prev: np.ndarray, phi_cur: np.ndarray) -> np.ndarray:
    # Columns must be normalized for MAC = |phi_prev^H phi_cur|
    return np.abs(phi_prev.conj().T @ phi_cur)

def _greedy_match(mac: np.ndarray, k_target: int) -> np.ndarray:
    # Returns array of length k_target, mapping prev_idx -> cur_idx (or -1 if missing)
    r, c = mac.shape
    used_cols = np.zeros(c, dtype=bool)
    mapping = -np.ones(k_target, dtype=int)
    # Flatten and sort descending by MAC
    order = np.argsort(mac.ravel())[::-1]
    count = 0
    for idx in order:
        i = idx // c
        j = idx % c
        if i < k_target and mapping[i] == -1 and not used_cols[j]:
            mapping[i] = j
            used_cols[j] = True
            count += 1
            if count == k_target:
                break
    return mapping

def _extract_modes_over_speeds(results_list, ip_sel: int, k_keep: int, is_sparse: bool):
    # Ensure results ordered by iw
    res_sorted = sorted(results_list, key=lambda x: x[0])
    eigs = []
    modes = []
    for iw_local, w_all, phi_all in res_sorted:
        if is_sparse:
            w_cur = w_all[ip_sel]
            phi_cur = phi_all[ip_sel]
            # Filter valid
            valid = np.isfinite(w_cur.real) & np.isfinite(w_cur.imag)
            w_cur = w_cur[valid]
            phi_cur = phi_cur[:, valid]
        else:
            # Dense: forward modes in w_all[ip_sel] already, sort by |lambda|
            w_fwd = w_all[ip_sel]
            phi_fwd = phi_all[ip_sel]
            order = np.argsort(np.abs(w_fwd))
            w_cur = w_fwd[order][:k_keep]
            phi_cur = phi_fwd[:, order][:, :k_keep]

        # Normalize vectors for MAC robustness
        phi_cur = _normalize_modes(phi_cur)
        # Trim/pad to k_keep
        m = min(k_keep, w_cur.shape[0])
        if m < k_keep:
            w_pad = np.full((k_keep,), np.nan+1j*np.nan, dtype=np.complex128)
            phi_pad = np.full((phi_cur.shape[0], k_keep), np.nan+1j*np.nan, dtype=np.complex128)
            if m > 0:
                w_pad[:m] = w_cur[:m]
                phi_pad[:, :m] = phi_cur[:, :m]
            w_cur = w_pad
            phi_cur = phi_pad
        eigs.append(w_cur)
        modes.append(phi_cur)
    return np.array(eigs), np.stack(modes, axis=0)  # (W, k), (W, n, k)

def _track_modes_mac(modes_seq: np.ndarray, eigs_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # modes_seq: (W, n, k), eigs_seq: (W, k)
    W_local, n_local, k_local = modes_seq.shape
    eigs_tr = np.full_like(eigs_seq, np.nan+1j*np.nan)
    modes_tr = np.full_like(modes_seq, np.nan+1j*np.nan)
    # Initialize with first speed
    eigs_tr[0] = eigs_seq[0]
    modes_tr[0] = modes_seq[0]
    for iw_local in range(1, W_local):
        prev = modes_tr[iw_local-1]
        cur = modes_seq[iw_local]
        # Handle NaNs by zeroing those columns (so MAC=0)
        prev_ok = np.isfinite(prev).all(axis=0)
        cur_ok = np.isfinite(cur).all(axis=0)
        prev_n = _normalize_modes(np.where(prev_ok[None, :], prev, 0.0))
        cur_n = _normalize_modes(np.where(cur_ok[None, :], cur, 0.0))
        mac = _mac_matrix(prev_n, cur_n)
        mapping = _greedy_match(mac, k_local)
        # Reorder current by mapping
        cur_eigs = eigs_seq[iw_local]
        eigs_re = np.full((k_local,), np.nan+1j*np.nan, dtype=np.complex128)
        modes_re = np.full((n_local, k_local), np.nan+1j*np.nan, dtype=np.complex128)
        for i_prev in range(k_local):
            j_cur = mapping[i_prev]
            if j_cur >= 0:
                eigs_re[i_prev] = cur_eigs[j_cur]
                modes_re[:, i_prev] = cur[:, j_cur]
        eigs_tr[iw_local] = eigs_re
        modes_tr[iw_local] = modes_re
    return eigs_tr, modes_tr

# Use first population for comparison
ip_sel = 0
W_local = W

# Dense sequences per speed
d_eigs, d_modes = _extract_modes_over_speeds(results, ip_sel, k_sparse, is_sparse=False)
# Sparse sequences per speed
s_eigs, s_modes = _extract_modes_over_speeds(results_sparse, ip_sel, k_sparse, is_sparse=True)

# Track with MAC
d_eigs_tr, d_modes_tr = _track_modes_mac(d_modes, d_eigs)
s_eigs_tr, s_modes_tr = _track_modes_mac(s_modes, s_eigs)

# Compare Campbell diagrams (frequency vs rotor speed)
rpm = w_vec * 30.0 / np.pi
freq_dense = np.imag(d_eigs_tr) / (2*np.pi)
freq_sparse = np.imag(s_eigs_tr) / (2*np.pi)

# Simple quantitative comparison (ignore NaNs)
mask = np.isfinite(freq_dense) & np.isfinite(freq_sparse)
if np.any(mask):
    diff = np.abs(freq_dense - freq_sparse)[mask]
    print(f"Campbell diff mean={float(np.mean(diff)):.3e} Hz, max={float(np.max(diff)):.3e} Hz")

# Plot overlay
plt.figure()
for j in range(k_sparse):
    plt.plot(rpm, freq_dense[:, j], '-', alpha=0.7, label='dense' if j == 0 else None)
for j in range(k_sparse):
    plt.plot(rpm, freq_sparse[:, j], '--', alpha=0.7, label='sparse' if j == 0 else None)
plt.xlabel('Speed (RPM)')
plt.ylabel('Frequency (Hz)')
plt.title('Campbell Diagram: Dense vs Sparse (MAC-tracked)')
plt.legend()
plt.tight_layout()
plt.show()
#%%
# # %%


# # Sparse eig test and speed/accuracy comparison
# # Compute k smallest eigenvalues (by |lambda|, imag >= 0) using sparse ARPACK
# # on the generalized problem A z = lambda B z where
# # A = [[0, I], [-K, -C]], B = [[I, 0], [0, M]]
# from scipy.sparse import csr_matrix, bmat
# from scipy.sparse.linalg import eigs

# ip0 = 0
# iw0 = W // 2 if W > 0 else 0
# Kp0 = K_all[ip0, iw0]
# Ceff0 = Ceff_all[ip0, iw0]

# # Dense full eig (reference)
# t0 = tc()
# A_top0, A_bot0 = _build_state_space_blocks(Kp0, Ceff0)
# A_dense = np.concatenate([A_top0, A_bot0], axis=0)
# w_dense, _ = scipy.linalg.eig(A_dense)
# t1 = tc()

# # Keep positive-imag roots and sort by |lambda|
# pos_dense = w_dense[w_dense.imag >= 0]
# pos_dense = pos_dense[np.argsort(np.abs(pos_dense))]

# # Target number of modes
# k = int(min(10, max(1, pos_dense.shape[0] // 2)))

# # Sparse generalized eig with shift-invert near zero
# I_n = csr_matrix(np.eye(n, dtype=float))
# A_sparse = bmat([[csr_matrix((n, n)), I_n], [-csr_matrix(Kp0), -csr_matrix(Ceff0)]], format='csr')
# B_sparse = bmat([[I_n, None], [None, csr_matrix(M)]], format='csr')

# t2 = tc()
# try:
#     w_sparse, _ = eigs(A_sparse, k=k*2, M=B_sparse, sigma=0.0, which='LM', tol=1e-8, maxiter=2000)
# except Exception:
#     # Fallback without shift-invert
#     w_sparse, _ = eigs(A_sparse, k=k*2, M=B_sparse, which='SM', tol=1e-8, maxiter=4000)
# t3 = tc()

# pos_sparse = w_sparse[w_sparse.imag >= 0]
# pos_sparse = pos_sparse[np.argsort(np.abs(pos_sparse))]

# # Align and compare first k modes
# k_eff = int(min(k, pos_dense.shape[0], pos_sparse.shape[0]))
# lam_d = pos_dense[:k_eff]
# lam_s = pos_sparse[:k_eff]

# rel_err = np.abs(lam_s - lam_d) / np.maximum(1.0, np.abs(lam_d))

# print(f"dense_ref_time={t1 - t0:.4f}s, sparse_time={t3 - t2:.4f}s, k={k_eff}")
# print("abs(lam_dense)[:5]=", np.round(np.abs(lam_d[:5]), 6))
# print("abs(lam_sparse)[:5]=", np.round(np.abs(lam_s[:5]), 6))
# print("rel_err_mean=", float(np.mean(rel_err)))
# print("rel_err_max=", float(np.max(rel_err)))

# %%

iw = 0
ip = 0
w_fwd  = np.empty((P, k_sparse), dtype=np.complex128)
phi_fwd= np.empty((P, n, k_sparse), dtype=np.complex128)
Kp   = K_all[ip, iw]
Ceff = Ceff_all[ip, iw]
A_sparse = bmat([[csr_matrix((n, n)), I_n_sparse],
                [-csr_matrix(Kp), -csr_matrix(Ceff)]], format='csr')

# Request a bit more than needed to have enough forward modes
N = 2 * n
# ARPACK requirement: 1 < k_req < N
k_req = max(2, min(2 * k_sparse, N - 2))
w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse, sigma=0.0,
                            which='LI', maxiter=2000)
# try:
#     w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse, sigma=0.0,
#                             which='LM', maxiter=2000)
# except Exception:
#     # Fallback without shift-invert
#     w_sparse, v_sparse = eigs(A_sparse, k=k_req, M=B_sparse,
#                             which='SM', maxiter=4000)


A_top, A_bot = _build_state_space_blocks(Kp, Ceff)
A = np.concatenate([A_top, A_bot], axis=0)
# w, v = np.linalg.eig(A)
w, v = scipy.linalg.eig(A)

w0 = np.sort(w.imag)
w0 = w0[n:n+k]
w1 = np.sort(w_sparse.imag)
w1 = w1[k:]
x_ = np.arange(0,k)

plt.plot(x_,w0,'-o')
plt.plot(x_,w1,'-o')
plt.show()
# %%
