#%%
import os
import numpy as np
from import_data import rotor_import, plot_rotor_3d
from loader_brg_seal import BearingNondModel, SealFNOModel, SealDONModel, SealLeakModel
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from solver_rotordyn import assemble_system_matrix
from solver_rotordyn import eig_batch as eig
from solver_rotordyn import unbalance_response_batch_cpu_parallel as unbalanced_response
import time
from scipy.signal import find_peaks

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.callback import Callback
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column

import pickle

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

## Define constant values
w_range = np.array([1000, 6000]) * np.pi / 30
w_oper = 3500 * np.pi / 30
oper = {
        'w_min': w_range[0],
        'w_max': w_range[1],
        'range': w_oper,
    }
bs_params = {
        'mu_brg': 0.01, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }

n_w = 12
n_pop = 200
n_max_gen = 500

## Select algorithm

# which_algorithm = "NSGA2"
which_algorithm = "UNSGA3"

#%%

w_vec = np.linspace(w_range[0], w_range[1], n_w)

## Import and generate rotor data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

print("Importing rotor data...")

n_ele, n_node, n_dof, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb, brgs, seals, support_dofs = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

# plot_rotor_3d(rotor_elements=rotor_elements)

matrix_params = {
    'mat_M': mat_M,
    'mat_K_r': mat_K_r,
    'mat_C_g': mat_C_g,
    'n_ele': n_ele,
    'n_node': n_node,
    'n_dof': n_dof,
}
print("Rotor data loaded >.<\n")

#%%
print("Loading bearing and seal models...")
## Load brg model
model_brg = BearingNondModel()
f_brg_dim = np.array([[1, 1e-4],[1, 1e-4]])

## brg property calculation function
# --- input: Brg parameters and w_vec
# Brg parameters = [Brg index: int, clearance ratio] (n_pop, n_brg, 2)
# w_vec shape = (n_w,)
# --- output: K_brg_batch, C_brg_batch
# K_brg_batch, C_brg_batch shape = (n_pop, n_brg, 4, n_w)

# # example
# brg_indices = np.random.randint(1, 55, size=(n_pop, n_brg, 1))
# cr_ratios = np.random.uniform(10, 30, size=(n_pop, n_brg, 1))
# params_batch = np.concatenate([brg_indices, cr_ratios], axis=2)
# K_brg_batch, C_brg_batch = brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=params_batch, w_vec=w_vec)

## Load seal model
# seal = SealFNOModel()
model_seal = SealDONModel(device=device)
model_seal_leak = SealLeakModel(device=device)

## seal property prediction function
# --- input: Seal geometric parameters, w_vec
# geometric parameters shape = (n_pop, n_seal, 3)
# w_vec shape = (n_w,)
# --- output: leakage flow rate and rotordynamic coefficient
# leakage flow rate shape = (n_pop*n_seal, 1)
# rotordynamic coefficient = (n_pop*n_seal, 4, n_w), order = [C c K k]
print("Bearing and Seal models loaded\n")

n_type_seal = 3 # seal types

groups = defaultdict(list)
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
outputs = [None] * len(seals)

idx_seal = [np.array(groups[t+1], dtype=int) for t in range(n_type_seal)]

f_seal_dim = [1e-6, 1e-6, 1e-1]
rdc_signs = np.array([1, 1, -1, 1])

# example
# h_in  = np.random.randint(100, 500, size=(n_pop, n_seal, 1))
# h_out = np.random.randint(100, 500, size=(n_pop, n_seal, 1))
# psr   = np.random.randint(-10, 10,   size=(n_pop, n_seal, 1))
# params_batch = np.concatenate([h_in, h_out, psr], axis=2).reshape(-1, 3)
# params_batch = params_batch * f_seal_dim
# leak = model_seal_leak.predict(1, params_batch)
# seal_rdc = model_seal.predict(1, params_batch, w_vec)
# rdc_flat = seal_rdc.reshape(n_pop, n_seal, 4, n_w)
# K_seal = rdc_flat[:,:,[2, 3, 3, 2],:] * rdc_signs[None, None, :, None]

#%%
print("Defining optimization problem...")
## prepare rotordynamic analysis
# --- input: mass, damping, stiffness matrix, w_vec, harmonic excitation cases
# --- output: critical speed, logarithmic decrement, damping ratio, unbalanced response,
# amplification factor, separation margin, minimum clearance, ...
rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
C_struct = np.zeros_like(mat_K_r)

## optimization parameter
n_type_brg = 55 # brg types
LB_brg_idx = 1; UB_brg_idx = 55
LB_cr = 10;  UB_cr = 30   # Cr/D = 10/10000 ~ 30/10000

n_type_seal = 3 # seal types
LB_h = 100; UB_h = 500 # seal clearance range
LB_psr = -10;  UB_psr = 10   # -> *0.1 해서 [0,1.0]

## Define optimization problem with vector computation
n_var = 2 * n_brg + 3 * n_seal
n_objs = 6  # [total_leak, power_loss, max_AF, -min_logdec, max_ampRatioBrg, max_ampRatioSeal]



#%%
d = np.load('checkpoints/latest.npz')
X_pop, F_pop = d['pop_X'], d['pop_F']
X_pareto, F_pareto = d['opt_X'], d['opt_F']

#%%

idx_sorted = np.argsort(F_pareto[:,3])
# F_pareto[idx_sorted,3]


idx_chk = idx_sorted[-1]
X = X_pareto[idx_chk,:]
#%%

# X: [pop, n_var] 정수 행렬
pop = 1
X_brg = X[:n_brg*2].reshape(pop, n_brg, 2)
X_seal = X[n_brg*2:].reshape(pop, n_seal, 3)

#%%
F = np.zeros((pop, n_objs), dtype=float) # objective function value

x_brg = X_brg * f_brg_dim
K_brg, C_brg, loss_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
seal_leak = np.zeros((pop, n_seal), dtype=float)

for t in range(n_type_seal):
    idx = idx_seal[t]
    if len(idx) == 0:
        continue
    
    params_t = X_seal[:, idx]
    x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    
    leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))        # [pop, m]
    rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)  # [pop, m, 4, n_w]
    
    seal_leak[:, idx] = leak_flat
    seal_rdc[:, idx] = rdc_flat

K_seal = seal_rdc[:,:,[2, 3, 3, 2],:] * rdc_signs[None, None, :, None]
C_seal = seal_rdc[:,:,[0, 1, 1, 0],:] * rdc_signs[None, None, :, None]

K_vals = np.concatenate([K_brg, K_seal], axis=1)  # (pop, n_sup, 4, n_w)
C_vals = np.concatenate([C_brg, C_seal], axis=1)  # (pop, n_sup, 4, n_w)

K_all, Ceff_all = assemble_system_matrix(
    mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals
)

eigvals, _ = eig(
    M=mat_M,
    K_all=K_all,
    Ceff_all=Ceff_all,
    track=True,
)

harmonic = unbalanced_response(
    M=mat_M,
    unb=unb,
    K_all=K_all,
    Ceff_all=Ceff_all,
    w_vec=w_vec,
)

# Operating index
idx_op = int(np.argmin(np.abs(w_vec - w_oper)))

# 1) Total seal leakage per individual (sum across all seals)
F[:, 0] = seal_leak.sum(axis=1)

# 2) Total bearing power loss (sum across all brgs)
F[:, 1] = loss_brg[:,:,:,idx_op].sum(axis=1).squeeze()

#%%
# Node-wise amplitudes from harmonic response
# harmonic: [pop, n_w, n_dof, 1] complex
assert n_dof == 4 * n_node
idx_x = np.arange(n_node) * 4
idx_y = idx_x + 2
Ux = harmonic[:, :, idx_x, 0]  # [pop, n_w, n_node]
Uy = harmonic[:, :, idx_y, 0]
amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)  # [pop, n_w, n_node]

#%%

plt.figure()
plt.plot(amp.squeeze())


#%%
# 3) Amplification Factor using half-power bandwidth: AF = Nc / (N2 - N1)
# Use calibration nodes (bearings + seals). If empty, fallback to all nodes.
brg_nodes = np.array([b.node for b in brgs], dtype=int)
seal_nodes = np.array([s.node for s in seals], dtype=int)
cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))

eps = 1e-18
AF_max = np.zeros(pop, dtype=float)
w = w_vec
p = 0
af_p = 0.0
A = amp[p, :, cal_nodes]  # [n_cal, n_w] due to fancy indexing
for c in range(len(cal_nodes)):
    y = A[c, :]

    if y.size < 3:
        continue
    pk, _ = find_peaks(y)
    if pk.size == 0:
        continue
    plt.plot(w_vec,y)
    
    print(pk.shape)
    for j in pk:
        Ac = y[j]
        print(Ac)
        if Ac <= eps:
            continue
        yhpp = Ac / np.sqrt(2.0)
        plt.axhline(yhpp, color='r')
        plt.axvline(w[j], color='r')
        
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
                plt.axvline(x0, color='b')
                plt.axhline(y0, color='b')
                plt.show()
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
        if af_peak > af_p:
            af_p = af_peak
    print(af_p)
AF_max[p] = af_p
F[:, 2] = AF_max

#%%
# min log_dec
alpha = np.real(eigvals)  # [pop, n_w, 2n]
beta  = np.imag(eigvals)
alpha8 = alpha[:,:,:8]
beta8  = beta[:,:,:8]
logdec = -2 * np.pi * alpha8 / np.sqrt(alpha8**2 + beta8**2)
min_logdec = np.min(logdec, axis=(1, 2))

F[:, 3] = -min_logdec



# wn = np.sqrt(np.maximum(alpha8**2 + beta8**2, 1e-30))
# zeta = np.clip(-alpha8 / (wn + 1e-30), a_min=1e-12, a_max=0.999999)


# beta_pos = np.where(beta > 1e-12, beta, np.inf)
# idx_sorted = np.argsort(beta_pos, axis=2)
# take = min(8, idx_sorted.shape[2])
# sel = idx_sorted[:, :, :take]  # [pop, n_w, take]
# ip = np.arange(pop)[:, None, None]
# iw = np.arange(n_w)[None, :, None]
# alpha8 = alpha[ip, iw, sel]
# beta8  = beta[ip, iw, sel]
# wn = np.sqrt(np.maximum(alpha8**2 + beta8**2, 1e-30))
# zeta = np.clip(-alpha8 / (wn + 1e-30), 1e-12, 0.999999)
# logdec = 2.0 * np.pi * zeta / np.sqrt(1.0 - zeta**2)
# valid = np.isfinite(beta_pos[ip, iw, sel])
# logdec_masked = np.where(valid, logdec, np.inf)
# min_logdec = np.min(logdec_masked, axis=(1, 2))  # [pop]
# F[:, 3] = -min_logdec

# # alpha8, beta8, logdec (너가 이미 계산) 가정
# alpha_max = np.max(alpha8, axis=(1,2))          # [pop]
# g = np.min(logdec_masked, axis=(1,2))           # [pop]  (유효치 외 inf)

# # A안: 기본형
# big = 1e6
# unstable = alpha_max > 0
# f_dec = np.where(unstable, big + np.maximum(0.0, alpha_max), -g)

# F[:, 3] = f_dec  # 대수감쇠 목적

#%%
# 5) Bearing amplitude ratio (%): amp(node_brg,:)/Cp * 100 -> global max
brg_nodes = np.array([b.node for b in brgs], dtype=int)
if brg_nodes.size > 0:
    brg_ids = X_brg[:, :, 0].astype(int)                         # [pop, n_brg]
    cr_ratio = (X_brg[:, :, 1] * f_brg_dim[0, 1]).astype(float)  # scaled
    Db_all = np.array([b.Db for b in brgs], dtype=float)[None, :]
    Cr = Db_all * cr_ratio                                       # [pop, n_brg]
    # Vectorized Mp lookup
    # Build lookup array for Mp where index = bearing_id
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

# 6) Seal amplitude ratio (%): amp(node_seal,:)/min(h_in,h_out) * 100 -> global max
if seal_nodes.size > 0:
    h_in  = (X_seal[:, :, 0].astype(float) * f_seal_dim[0])   # [pop, n_seal]
    h_out = (X_seal[:, :, 1].astype(float) * f_seal_dim[1])   # [pop, n_seal]
    h_min = np.minimum(h_in, h_out)
    amp_seal = amp[:, :, seal_nodes]                          # [pop, n_w, n_seal]
    amp_ratio_seal = (amp_seal / (h_min[:, None, :] + eps)) * 100.0
    F[:, 5] = amp_ratio_seal.reshape(pop, -1).max(axis=1)
else:
    F[:, 5] = 0.0





#%%
# %%
