#%%
import os
import numpy as np
from import_data import rotor_import
from loader_brg_seal import BearingNondModel, SealFNOModel, SealDONModel, SealLeakModel
from collections import defaultdict
from torch.linalg import inv
import torch
import matplotlib.pyplot as plt
from solver_rotordyn import assemble_system_matrix
from solver_rotordyn import eig_batch_cpu_parallel as eig
from solver_rotordyn import unbalance_response_batch_cpu_parallel as unbalanced_response
import time

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

##
device = "cuda" if torch.cuda.is_available() else "cpu"

## Define constant values
w_range = np.array([500, 6000]) * np.pi / 30
w_oper = 3500 * np.pi / 30
oper = {
        'w_min': w_range[0],
        'w_max': w_range[1],
        'range': w_oper,
    }
bs_params = {
        'mu_brg': 0.04, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }

n_w = 12
n_pop = 100
w_vec = np.linspace(w_range[0], w_range[1], n_w)

## Import and generate rotor data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

n_ele, n_node, n_dof, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb, brgs, seals, support_dofs = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

matrix_params = {
    'mat_M': mat_M,
    'mat_K_r': mat_K_r,
    'mat_C_g': mat_C_g,
    'n_ele': n_ele,
    'n_node': n_node,
    'n_dof': n_dof,
}

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
model_seal = SealDONModel()
model_seal_leak = SealLeakModel()

## seal property prediction function
# --- input: Seal geometric parameters, w_vec
# geometric parameters shape = (n_pop, n_seal, 3)
# w_vec shape = (n_w,)
# --- output: leakage flow rate and rotordynamic coefficient
# leakage flow rate shape = (n_pop*n_seal, 1)
# rotordynamic coefficient = (n_pop*n_seal, 4, n_w), order = [C c K k]

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
LB_psr = 0;  UB_psr = 10   # -> *0.1 해서 [0,1.0]

## Define optimization problem with vector computation
n_var = 2*n_brg + 3 * n_seal
n_objs = 3 # temporary

class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 경계+정수 보정
        X = np.clip(X, problem.xl, problem.xu)
        return np.rint(X).astype(int)

class RotordynamicProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=n_objs, n_constr=0, xl=self._xl(), xu=self._xu(), elementwise_evaluation=False)

    def _xl(self):
        # 전 씰 동일 범위라면 반복
        lb = []
        for _ in range(n_brg):
            lb += [LB_brg_idx, LB_cr]
        for _ in range(n_seal):
            lb += [LB_h, LB_h, LB_psr]
        return np.array(lb, dtype=int)

    def _xu(self):
        ub = []
        for _ in range(n_brg):
            ub += [UB_brg_idx, UB_cr]
        for _ in range(n_seal):
            ub += [UB_h, UB_h, UB_psr]
        return np.array(ub, dtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        # X: [pop, n_var] 정수 행렬
        pop = X.shape[0]
        X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
        X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)
        
        F = np.zeros((pop, n_objs), dtype=float) # objective function value
        
        x_brg = X_brg * f_brg_dim
        K_brg, C_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)
        
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

        eigvals = eig(
            M=mat_M,
            K_all=K_all,
            Ceff_all=Ceff_all,
        )
        
        harmonic = unbalanced_response(
            M=mat_M,
            unb=unb,
            K_all=K_all,
            Ceff_all=Ceff_all,
            w_vec=w_vec,
        )
        
        #TODO calculate objective functions
        
        # Example objective: per-type leakage sum
        for t in range(n_type_seal):
            if len(idx_seal[t]) == 0:
                continue
            F[:, t] = seal_leak[:, idx_seal[t]].sum(axis=1)

        out["F"] = F

## Start optimization
sampling = IntegerRandomSampling()
crossover = TwoPointCrossover()
mutation = PolynomialMutation(eta=20)
repair = RoundRepair()


n_pop = 200
algorithm = NSGA2(
    pop_size=n_pop,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
    repair=repair
)

termination = get_termination("n_gen", 1)
problem = RotordynamicProblem()

#%%
t_start = time.time()

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True
)

t_end = time.time()
t_elapsed = t_end - t_start
print(f"n_pop = ",n_pop)
print(f"elapsed time: ",t_elapsed)




#%%
X = np.array([np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)]), 
np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)])])

pop = X.shape[0]

X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

F = np.zeros((pop, n_objs), dtype=float) # objective function value

x_brg = X_brg * f_brg_dim
K_brg, C_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

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

eigvals = eig(
    M=mat_M,
    K_all=K_all,
    Ceff_all=Ceff_all,
)

harmonic = unbalanced_response(
    M=mat_M,
    unb=unb,
    K_all=K_all,
    Ceff_all=Ceff_all,
    w_vec=w_vec,
)
# %%
eig_ = eigvals[0,:,:]
eig0 = np.imag(eig_)

plt.figure()
plt.plot(w_vec*30/np.pi,eig0)





# %%
