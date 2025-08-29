#%%
import os
import numpy as np
from import_data import rotor_import
from loader_brg_seal import BearingNondModel, SealFNOModel, SealDONModel, SealLeakModel
from collections import defaultdict
from torch.linalg import inv
import torch
import matplotlib.pyplot as plt
from solver_rotordyn import eig_batch_cpu_parallel as eig
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

        eigvals = eig(
            M=mat_M,
            K_shaft=mat_K_r,
            C_struct=np.zeros_like(mat_K_r),
            G=mat_C_g,
            w_vec=w_vec,
            rows_sup=rows_sup,
            cols_sup=cols_sup,
            K_vals=K_vals,
            C_vals=C_vals,
        )  # shape: (pop, n_w, 2*n_dof)
        
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










# for order in range(3):
#     n_pop = 4**order
#     algorithm = NSGA2(
#         pop_size=n_pop,
#         sampling=sampling,
#         crossover=crossover,
#         mutation=mutation,
#         eliminate_duplicates=True,
#         repair=repair
#     )

#     termination = get_termination("n_gen", 1)
#     problem = RotordynamicProblem()

#     t_start = time.time()

#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=42,
#         save_history=True,
#         verbose=True
#     )

#     t_end = time.time()
#     t_elapsed = t_end - t_start
#     print(f"n_pop = ",n_pop)
#     print(f"elapsed time: ",t_elapsed)

# X_opt = res.X
# F_opt = res.F
# print("pareto solutions:", X_opt.shape, "objectives:", F_opt.shape)



#%%

# X_test = np.concatenate([np.tile([1, 10*1e-4],2), np.tile([100*1e-6, 100*1e-6, 0], n_seal)])
# out = problem.evaluate(X_test)






# def xy_dofs(node_idx: int) -> tuple[int,int]:
#     """노드 인덱스(0-based) -> 전역 행렬에서 x,y DOF 인덱스."""
#     base = 4*node_idx
#     return base + 0, base + 2   # [x, thx, y, thy] 가정

# def submat_from_bearing(vec4: np.ndarray) -> np.ndarray:
#     """vec4 = [Kxx, Kxy, Kyx, Kyy] 또는 [Cxx, Cxy, Cyx, Cyy] -> (2,2)"""
#     return np.array([[vec4[0], vec4[1]],
#                      [vec4[2], vec4[3]]], dtype=float)

# def submat_from_seal(K: float, k: float, C: float, c: float) -> tuple[np.ndarray,np.ndarray]:
#     """씰 구조 Kxx=Kyy=K, Kxy=-Kyx=k; C도 동일 패턴."""
#     K2 = np.array([[K,  k],
#                    [-k, K]], dtype=float)
#     C2 = np.array([[C,  c],
#                    [-c, C]], dtype=float)
#     return K2, C2

# def assemble_support_mats_per_speed(n_dof: int,
#                                     w_vec: np.ndarray,
#                                     # 베어링 계수: K_brg, C_brg shape = (n_pop, n_w, 4, n_brg)
#                                     K_brg: np.ndarray,
#                                     C_brg: np.ndarray,
#                                     brg_nodes: list[int],   # 각 베어링의 노드(0-based)
#                                     # 씰 계수: 각 타입/개체 정리 후 speed별로 (K,k,C,c)를 제공
#                                     seal_KkCc_per_speed: list[dict],
#                                     # 예: [{'node': i_node, 'K': Kw, 'k': kw, 'C': Cw, 'c': cw}, ...]
#                                    ):
#     """
#     반환: K_sup, C_sup shape = (n_pop, n_w, n_dof, n_dof)
#     """
#     n_pop, n_w, _, n_brg = K_brg.shape
#     K_sup = np.zeros((n_pop, n_w, n_dof, n_dof))
#     C_sup = np.zeros((n_pop, n_w, n_dof, n_dof))

#     for j in range(n_pop):
#         for iw, w in enumerate(w_vec):
#             # --- 베어링 추가 ---
#             for b in range(n_brg):
#                 ix, iy = xy_dofs(brg_nodes[b])
#                 K2 = submat_from_bearing(K_brg[j, iw, :, b])   # (2,2)
#                 C2 = submat_from_bearing(C_brg[j, iw, :, b])   # (2,2)

#                 # 전역 조립
#                 dofs = [ix, iy]
#                 for a, I in enumerate(dofs):
#                     for b_, J in enumerate(dofs):
#                         K_sup[j, iw, I, J] += K2[a, b_]
#                         C_sup[j, iw, I, J] += C2[a, b_]

#             # --- 씰 추가 ---
#             # seal_KkCc_per_speed 는 같은 iw에 대응하는 리스트(여러 씰)라고 가정
#             for item in seal_KkCc_per_speed[iw]:
#                 ix, iy = xy_dofs(item['node'])
#                 K2, C2 = submat_from_seal(item['K'], item['k'], item['C'], item['c'])
#                 dofs = [ix, iy]
#                 for a, I in enumerate(dofs):
#                     for b_, J in enumerate(dofs):
#                         K_sup[j, iw, I, J] += K2[a, b_]
#                         C_sup[j, iw, I, J] += C2[a, b_]
#     return K_sup, C_sup

# def build_system_mats(mat_K_r: np.ndarray,
#                       mat_C_g: np.ndarray,
#                       K_sup: np.ndarray,
#                       C_sup: np.ndarray,
#                       w_vec: np.ndarray):
#     """
#     K(ω) = K_r + K_sup(ω)
#     C(ω) = ω*C_g + C_sup(ω)
#     입력 K_sup, C_sup shape = (n_pop, n_w, n_dof, n_dof)
#     반환 K_sys, C_sys 동일 shape
#     """
#     n_pop, n_w, n_dof, _ = K_sup.shape
#     K_sys = np.zeros_like(K_sup)
#     C_sys = np.zeros_like(C_sup)
#     for j in range(n_pop):
#         for iw, w in enumerate(w_vec):
#             K_sys[j, iw] = mat_K_r + K_sup[j, iw]
#             C_sys[j, iw] = w*mat_C_g + C_sup[j, iw]
#     return K_sys, C_sys

# def _reshape_type(y_type, idx_list):
#     n_k = len(idx_list)
#     return y_type.reshape(n_pop, n_k, 4, n_w)  # (n_pop, n_k, 4, n_w)

# y1 = _reshape_type(y_type1, groups[1])
# y2 = _reshape_type(y_type2, groups[2])
# y3 = _reshape_type(y_type3, groups[3])

# outputs = np.zeros((n_pop, n_seal, 4, n_w), dtype=float)

# for k, i in enumerate(groups[1]):
#     outputs[:, i, :, :] = y1[:, k, :, :]
# for k, i in enumerate(groups[2]):
#     outputs[:, i, :, :] = y2[:, k, :, :]
# for k, i in enumerate(groups[3]):
#     outputs[:, i, :, :] = y3[:, k, :, :]
    
# seal_KkCc_per_speed = []
# for iw in range(n_w):
#     lst = []
#     for i, s in enumerate(seals):
#         # 채널: [C, c, K, k]
#         C_ = rdc_seal*float(outputs[0, i, 0, iw])
#         c_ = rdc_seal*float(outputs[0, i, 1, iw])
#         K_ = rdc_seal*float(outputs[0, i, 2, iw])
#         k_ = rdc_seal*float(outputs[0, i, 3, iw])
#         lst.append({
#             'node': int(s.node - 1),  # xy_dofs가 0-based이므로 -1
#             'K': K_, 'k': k_, 'C': C_, 'c': c_
#         })
#     seal_KkCc_per_speed.append(lst)
    

# brg_nodes = [b.node for b in brgs]

# K_sup, C_sup = assemble_support_mats_per_speed(
#     n_dof=n_dof,
#     w_vec=w_vec,
#     K_brg=rdc_brg*K_brg,          # (n_pop, n_w, 4, n_brg)
#     C_brg=rdc_brg*C_brg,
#     brg_nodes=brg_nodes,  # 각 베어링이 연결된 노드
#     seal_KkCc_per_speed=seal_KkCc_per_speed
# )

# K_sys, C_sys = build_system_mats(
#     mat_K_r=mat_K_r,
#     mat_C_g=mat_C_g,
#     K_sup=K_sup,
#     C_sup=C_sup,
#     w_vec=w_vec
# )

# # K_sys, C_sys = build_system_mats(
# #     mat_K_r=mat_K_r,
# #     mat_C_g=mat_C_g,
# #     K_sup=0*K_sup,
# #     C_sup=0*C_sup,
# #     w_vec=w_vec
# # )

# import numpy as np
# from scipy.linalg import eig

# def modal_scan(mat_M: np.ndarray,
#                K_sys: np.ndarray,
#                C_sys: np.ndarray):

#     n_pop, n_w, n_dof, _ = K_sys.shape
#     I = np.eye(n_dof)
#     Z = np.zeros((n_dof, n_dof))

#     raw_eig = np.empty((n_pop, n_w, n_dof))
#     raw_V   = np.empty((n_pop, n_w, n_dof, n_dof), dtype=np.complex128)

#     for j in range(n_pop):
#         for i in range(n_w):
#             A = np.block([[-C_sys[j, i], -K_sys[j, i]],
#                           [ I          ,  Z           ]])
#             B = np.block([[ mat_M,  Z ],
#                           [ Z    ,  I ]])
#             w, V = eig(A, B)

#             idx_all = np.argsort(np.imag(w))
#             idx_sel = idx_all[-n_dof:]

#             raw_eig[j, i] = np.imag(w[idx_sel])
#             raw_V[j, i]   = V[n_dof:, :][:, idx_sel]
    
#     sigma = np.real(raw_eig)
#     imag_part = np.imag(raw_eig)
#     zeta = -sigma / np.sqrt(sigma**2 + imag_part**2)

#     return raw_eig, raw_V, zeta


# raw_eig, raw_vec, zeta = modal_scan(mat_M, K_sys, C_sys)

# # --- Campbell diagram plotting ---

# def plot_campbell(w_vec: np.ndarray,
#                   raw_eig: np.ndarray,
#                   modes: list[int] | None = None,
#                   orders: tuple[int, ...] = (1, 2, 3),
#                   j: int = 0,
#                   title: str = "Campbell Diagram") -> None:
#     """
#     w_vec: (n_w,) shaft speed [rad/s]
#     raw_eig: (n_pop, n_w, n_dof)  # imag(λ) already, [rad/s]
#     modes: indices of modes to plot along speed; default first min(8, n_dof)
#     orders: excitation line orders (1x, 2x, ...)
#     j: population index to visualize
#     """
#     n_w = w_vec.shape[0]
#     n_dof = raw_eig.shape[2]
#     if modes is None:
#         modes = list(range(min(8, n_dof)))

#     # axes: x = shaft speed [RPM], y = frequency [Hz]
#     rpm = w_vec * 60.0 / (2*np.pi)
#     f_modes = raw_eig[j] / (2*np.pi)  # (n_w, n_dof) in Hz
#     f_shaft = w_vec / (2*np.pi)               # (n_w,) Hz

#     plt.figure()
#     # plot modes
#     for m in modes:
#         plt.plot(rpm, f_modes[:, m])

#     # excitation lines r×
#     # for r in orders:
#     #     plt.plot(rpm, r * f_shaft, linestyle='--', linewidth=1)

#     plt.xlabel('Shaft speed (RPM)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title(title)
#     plt.grid(True, which='both', linestyle=':')

# # draw Campbell
# plot_campbell(w_vec, raw_eig, modes=None, orders=(1,2,3), j=0, title='Campbell Diagram (pop 0)')


# def build_unbalance_force(unb, n_dof, w_vec: np.ndarray):
#     unb_force = np.zeros((n_w, n_dof),dtype=np.complex128)
#     nodes = np.array(unb.cases[0].node, dtype=int)
#     dofs = np.ravel(np.column_stack([4*nodes, 4*nodes+2]))

#     for iw, w in enumerate(w_vec):
#         fvec = np.zeros(n_dof, dtype=np.complex128)
#         f = np.asarray(unb.cases[0].force, dtype=np.complex128)
#         fvec[dofs] = f * (w**2)
#         unb_force[iw] = fvec
#     return unb_force

# unb_force = build_unbalance_force(unb,n_dof,w_vec)


# def forced_response(mat_M: np.ndarray,
#                K_sys: np.ndarray,
#                C_sys: np.ndarray,
#                unb_force: np.ndarray,
#                w_vec: np.ndarray):
#     n_w, n_dof = unb_force.shape
#     resp = np.empty((n_w, n_dof), dtype=np.complex128)
    
#     for i, w in enumerate(w_vec):
#         resp[i] = np.linalg.solve(K_sys[0,i] - w**2*mat_M + 1j*w*C_sys[0,i], unb_force[i])
        
#     return resp

# resp = forced_response(mat_M, K_sys, C_sys, unb_force, w_vec)

# # Plot amplitude at each DOF (e.g., norm of displacement)
# rpm = w_vec * 60.0 / (2*np.pi)
# amplitude = np.abs(resp)


# dof_x = np.arange(0,n_dof,4)
# plt.figure()
# plt.plot(rpm, amplitude[:, dof_x]*1e6, linewidth=1.8)

# # --- Helper: build x/y DOF indices from node indices ---

# def dof_xy_from_nodes(nodes_list):
#     nodes_arr = np.array(list(nodes_list), dtype=int)
#     dof_x = 4*nodes_arr
#     dof_y = 4*nodes_arr + 2
#     return dof_x, dof_y

# # Collect node sets
# brg_nodes_ = [b.node for b in brgs]                      # assumed 0-based (consistent with assembly)
# seal_nodes_ = [s.node - 1 for s in seals]                # seals used 1-based earlier → convert to 0-based
# add_nodes_ = []
# try:
#     add_nodes_ = [e.node for e in added_elements if hasattr(e, 'node')]
# except Exception:
#     pass

# # Convert to DOF indices and clip to valid range
# brg_x, brg_y = dof_xy_from_nodes(brg_nodes_)
# seal_x, seal_y = dof_xy_from_nodes(seal_nodes_)
# add_x,  add_y = dof_xy_from_nodes(add_nodes_)

# valid = np.arange(amplitude.shape[1])
# brg_x = [d for d in brg_x if d in valid]
# brg_y = [d for d in brg_y if d in valid]
# seal_x = [d for d in seal_x if d in valid]
# seal_y = [d for d in seal_y if d in valid]
# add_x  = [d for d in add_x  if d in valid]
# add_y  = [d for d in add_y  if d in valid]

# # --- Plot categorized forced response (μm) ---

# plt.figure()

# # Bearings
# for dof in brg_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'BRG x@{dof}', linewidth=1.8)
# for dof in brg_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'BRG y@{dof}', linestyle='--', linewidth=1.8)
# # Seals
# for dof in seal_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'SEAL x@{dof}', linewidth=1.2)
# for dof in seal_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'SEAL y@{dof}', linestyle='--', linewidth=1.2)
# # Added mass
# for dof in add_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'ADD x@{dof}', linewidth=1.2)
# for dof in add_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'ADD y@{dof}', linestyle='--', linewidth=1.2)


# plt.xlabel('Shaft speed (RPM)')
# plt.ylabel('Amplitude [um]')
# plt.title('Forced Response')
# # plt.legend()
# plt.grid(True)
# plt.show()
    
# # for dof in range(amplitude.shape[1]):
# #     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'DOF {dof}')

# %%
