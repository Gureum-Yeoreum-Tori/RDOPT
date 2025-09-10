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
# from pymoo.util.display.output import Output
# from pymoo.util.display.column import Column

# import pickle

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

## Define constant values
w_range = np.array([500, 7000]) * np.pi / 30
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

n_w = 7
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
from datetime import datetime
from pathlib import Path

date_now   = datetime.now().strftime("%y%m%d_T_%H%M%S")
res_       = "rotordyn"
save_path  = Path("result") / res_ / rotor_sheet / date_now
save_path.mkdir(parents=True, exist_ok=True)

output_file = save_path / "result"
hist_file = save_path / "result_hist"
print("Result file path = ",save_path," \n")


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
n_constr = 4 # AF, logdec, ampRatio * 2

N_FWD_EVAL = 4 # forward 모드 n개만 평가
LOGDEC_MIN = 0.0 # 대수감쇠율 > 0
AF_MAX_ALLOW = 8.0
RATIO_MAX = 75.0

# class Output_re(Output):
#     def __init__(self):
#         super().__init__()
#         # self.x_mean = Column("x_mean", width=14)
#         # self.x_std = Column("x_std", width=14)
#         self.col_nds = Column("n_gen", width=8)
#         self.col_nds = Column("n_nds", width=8)
#         self.cur_time  = Column("t_cur_gen [s]", width=14)
#         self.elapsed_time  = Column("t_elapsed [s]", width=14)
        
#         self.columns += [self.x_mean, self.x_std, self.cur_time, self.elapsed_time]

#         self.start_time = time.time()
#         self.prev_time  = self.start_time

#     def update(self, algorithm):
#         super().update(algorithm)
        
#         now = time.time()
#         dt_gen   = now - self.prev_time
#         t_total  = now - self.start_time
#         self.prev_time = now
        
#         X = algorithm.pop.get("X")
        
#         self.x_mean.set(f"{np.mean(X):>14.6E}")
#         self.x_std.set( f"{np.std(X):>14.6E}")
#         self.cur_time.set( f"{dt_gen:>14.2f}")
#         self.elapsed_time.set( f"{t_total:>14.2f}")
        
        
class TimerCheckpointCallback(Callback):
    def __init__(self, out_dir: str = "checkpoints", freq: int = 5, save_pickle: bool = True):
        super().__init__()
        self.prev_time = None
        self.out_dir = out_dir
        self.freq = max(1, int(freq))
        self.save_pickle = save_pickle
        os.makedirs(self.out_dir, exist_ok=True)

    def _save_npz(self, algorithm):
        try:
            gen = int(getattr(algorithm, "n_gen", -1))
            pop_X = algorithm.pop.get("X") if algorithm.pop is not None else None
            pop_F = algorithm.pop.get("F") if algorithm.pop is not None else None
            pop_CV = algorithm.pop.get("CV") if algorithm.pop is not None else None
            opt_X = algorithm.opt.get("X") if getattr(algorithm, "opt", None) is not None else None
            opt_F = algorithm.opt.get("F") if getattr(algorithm, "opt", None) is not None else None
            path = os.path.join(self.out_dir, f"gen_{gen:04d}.npz")
            np.savez_compressed(
                path,
                gen=gen,
                pop_X=pop_X,
                pop_F=pop_F,
                pop_CV=pop_CV,
                opt_X=opt_X,
                opt_F=opt_F,
                # time=time.time(),
            )
            # also save/overwrite a quick pointer to the latest state
            latest = os.path.join(self.out_dir, "latest.npz")
            np.savez_compressed(
                latest,
                gen=gen,
                pop_X=pop_X,
                pop_F=pop_F,
                pop_CV=pop_CV,
                opt_X=opt_X,
                opt_F=opt_F,
                # time=time.time(),
            )
        except Exception as e:
            print(f"[checkpoint] Failed to save npz: {e}")

    # def _save_pickle(self, algorithm):
    #     if not self.save_pickle:
    #         return
    #     try:
    #         gen = int(getattr(algorithm, "n_gen", -1))
    #         path = os.path.join(self.out_dir, f"algorithm_gen_{gen:04d}.pkl")
    #         with open(path, "wb") as f:
    #             pickle.dump(algorithm, f)
    #     except Exception as e:
    #         # Some algorithm objects may not be fully picklable; ignore failures.
    #         print(f"[checkpoint] Pickle save skipped/failed: {e}")

    def notify(self, algorithm):
        # timing log
        now = time.time()
        if self.prev_time is not None:
            elapsed = now - self.prev_time
            print(f"Gen {algorithm.n_gen} elapsed: {elapsed:.3f} sec")
        self.prev_time = now

        # periodic checkpoint
        gen = int(getattr(algorithm, "n_gen", 0) or 0)
        if gen % self.freq == 0:
            self._save_npz(algorithm)
            # self._save_pickle(algorithm)
        
class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 경계+정수 보정
        X = np.clip(X, problem.xl, problem.xu)
        return np.rint(X).astype(int)

class RotordynamicProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=n_objs, n_constr=n_constr, xl=self._xl(), xu=self._xu(), elementwise_evaluation=False)

    def _xl(self):
        lb = []
        for _ in range(n_brg):
            lb += [LB_brg_idx, LB_cr]
        # for _ in range(n_seal):
        #     lb += [LB_h, LB_h, LB_psr]
        for _, s in enumerate(seals):
            lb += [LB_h, LB_h, LB_psr - LB_psr*s.is_bush]
        return np.array(lb, dtype=int)

    def _xu(self):
        ub = []
        for _ in range(n_brg):
            ub += [UB_brg_idx, UB_cr]
        # for _ in range(n_seal):
        #     ub += [UB_h, UB_h, UB_psr]
        for _, s in enumerate(seals):
            ub += [UB_h, UB_h, UB_psr - UB_psr*s.is_bush]
        return np.array(ub, dtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        pop = X.shape[0]
        X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
        X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)
        
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
            forward=True,
        )
        
        harmonic = unbalanced_response(
            M=mat_M,
            unb=unb,
            K_all=K_all,
            Ceff_all=Ceff_all,
            w_vec=w_vec,
        )
        
        idx_op = int(np.argmin(np.abs(w_vec - w_oper)))
        F[:, 0] = seal_leak.sum(axis=1)
        F[:, 1] = loss_brg[:,:,:,idx_op].sum(axis=1).squeeze()

        assert n_dof == 4 * n_node
        idx_x = np.arange(n_node) * 4
        idx_y = idx_x + 2
        Ux = harmonic[:, :, idx_x, 0]  # [pop, n_w, n_node]
        Uy = harmonic[:, :, idx_y, 0]
        amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)  # [pop, n_w, n_node]

        brg_nodes = np.array([b.node for b in brgs], dtype=int)
        seal_nodes = np.array([s.node for s in seals], dtype=int)
        cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))

        eps = 1e-18
        AF_max = np.zeros(pop, dtype=float)
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

        # Constraints g(x) <= 0
        cv_logdec = LOGDEC_MIN - min_logdec
        cv_af     = F[:, 2] - AF_MAX_ALLOW
        cv_brg    = F[:, 4] - RATIO_MAX
        cv_seal   = F[:, 5] - RATIO_MAX
        out["F"] = F
        out["G"] = np.vstack([cv_logdec, cv_af, cv_brg, cv_seal]).T


## Start optimization
sampling = IntegerRandomSampling()
crossover = TwoPointCrossover()
mutation = PolynomialMutation(eta=20)
repair = RoundRepair()

if  which_algorithm == 'NSGA2':
    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        repair=repair
    )
elif which_algorithm == 'UNSGA3':
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("energy", n_dim=n_objs, n_points=n_pop, seed=42)

    algorithm = UNSGA3(
        ref_dirs=ref_dirs,
        pop_size=n_pop,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        repair=repair
    )
else:
    print("optimization algorithm not defined")
    raise ValueError()


from pymoo.core.termination import TerminateIfAny
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.max_gen import MaximumGenerationTermination

termination = TerminateIfAny(DefaultMultiObjectiveTermination(), MaximumGenerationTermination(n_max_gen))

problem = RotordynamicProblem()
print("Done >.<\n\n\n")
#%%

print("Starting optimization...")
t_start = time.time()

res = minimize(
    problem,
    algorithm,
    termination,
    # output=Output_re(),
    seed=42,
    save_history=True,
    verbose=True,
    callback=TimerCheckpointCallback(),
)

t_end = time.time()
t_elapsed = t_end - t_start
print("Optimization completed!")
print("elapsed time =",f"{t_elapsed:.2f}","sec\n")


np.savez(output_file.with_suffix(".npz"), X=res.X, F=res.F, OPT = res.opt, POP = res.pop)
np.savez(hist_file.with_suffix(".npz"), HISTORY = res.history)




# with open(output_file.with_suffix(".pkl"), "wb") as f:
#     pickle.dump(res, f)


#%%
# d = np.load('checkpoints/latest.npz')
# X_pop, F_pop = d['pop_X'], d['pop_F']
# X_pareto, F_pareto = d['opt_X'], d['opt_F']








#%%
# # gpu test
# t_start = time.time()

# X = np.array([np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)]), 
# np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)])])

# X = np.repeat(X, 200, axis=0)

# pop = X.shape[0]

# X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
# X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

# F = np.zeros((pop, n_objs), dtype=float) # objective function value

# # x_brg = X_brg * f_brg_dim
# # K_brg, C_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)
# device_test = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# model_seal = SealDONModel(device=device_test)
# model_seal_leak = SealLeakModel(device=device_test)

# seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
# seal_leak = np.zeros((pop, n_seal), dtype=float)
# t0 = time.time()
# for t in range(n_type_seal):
#     idx = idx_seal[t]
#     if len(idx) == 0:
#         continue
    
#     params_t = X_seal[:, idx]
#     x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    
#     leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))        # [pop, m]
#     rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)  # [pop, m, 4, n_w]
    
#     seal_leak[:, idx] = leak_flat
#     seal_rdc[:, idx] = rdc_flat
# t1 = time.time()
# elapsed = t1 - t0
# print(f"{elapsed:.2f} sec elapsed using",device_test)




# device_test = 'cpu'
# model_seal = SealDONModel(device=device_test)
# model_seal_leak = SealLeakModel(device=device_test)

# seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
# seal_leak = np.zeros((pop, n_seal), dtype=float)
# t0 = time.time()
# for t in range(n_type_seal):
#     idx = idx_seal[t]
#     if len(idx) == 0:
#         continue
    
#     params_t = X_seal[:, idx]
#     x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    
#     leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))        # [pop, m]
#     rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)  # [pop, m, 4, n_w]
    
#     seal_leak[:, idx] = leak_flat
#     seal_rdc[:, idx] = rdc_flat
# t1 = time.time()
# elapsed = t1 - t0
# print(f"{elapsed:.2f} sec elapsed using",device_test)



#%%
# # code for test
# t_start = time.time()

# X = np.array([np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)]), 
# np.concatenate([np.tile([21, 15],2), np.tile([200, 200, 0], n_seal)])])

# pop = X.shape[0]

# X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
# X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)

# F = np.zeros((pop, n_objs), dtype=float) # objective function value

# x_brg = X_brg * f_brg_dim
# K_brg, C_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

# seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
# seal_leak = np.zeros((pop, n_seal), dtype=float)

# for t in range(n_type_seal):
#     idx = idx_seal[t]
#     if len(idx) == 0:
#         continue
    
#     params_t = X_seal[:, idx]
#     x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    
#     leak_flat = model_seal_leak.predict(t+1, x_seal).reshape(pop, len(idx))        # [pop, m]
#     rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)  # [pop, m, 4, n_w]
    
#     seal_leak[:, idx] = leak_flat
#     seal_rdc[:, idx] = rdc_flat

# K_seal = seal_rdc[:,:,[2, 3, 3, 2],:] * rdc_signs[None, None, :, None]
# C_seal = seal_rdc[:,:,[0, 1, 1, 0],:] * rdc_signs[None, None, :, None]

# K_vals = np.concatenate([K_brg, K_seal], axis=1)  # (pop, n_sup, 4, n_w)
# C_vals = np.concatenate([C_brg, C_seal], axis=1)  # (pop, n_sup, 4, n_w)

# K_all, Ceff_all = assemble_system_matrix(
#     mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals
# )

# eigvals = eig(
#     M=mat_M,
#     K_all=K_all,
#     Ceff_all=Ceff_all,
# )

# harmonic = unbalanced_response(
#     M=mat_M,
#     unb=unb,
#     K_all=K_all,
#     Ceff_all=Ceff_all,
#     w_vec=w_vec,
# )

# # calculate objective functions
# # 1) total seal leakage
# # 2) maximum amplification factor (half-power method per node/peak)
# # 3) -min log decrement across first 8 forward modes over all speeds
# # 4) max bearing amplitude ratio (% of clearance)
# # 5) max seal amplitude ratio (% of min clearance)

# # Operating index
# idx_op = int(np.argmin(np.abs(w_vec - w_oper)))

# # 1) Total seal leakage per individual (sum across all seals)
# F[:, 0] = seal_leak.sum(axis=1)

# # Node-wise amplitudes from harmonic response
# # harmonic: [pop, n_w, n_dof, 1] complex
# assert n_dof == 4 * n_node
# idx_x = np.arange(n_node) * 4
# idx_y = idx_x + 2
# Ux = harmonic[:, :, idx_x, 0]  # [pop, n_w, n_node]
# Uy = harmonic[:, :, idx_y, 0]
# amp = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)  # [pop, n_w, n_node]

# # 2) Amplification Factor using half-power bandwidth: AF = Nc / (N2 - N1)
# # Use calibration nodes (bearings + seals). If empty, fallback to all nodes.
# brg_nodes = np.array([b.node for b in brgs], dtype=int)
# seal_nodes = np.array([s.node for s in seals], dtype=int)
# cal_nodes = np.unique(np.concatenate([brg_nodes, seal_nodes]))

# eps = 1e-18
# AF_max = np.zeros(pop, dtype=float)
# w = w_vec
# for p in range(pop):
#     af_p = 0.0
#     A = amp[p, :, cal_nodes]  # [n_w, n_cal]
#     for c in range(A.shape[1]):
#         y = A[c, :]
#         if y.size < 3:
#             continue
#         # SciPy peak detection (indices)
#         pk, _ = find_peaks(y)
#         if pk.size == 0:
#             continue
#         for j in pk:
#             Ac = y[j]
#             if Ac <= eps:
#                 continue
#             yhpp = Ac / np.sqrt(2.0)
#             # Left half-power crossing
#             if j == 0:
#                 N1 = w[0]
#             else:
#                 l_idx = np.where(y[:j] <= yhpp)[0]
#                 if l_idx.size == 0:
#                     N1 = w[0]
#                 else:
#                     i0 = l_idx[-1]
#                     x0, y0 = w[i0], y[i0]
#                     N1 = x0 + (yhpp - y0) / max(Ac - y0, eps) * (w[j] - x0)
#             # Right half-power crossing
#             if j >= y.size - 1:
#                 N2 = w[-1]
#             else:
#                 r_idx_rel = np.where(y[j+1:] <= yhpp)[0]
#                 if r_idx_rel.size == 0:
#                     N2 = w[-1]
#                 else:
#                     i1 = j + 1 + r_idx_rel[0]
#                     x1, y1 = w[i1], y[i1]
#                     N2 = x1 - (y1 - yhpp) / max(y1 - Ac, eps) * (x1 - w[j])
#             af_peak = w[j] / max(N2 - N1, eps)
#             if af_peak > af_p:
#                 af_p = af_peak
#     AF_max[p] = af_p
# F[:, 1] = AF_max

# # 3) -min log decrement over first 8 forward modes
# alpha = np.real(eigvals)  # [pop, n_w, 2n]
# beta  = np.imag(eigvals)
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
# F[:, 2] = -min_logdec

# # 4) Bearing amplitude ratio (%): amp(node_brg,:)/Cp * 100 -> global max
# brg_nodes = np.array([b.node for b in brgs], dtype=int)
# if brg_nodes.size > 0:
#     brg_ids = X_brg[:, :, 0].astype(int)                         # [pop, n_brg]
#     cr_ratio = (X_brg[:, :, 1] * f_brg_dim[0, 1]).astype(float)  # scaled
#     Db_all = np.array([b.Db for b in brgs], dtype=float)[None, :]
#     Cr = Db_all * cr_ratio                                       # [pop, n_brg]
#     # Vectorized Mp lookup
#     # Build lookup array for Mp where index = bearing_id
#     max_id = int(np.max(brg_ids)) if brg_ids.size else 0
#     mp_lookup = np.zeros(max(UB_brg_idx + 1, max_id + 1), dtype=float)
#     for bid in range(1, mp_lookup.size):
#         try:
#             mp_lookup[bid] = float(model_brg.get_bearing_by_id(bid)['Mp'])
#         except Exception:
#             mp_lookup[bid] = 0.0
#     Mp = mp_lookup[brg_ids]
#     Cp = Cr / np.clip(1.0 - Mp, 1e-9, None)                     # [pop, n_brg]
#     amp_brg = amp[:, :, brg_nodes]                               # [pop, n_w, n_brg]
#     amp_ratio_brg = (amp_brg / (Cp[:, None, :] + eps)) * 100.0
#     F[:, 3] = amp_ratio_brg.reshape(pop, -1).max(axis=1)
# else:
#     F[:, 3] = 0.0

# # 5) Seal amplitude ratio (%): amp(node_seal,:)/min(h_in,h_out) * 100 -> global max
# if seal_nodes.size > 0:
#     h_in  = (X_seal[:, :, 0].astype(float) * f_seal_dim[0])   # [pop, n_seal]
#     h_out = (X_seal[:, :, 1].astype(float) * f_seal_dim[1])   # [pop, n_seal]
#     h_min = np.minimum(h_in, h_out)
#     amp_seal = amp[:, :, seal_nodes]                          # [pop, n_w, n_seal]
#     amp_ratio_seal = (amp_seal / (h_min[:, None, :] + eps)) * 100.0
#     F[:, 4] = amp_ratio_seal.reshape(pop, -1).max(axis=1)
# else:
#     F[:, 4] = 0.0

# t_end = time.time()
# t_elapsed = t_end - t_start
# print(f"elapsed time: ",t_elapsed)

#%%
