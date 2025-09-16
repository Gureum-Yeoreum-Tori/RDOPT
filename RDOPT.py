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
from solver_seal import main_seal_solver

from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

n_physical = psutil.cpu_count(logical=False)  
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# ---- Lightweight line-profiler toggle (no external deps) ----
# Enable with env var: RDOPT_PROFILE=1 (optional RDOPT_PROFILE_TOP=N)
PROFILE_EVAL = bool(int(os.getenv("RDOPT_PROFILE", "0")))
PROFILE_EVAL_TOP = int(os.getenv("RDOPT_PROFILE_TOP", "40"))
_PROFILE_EVAL_DONE = False
_BENCH_SEAL_SOLVER_DONE = False
def _profile_lines(func, *args, top=40, **kwargs):
    """Minimal line-level profiler for a single function call.
    Prints top lines by cumulative time within that function only.
    """
    import sys, time as _time, linecache
    code = func.__code__
    filename = code.co_filename
    timings = {}
    last_ts = None
    last_lineno = None
    def local_tracer(frame, event, arg):
        nonlocal last_ts, last_lineno
        if event == 'line':
            now = _time.perf_counter()
            if last_ts is not None and last_lineno is not None:
                timings[last_lineno] = timings.get(last_lineno, 0.0) + (now - last_ts)
            last_ts = now
            last_lineno = frame.f_lineno
        elif event == 'return':
            now = _time.perf_counter()
            if last_ts is not None and last_lineno is not None:
                timings[last_lineno] = timings.get(last_lineno, 0.0) + (now - last_ts)
            return None
        return local_tracer
    def global_tracer(frame, event, arg):
        if event == 'call' and frame.f_code is code:
            return local_tracer
        return None
    old_tracer = sys.getprofile()
    try:
        sys.settrace(global_tracer)
        result = func(*args, **kwargs)
    finally:
        sys.settrace(None)
    # Print report
    items = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top))]
    total = sum(timings.values()) or 1e-12
    print("\n[RDOPT] _evaluate line profile (top {} lines)".format(len(items)))
    for lineno, t in items:
        src = linecache.getline(filename, lineno).rstrip()
        pct = 100.0 * t / total
        print(f"  {os.path.basename(filename)}:{lineno:>4d}  {t:>8.4f}s  ({pct:5.1f}%)  | {src}")
    print(f"[RDOPT] _evaluate total measured time: {total:.4f}s\n")
    return result

## Define constant values
w_range = np.array([500, 7000]) * np.pi / 30
w_oper = 3500 * np.pi / 30
oper = {
        'w_min': w_range[0],
        'w_max': w_range[1],
        'range': w_oper,
    }
bs_params = {
        'mu_brg': 0.025, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }

n_w = 11
n_pop = 200
n_max_gen = 400

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

unique_seals = {s.SealNet for s in seals}
n_type_seal = len(unique_seals)

groups = defaultdict(list)
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
outputs = [None] * len(seals)

idx_seal = [np.array(groups[t+1], dtype=int) for t in range(n_type_seal)]

f_seal_dim = [1e-6, 1e-6, 1e-1]
rdc_signs = np.array([1, 1, -1, 1])

# --- helper for CPU-parallel seal solver benchmark (must be top-level picklable) ---
def _seal_solver_task(task):
    geometry, fluid, op_conditions = task
    try:
        # compute and discard results; timing handled outside
        main_seal_solver(geometry=geometry, fluid=fluid, op_conditions=op_conditions)
        return 0
    except Exception:
        return 1

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

LB_h = 100; UB_h = 500 # seal clearance range
LB_psr = -10;  UB_psr = 10   # -> *0.1 해서 [0,1.0]

## Define optimization problem with vector computation
n_var = 2 * n_brg + 3 * n_seal
n_objs = 6  # [total_leak, power_loss, max_AF, -min_logdec, max_ampRatioBrg, max_ampRatioSeal]
n_constr = 5 # AF, logdec, ampRatio * 2, separation margin

N_FWD_EVAL = 4 # forward 모드 n개만 평가
LOGDEC_MIN = 0.1 # 대수감쇠율 > 0
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
        global _PROFILE_EVAL_DONE
        if PROFILE_EVAL and not _PROFILE_EVAL_DONE:
            _PROFILE_EVAL_DONE = True
            return _profile_lines(self._evaluate_impl, X, out, *args, top=PROFILE_EVAL_TOP, **kwargs)
        return self._evaluate_impl(X, out, *args, **kwargs)
    
    def _evaluate_impl(self, X, out, *args, **kwargs):
        pop = X.shape[0]
        # np.savez_compressed(
        #     'X.npz',
        #     pop_X=X,
        #     # time=time.time(),
        # )
        X_brg = X[:, :n_brg*2].reshape(pop, n_brg, 2)
        X_seal = X[:, n_brg*2:].reshape(pop, n_seal, 3)
        
        F = np.zeros((pop, n_objs), dtype=float) # objective function value
        
        x_brg = X_brg * f_brg_dim
        K_brg, C_brg, loss_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)
        
        seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
        seal_leak = np.zeros((pop, n_seal), dtype=float)
        
        t0 = time.time()
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
        t1 = time.time()
        t_elapsed = t1 - t0
        print(t_elapsed)
        
        # 임시 벤치마크: 한 세대(pop)의 모든 씰에 대해
        # 수치 솔버(main_seal_solver)로 누설량/동특성 예측 시간을 측정하고
        # 위 DeepONet 추론 시간(t_elapsed)과 비교 출력
        try:
            BENCH_ON = bool(int(os.getenv("RDOPT_BENCH_SEAL_SOLVER", "0")))
        except Exception:
            BENCH_ON = True
        global _BENCH_SEAL_SOLVER_DONE
        if BENCH_ON and not _BENCH_SEAL_SOLVER_DONE:
            _BENCH_SEAL_SOLVER_DONE = True  # 한 번만 수행
            t0_solver = time.time()
            NX_SEAL_BENCH = 15  # 빠른 벤치를 위해 비교적 작은 격자
            tasks = []
            for t in range(n_type_seal):
                idx = idx_seal[t]
                if len(idx) == 0:
                    continue
                params_t = X_seal[:, idx]                      # [pop, m, 3]
                x_seal_dim = (params_t.reshape(-1, 3) * f_seal_dim)  # [pop*m, 3]
                m = len(idx)
                for i in range(pop):
                    for j, s_local in enumerate(idx):
                        pos = i * m + j
                        hIn  = float(x_seal_dim[pos, 0])
                        hOut = float(x_seal_dim[pos, 1])
                        psr  = float(x_seal_dim[pos, 2])
                        s_obj = seals[s_local]
                        geometry = {
                            'hIn': hIn,
                            'hOut': hOut,
                            'Ds': float(s_obj.Ds),
                            'Ls': float(s_obj.Ls),
                            'NxSeal': NX_SEAL_BENCH,
                        }
                        fluid = {
                            'mu': float(s_obj.mu),
                            'rho': float(s_obj.rho),
                        }
                        op_conditions = {
                            'dp': float(s_obj.dp),
                            'psr': psr,
                            'w_vec': w_vec,
                        }
                        tasks.append((geometry, fluid, op_conditions))

            # CPU 병렬 실행 옵션
            # n_workers_env = os.getenv("RDOPT_BENCH_NPROC", "")
            n_workers_env = n_physical
            try:
                n_workers = int(n_workers_env) if n_workers_env else (os.cpu_count() or 1)
            except Exception:
                n_workers = os.cpu_count() or 1
            n_workers = max(1, n_workers)
            use_parallel = (os.getenv("RDOPT_BENCH_PARALLEL", "1") != "0") and n_workers > 1

            if use_parallel and len(tasks) > 0:
                try:
                    chunksize = max(1, len(tasks) // (n_workers * 4))
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        for _ in ex.map(_seal_solver_task, tasks, chunksize=chunksize):
                            pass
                except Exception as e:
                    print(f"[seal-timing] parallel failed, fallback to serial: {e}")
                    for task in tasks:
                        _seal_solver_task(task)
            else:
                for task in tasks:
                    _seal_solver_task(task)

            t1_solver = time.time()
            t_solver = t1_solver - t0_solver
            speed_ratio = (t_solver / max(t_elapsed, 1e-12))
            print(f"[seal-timing] DeepONet: {t_elapsed:.4f}s, Solver: {t_solver:.4f}s, x{speed_ratio:.1f} slower")
        
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

        # Constraints g(x) <= 0
        cv_logdec = LOGDEC_MIN - min_logdec
        cv_af     = F[:, 2] - AF_MAX_ALLOW
        cv_brg    = F[:, 4] - RATIO_MAX
        cv_seal   = F[:, 5] - RATIO_MAX

        # Separation margin constraint per API guideline
        # No margin required if AF_at_nearest < 2.5. Otherwise required margin depends on
        # location of nearest critical relative to operating range.
        wmin = w_range[0]
        wmax = w_range[1]
        cv_sm = np.zeros(pop, dtype=float)
        for p in range(pop):
            peaks_w = np.array(peak_centers_list[p], dtype=float)
            peaks_af = np.array(peak_af_list[p], dtype=float)
            if peaks_w.size == 0:
                cv_sm[p] = 0.0  # no detected peaks -> treat as satisfied
                continue
            # Compute separation from operating range for each peak
            # distance outside range (0 if inside)
            dist_below = np.maximum(0.0, wmin - peaks_w)
            dist_above = np.maximum(0.0, peaks_w - wmax)
            dist_to_range = dist_below + dist_above

            # Find nearest peak to the operating range (inside yields distance 0)
            k = int(np.argmin(dist_to_range))
            Nc = float(peaks_w[k])
            AFc = float(peaks_af[k])

            # Measured margin in percent relative to operating speed w_oper
            if Nc < wmin:
                sm_meas = (wmin - Nc) / w_oper * 100.0
                location = 'below'
            elif Nc > wmax:
                sm_meas = (Nc - wmax) / w_oper * 100.0
                location = 'above'
            else:
                sm_meas = 0.0
                location = 'inside'

            # Required margin per AF and location
            if AFc < 2.5:
                sm_req = 0.0
            else:
                base = 17.0 * (1.0 - 1.0 / max(AFc - 1.5, 1e-12))
                if location == 'below':
                    sm_req = max(base, 16.0)
                elif location == 'above':
                    sm_req = max(10.0 + base, 26.0)
                else:  # inside operating range is not allowed unless sm_req == 0
                    # choose more conservative branch (treat as below for requirement), but measured=0 typically fails
                    sm_req = max(base, 16.0)

            cv_sm[p] = sm_req - sm_meas
        out["F"] = F
        out["G"] = np.vstack([cv_logdec, cv_af, cv_brg, cv_seal, cv_sm]).T


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

def debug_cb(algorithm):
    pop = algorithm.pop
    if pop is None:
        print("pop is None"); return
    F = pop.get("F"); G = pop.get("G")
    print(f"gen={algorithm.n_gen}, pop={len(pop)}",
        f"F_shape={None if F is None else F.shape}",
        f"G_shape={None if G is None else G.shape}",
        f"nanF={None if F is None else np.isnan(F).sum()}",
        f"nanG={None if G is None else (None if G is None else np.isnan(G).sum())}")

algorithm.callback = debug_cb
from pymoo.core.termination import TerminateIfAny
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.max_gen import MaximumGenerationTermination

termination = TerminateIfAny(DefaultMultiObjectiveTermination(), MaximumGenerationTermination(n_max_gen))

problem = RotordynamicProblem()
print("Done >.<\n\n\n")
#%%

print("optimization started...")
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

save_path.mkdir(parents=True, exist_ok=True)
np.savez(output_file.with_suffix(".npz"), X=res.X, F=res.F, OPT = res.opt, POP = res.pop)
np.savez(hist_file.with_suffix(".npz"), HISTORY = res.history)


