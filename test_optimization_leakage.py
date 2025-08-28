import os
import numpy as np
from import_data import rotor_import, calculate_bearing_loads
from loader_brg_seal import BearingH5Loader, SealDONModel, SealLeakModel
from collections import defaultdict
from solver_seal import main_seal_solver as seal_solver
import torch
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair


## Import data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

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
w_vec = np.linspace(w_range[0], w_range[1], n_w)

_, _, _, _, _, n_seal, _, _, _, _, _, _, _, _, _, _, _, _, _, seals = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)


## initialize seal
# seal = SealDONModel()
seal_leak = SealLeakModel()

groups = defaultdict(list)          # {SealNet: [원본 인덱스,...]}
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
outputs = [None] * len(seals)  # 원래 순서로 채울 버퍼


## Optimization
n_types = 3
LB_h = 100; UB_h = 500
LB_psr = 0;  UB_psr = 10   # -> *0.1 해서 [0,1.0]

# 선택: 정수 도메인 연산자를 쓰고 싶을 때(권장)
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

n_var = 3 * n_seal

class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # 경계+정수 보정
        X = np.clip(X, problem.xl, problem.xu)
        return np.rint(X).astype(int)

class SealLeakageProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=n_types, n_constr=0, xl=self._xl(), xu=self._xu(), elementwise_evaluation=False)

    def _xl(self):
        # 전 씰 동일 범위라면 반복
        lb = []
        for _ in range(n_seal):
            lb += [LB_h, LB_h, LB_psr]
        return np.array(lb, dtype=int)

    def _xu(self):
        ub = []
        for _ in range(n_seal):
            ub += [UB_h, UB_h, UB_psr]
        return np.array(ub, dtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        # X: [pop, n_var] 정수 행렬
        pop = X.shape[0]

        F = np.zeros((pop, n_types), dtype=float)

        # 배치 변환: 씰별 파라미터를 묶어서 타입별 배치 추론
        # X를 [pop, n_seal, 3]으로 재배열하고 실수/물리단위로 스케일링
        X3 = X.reshape(pop, n_seal, 3).astype(float)
        h_in  = X3[:, :, 0]*1e-6   # um -> m
        h_out = X3[:, :, 1]*1e-6   # um -> m
        psr   = X3[:, :, 2]*1e-1    # -> [0,1.0]

        # 타입별 인덱스
        idx_t = [np.array(groups[t+1], dtype=int) for t in range(n_types)]

        # 타입별로 배치 추론
        for t in range(n_types):
            idx = idx_t[t]
            if len(idx) == 0:
                continue

            # [pop, n_k, 3]
            params_t = np.stack([h_in[:, idx], h_out[:, idx], psr[:, idx]], axis=-1)  # -> [pop, n_k, 3]
            # [pop*n_k, 3]
            x_type = params_t.reshape(-1, 3)

            # 씰 기하/운전 조건은 타입 내에서 동일한 것으로 가정. 각 인덱스별로 다른 Ds,Ls,dp면 루프 누적 방식으로 처리.
            # 여기서는 네가 만든 predict(SealNet, X, w_vec)처럼 타입별 배치 추론이 있다고 가정.
            # 반환이 누설유량 1개라면 y.shape == [pop*n_k] 또는 [pop*n_k, 1] 가 되도록 맞춰라.
            # 예: y = seal_model.predict(t+1, x_type, w_vec)  # 사용 중인 API에 맞게 교체
            # 여기선 예시로 seal_solver를 루프-누적으로 호출하는 느린 버전도 보여준다.

            # --- 빠른 경로: 네 신경망이 타입별 배치 추론을 지원할 때 ---
            y = seal_leak.predict(t+1, x_type)  # [pop*n_k,] 누설유량
            leak = y.reshape(pop, len(idx)).sum(axis=1)  # [pop,]

            F[:, t] = leak

        out["F"] = F

# 정수 도메인용 샘플러/연산자
sampling = IntegerRandomSampling()
crossover = TwoPointCrossover()                   # 정수에서도 적용 가능
mutation = PolynomialMutation(eta=20)             # 이후 Repair로 반올림
repair = RoundRepair()

algorithm = NSGA2(
    pop_size=500,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
    repair=repair
)

termination = get_termination("n_gen", 200)

problem = SealLeakageProblem()

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True
)

X_opt = res.X         # 파레토 해 집합의 설계변수 (정수)
F_opt = res.F         # 각 해의 [L_type1, L_type2, L_type3]
print("pareto solutions:", X_opt.shape, "objectives:", F_opt.shape)

#%%
print(X_opt)
print(F_opt)

l1 = seal_leak.predict(1, X_opt.reshape(-1,3)*[1e-6, 1e-6, 1e-1])
l2 = seal_leak.predict(2, X_opt.reshape(-1,3)*[1e-6, 1e-6, 1e-1])
l3 = seal_leak.predict(3, X_opt.reshape(-1,3)*[1e-6, 1e-6, 1e-1])


geometry = {
    'hIn': 100*1e-6,
    'hOut': 100*1e-6,
    'Ds': seals[0].Ds,
    'Ls': seals[0].Ls,
    'NxSeal': 45,
}

fluid = {
    'mu': bs_params['mu_seal'],
    'rho': bs_params['rho_seal'],   
}

op_conditions = {
    'dp': seals[0].dp,
    'w_vec': 3500*np.pi/30,
    'psr': 1.0,
}

l11, _, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)






# n_pop = 24
# n_types = 3

# h_in  = np.random.randint(100, 500, size=(n_pop, n_seal, 1))*1e-6 # radial clearance
# h_out = np.random.randint(100, 500, size=(n_pop, n_seal, 1))*1e-6 # [m]
# psr   = np.random.randint(-10, 10,   size=(n_pop, n_seal, 1))*1e-1 


# params_per_type = []
# for t in range(n_types):
#     idx = np.array(groups[t+1], dtype=int)
#     n_k = len(idx)
#     # (n_pop, n_k, 3)
#     params_t = np.concatenate([h_in[:, idx, :], h_out[:, idx, :], psr[:, idx, :]], axis=-1)
#     params_per_type.append(params_t)
    
# x_type1 = params_per_type[0].reshape(-1, 3)
# x_type2 = params_per_type[1].reshape(-1, 3)
# x_type3 = params_per_type[2].reshape(-1, 3)


# leak1 = seal_leak.predict(1,x_type1*[1e6, 1e6, 1e1])
# leak2 = seal_leak.predict(2,x_type2*[1e6, 1e6, 1e1])
# leak3 = seal_leak.predict(3,x_type3*[1e6, 1e6, 1e1])

# rdc1 = seal.predict(1,x_type1, w_vec)
# rdc2 = seal.predict(2,x_type2, w_vec)
# rdc3 = seal.predict(3,x_type3, w_vec)

# Leak = np.zeros((len(np.array(groups[1], dtype=int))*n_pop,), dtype=float) 
# RDC = np.zeros((len(np.array(groups[1], dtype=int))*n_pop,4,n_w), dtype=float) 

# idx_seal = 0

# # for i, (hIn, hOut, psr) in enumerate(x_type1):
    
# #     geometry = {
# #     'hIn': hIn*1e-6,
# #     'hOut': hOut*1e-6,
# #     'Ds': seals[idx_seal].Ds,
# #     'Ls': seals[idx_seal].Ls,
# #     'NxSeal': 35,
# #     }

# #     fluid = {
# #         'mu': bs_params['mu_seal'],
# #         'rho': bs_params['rho_seal'],   
# #     }

# #     op_conditions = {
# #         'dp': seals[idx_seal].dp,
# #         'w_vec': w_vec,
# #         'psr': psr*1e-1,
# #     }

# #     Leak_, RDC_, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)
# #     Leak[i] = np.mean(Leak_)
# #     RDC[i] = RDC_[:,2:6].transpose()

# # np.mean((rdc1-RDC)/RDC*100,2)

# # from concurrent.futures import ProcessPoolExecutor, as_completed

# # def _solve_one_case(i, hIn_um, hOut_um, psr_tenth, Ds, Ls, dp, mu, rho, w_vec, nx_seal):
# #     # 단일 케이스 해석 (프로세스용 순수 함수)
# #     from solver_seal import main_seal_solver as seal_solver  # 각 프로세스에서 임포트
# #     geometry = {
# #         'hIn': float(hIn_um),
# #         'hOut': float(hOut_um),
# #         'Ds': float(Ds),
# #         'Ls': float(Ls),
# #         'NxSeal': int(nx_seal),
# #     }
# #     fluid = {
# #         'mu': float(mu),
# #         'rho': float(rho),
# #     }
# #     op_conditions = {
# #         'dp': float(dp),
# #         'w_vec': np.asarray(w_vec, dtype=float),
# #         'psr': float(psr_tenth),
# #     }

# #     Leak_, RDC_, *_ = seal_solver(geometry=geometry, fluid=fluid, op_conditions=op_conditions)
# #     leak_mean = float(np.mean(Leak_))
# #     rdc_slice = RDC_[:, 2:6].T  # shape (4, n_w)
# #     return i, leak_mean, rdc_slice

# # # --- Parallel evaluation for type-1 seals ---
# # # x_type1는 [n_pop * n_k, 3]이며, groups[1]의 인덱스 순서가 n_pop번 반복됨
# # idx_list_type1 = np.array(groups[1], dtype=int)
# # if len(idx_list_type1) == 0:
# #     raise RuntimeError("groups[1]에 해당하는 씰이 없습니다.")

# # args_list = []
# # for i, (hIn, hOut, psr) in enumerate(x_type1):
# #     seal_idx = idx_list_type1[i % len(idx_list_type1)]
# #     s = seals[seal_idx]
# #     args_list.append(
# #         (i, hIn, hOut, psr, s.Ds, s.Ls, s.dp, bs_params['mu_seal'], bs_params['rho_seal'], w_vec, 35)
# #     )
    
# # # 결과 버퍼
# # Leak = np.zeros((len(x_type1),), dtype=float)
# # RDC  = np.zeros((len(x_type1), 4, n_w), dtype=float)

# # # Windows 호환을 위해 __main__ 가드 권장. 이 스크립트가 직접 실행될 때만 병렬 수행.
# # #%%
# # with ProcessPoolExecutor(max_workers=24) as ex:
# #     futures = [ex.submit(_solve_one_case, *a) for a in args_list]
# #     for fut in as_completed(futures):
# #         i, leak_mean, rdc_slice = fut.result()
# #         Leak[i] = leak_mean
# #         RDC[i]  = rdc_slice




# #%%
# # errRDC = np.mean((rdc1-RDC)/(RDC+1e-12)*100,2)
# # errLeak = (leak1.squeeze()-Leak)/(Leak+1e-12)*100
# # plt.subplot(2,2)
# # plt.plot(w_vec*30/np.pi, RDC[0,0])
# # plt.plot(w_vec*30/np.pi, rdc1[0,0])
# # plt.show()


# # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# # axes = axes.flatten()

# # print("Percentage Error Comparison:")
# # for i in range(4):
# #     ax = axes[i]
# #     truth = RDC[0,i]
# #     pred = rdc1[0,i]
    
# #     ax.plot(w_vec*30/np.pi, RDC[0,i])
# #     ax.plot(w_vec*30/np.pi, rdc1[0,i])
# #     # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더함
# #     # percentage_error = np.abs((pred - truth) / (truth + 1e-12)) * 100

# #     # ax.plot(w_vec, percentage_error, 'go-', label='Percentage Error (%)')
# #     # ax.set_title(f"RDC: {rdc_labels[i]}")
# #     # ax.set_xlabel('Rotational Speed (rad/s)')
# #     # ax.set_ylabel('Percentage Error (%)')
# #     # ax.grid(True); ax.legend()
# #     # print(f"  - {rdc_labels[i]:<2}: Mean Percentage Error = {np.mean(percentage_error):.2f}%")

# # plt.tight_layout(rect=[0, 0, 1, 0.96])
# # plt.show()


# # geometry = {
# #     'hIn': x_type1[0,0]*1e-6,
# #     'hOut': x_type1[0,1]*1e-6,
# #     'Ds': seals[0].Ds,
# #     'Ls': seals[0].Ls,
# #     'NxSeal': 35,
# # }

# # fluid = {
# #     'mu': bs_params['mu_seal'],
# #     'rho': bs_params['rho_seal'],   
# # }

# # op_conditions = {
# #     'dp': seals[0].dp,
# #     'w_vec': w_vec,
# #     'psr': x_type1[0,2]*1e-1,
# # }
# # Leak, RDC, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)



# # print(y_type1)
# # print(y_type2)
# # print(y_type3)



# # # brgs = calculate_bearing_loads(
# # #     rotor_elements=rotor_elements,
# # #     brgs=brgs,
# # #     F_mass=F_mass,
# # #     F_ex=F_ex,
# # #     n_brg=n_brg,
# # # )
# # # n_dof= mat_M.shape[0]

# # # matrix_params = {
# # #     'mat_M': mat_M,
# # #     'mat_K_r': mat_K_r,
# # #     'mat_C_g': mat_C_g,
# # #     'n_ele': n_ele,
# # #     'n_node': n_node,
# # #     'n_dof': n_dof,
# # # }

# # # n_w = 14
# # # n_pop = 1
# # # w_vec = np.linspace(w_range[0], w_range[1], n_w)

# # ## predict bearing and seal rdc. single case

# # # brg
# # brg = BearingH5Loader()
# # # brg1_id = np.random.randint(1,55,size=n_pop)[:,None] # bearing_id: 1 ~ 55
# # # brg2_id = np.random.randint(1,55,size=n_pop)[:,None] 
# # # brg1_cr = np.random.randint(10,30,size=n_pop)[:,None] # Cr/D = 10/10000 ~ 30/10000
# # # brg2_cr = np.random.randint(10,30,size=n_pop)[:,None] 
# # # brg_params = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)
# # # brg_params = np.array([[21, 10, 21, 10]])


# # # K_brg = np.zeros([n_pop,n_w,4,2])
# # # C_brg = np.zeros([n_pop,n_w,4,2])
# # # for i in range(len(brgs)):
# # #     for j in range(n_pop):
# # #         K_, C_, _ = brg.calculate_bearing_coefficients(
# # #             brg_params[j,2*(i-1)].astype(int), 
# # #             brgs[i].Db, 
# # #             brgs[i].Db*brg_params[j,2*(i-1)+1]*1e-4, 
# # #             brgs[i].mu, 
# # #             brgs[i].load, 
# # #             w_vec
# # #         )
# # #         K_brg[j,:, :, i] = np.stack([
# # #                             K_[:,0,0],  # Kxx
# # #                             K_[:,0,1],  # Kxy
# # #                             K_[:,1,0],  # Kyx
# # #                             K_[:,1,1],  # Kyy
# # #                         ], axis=1)
# # #         C_brg[j,:, :, i] = np.stack([
# # #                             C_[:,0,0],  # Kxx
# # #                             C_[:,0,1],  # Kxy
# # #                             C_[:,1,0],  # Kyx
# # #                             C_[:,1,1],  # Kyy
# # #                         ], axis=1)

# # # seal
# # seal = SealFNOModel()

# # groups = defaultdict(list)          # {SealNet: [원본 인덱스,...]}
# # for i, s in enumerate(seals):
# #     groups[s.SealNet].append(i)
# # outputs = [None] * len(seals)  # 원래 순서로 채울 버퍼

# # n_types = 3

# # h_in  = np.random.randint(100, 500, size=(n_pop, n_seal, 1)) # radial clearance, 100 um ~ 200 um
# # h_out = np.random.randint(100, 500, size=(n_pop, n_seal, 1))
# # psr   = np.random.randint(1, 10,   size=(n_pop, n_seal, 1))

# # h_in = 100*np.ones((n_pop, n_seal, 1))
# # h_out = 100*np.ones((n_pop, n_seal, 1))
# # psr = 0*np.ones((n_pop, n_seal, 1))

# # params_per_type = []
# # for t in range(n_types):
# #     idx = np.array(groups[t+1], dtype=int)
# #     n_k = len(idx)
# #     # (n_pop, n_k, 3)
# #     params_t = np.concatenate([h_in[:, idx, :]*1e-6, h_out[:, idx, :]*1e-6, psr[:, idx, :]*1e-1], axis=-1)
# #     params_per_type.append(params_t)
    
# # x_type1 = params_per_type[0].reshape(-1, 3)
# # x_type2 = params_per_type[1].reshape(-1, 3)
# # x_type3 = params_per_type[2].reshape(-1, 3)
    
# # # y_type1 = seal.predict(1,x_type1,w_vec)
# # # y_type2 = seal.predict(2,x_type2,w_vec)
# # # y_type3 = seal.predict(3,x_type3,w_vec)


# geometry = {
#     'hIn': 100*1e-6,
#     'hOut': 100*1e-6,
#     'Ds': seals[0].Ds,
#     'Ls': seals[0].Ls,
#     'NxSeal': 25,
# }

# fluid = {
#     'mu': bs_params['mu_seal'],
#     'rho': bs_params['rho_seal'],   
# }

# op_conditions = {
#     'dp': seals[0].dp,
#     'w_vec': w_vec,
#     'psr': -1.0,
# }

# Leak, RDC, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)


# l1 = seal_leak.predict(1, [100, 100, -10])
# # # pip install -U pymoo
# # import numpy as np


# --- 1) 파레토 프론트: 목적 2개일 때 ---
def plot_pareto_2obj(F, xlabel='L_type1', ylabel='L_type2', log=False):
    x, y = F[:,0], F[:,1]
    plt.figure(figsize=(5,4))
    plt.scatter(x, y, s=12, alpha=0.7)
    if log:
        plt.xscale('log'); plt.yscale('log')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title('Pareto Front (2 objectives)')
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.tight_layout(); plt.show()

# --- 2) 파레토 프론트: 목적 3개일 때(3D 산점) ---
def plot_pareto_3obj(F, labels=('L_type1','L_type2','L_type3'), log=False):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = F[:,0], F[:,1], F[:,2]
    ax.scatter(x, y, z, s=10, alpha=0.7)
    if log:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_zscale('log')
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_title('Pareto Front (3 objectives)')
    plt.tight_layout(); plt.show()

# --- 3) 목적 3개 이상: 평행좌표 플롯 ---
def plot_parallel_coords(F, labels=None, log=False):
    import pandas as pd
    from pandas.plotting import parallel_coordinates
    # 간단히 상위 n개만 보기 원하면 여기서 정렬/슬라이스 가능
    df = pd.DataFrame(F, columns=labels if labels else [f'f{i+1}' for i in range(F.shape[1])])
    if log:
        df = np.log10(df)
    # 클래스 컬럼이 필요하므로 더미 추가
    df['cls'] = 'front'
    plt.figure(figsize=(8,4))
    parallel_coordinates(df, 'cls', color=['C0'], alpha=0.6)
    plt.title('Parallel Coordinates (Objectives)')
    plt.legend().remove()
    plt.tight_layout(); plt.show()

# --- 4) 세대별 하이퍼볼륨(히스토리 저장했을 때) ---
def plot_hypervolume(history):
    # minimize(..., save_history=True) 로 실행했을 때만 가능
    from pymoo.indicators.hv import HV
    # 참조점은 문제 스케일에 맞게 설정
    # 예: 모든 목적이 양수고 대략 이 정도 범위를 넘지 않는다고 가정
    ref = np.max(np.vstack([h.opt.get("F") for h in history]), axis=0) * 1.05
    hv = HV(ref_point=ref)
    hv_list = [hv.do(h.opt.get("F")) for h in history]
    plt.figure(figsize=(5,3.5))
    plt.plot(hv_list, marker='o', ms=3)
    plt.xlabel('Generation'); plt.ylabel('Hypervolume')
    plt.title('Hypervolume over Generations')
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.tight_layout(); plt.show()


plot_hypervolume(res.history)
