import os
import numpy as np
from import_data import rotor_import, calculate_bearing_loads
from loader_brg_seal import BearingH5Loader, SealFNOModel
from collections import defaultdict
from solver_seal import main_seal_solver as seal_solver
# from solver_rotordyn import BrgModel, SealModel
# from torch.linalg import inv
import torch
import matplotlib.pyplot as plt

## Import data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

# w_range = np.array([1500, 5000]) * np.pi / 30
# w_oper = 3500 * np.pi / 30
# oper = {
#         'w_min': w_range[0],
#         'w_max': w_range[1],
#         'range': w_oper,
#     }
bs_params = {
        'mu_brg': 0.04, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }
# rdc_seal = 0
# rdc_brg = 1

_, _, _, _, n_seal, _, _, _, _, _, _, _, _, _, _, _, _, _, seals = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

# brgs = calculate_bearing_loads(
#     rotor_elements=rotor_elements,
#     brgs=brgs,
#     F_mass=F_mass,
#     F_ex=F_ex,
#     n_brg=n_brg,
# )
# n_dof= mat_M.shape[0]

# matrix_params = {
#     'mat_M': mat_M,
#     'mat_K_r': mat_K_r,
#     'mat_C_g': mat_C_g,
#     'n_ele': n_ele,
#     'n_node': n_node,
#     'n_dof': n_dof,
# }

# n_w = 14
# n_pop = 1
# w_vec = np.linspace(w_range[0], w_range[1], n_w)

## predict bearing and seal rdc. single case

# brg
# brg = BearingH5Loader()
# brg1_id = np.random.randint(1,55,size=n_pop)[:,None] # bearing_id: 1 ~ 55
# brg2_id = np.random.randint(1,55,size=n_pop)[:,None] 
# brg1_cr = np.random.randint(10,30,size=n_pop)[:,None] # Cr/D = 10/10000 ~ 30/10000
# brg2_cr = np.random.randint(10,30,size=n_pop)[:,None] 
# brg_params = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)
# brg_params = np.array([[21, 10, 21, 10]])


# K_brg = np.zeros([n_pop,n_w,4,2])
# C_brg = np.zeros([n_pop,n_w,4,2])
# for i in range(len(brgs)):
#     for j in range(n_pop):
#         K_, C_, _ = brg.calculate_bearing_coefficients(
#             brg_params[j,2*(i-1)].astype(int), 
#             brgs[i].Db, 
#             brgs[i].Db*brg_params[j,2*(i-1)+1]*1e-4, 
#             brgs[i].mu, 
#             brgs[i].load, 
#             w_vec
#         )
#         K_brg[j,:, :, i] = np.stack([
#                             K_[:,0,0],  # Kxx
#                             K_[:,0,1],  # Kxy
#                             K_[:,1,0],  # Kyx
#                             K_[:,1,1],  # Kyy
#                         ], axis=1)
#         C_brg[j,:, :, i] = np.stack([
#                             C_[:,0,0],  # Kxx
#                             C_[:,0,1],  # Kxy
#                             C_[:,1,0],  # Kyx
#                             C_[:,1,1],  # Kyy
#                         ], axis=1)

# seal
seal = SealFNOModel()

groups = defaultdict(list)          # {SealNet: [원본 인덱스,...]}
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
outputs = [None] * len(seals)  # 원래 순서로 채울 버퍼

n_types = 3

h_in  = np.random.randint(100, 500, size=(n_pop, n_seal, 1)) # radial clearance, 100 um ~ 200 um
h_out = np.random.randint(100, 500, size=(n_pop, n_seal, 1))
psr   = np.random.randint(1, 10,   size=(n_pop, n_seal, 1))

h_in = 100*np.ones((n_pop, n_seal, 1))
h_out = 100*np.ones((n_pop, n_seal, 1))
psr = 0*np.ones((n_pop, n_seal, 1))

params_per_type = []
for t in range(n_types):
    idx = np.array(groups[t+1], dtype=int)
    n_k = len(idx)
    # (n_pop, n_k, 3)
    params_t = np.concatenate([h_in[:, idx, :]*1e-6, h_out[:, idx, :]*1e-6, psr[:, idx, :]*1e-1], axis=-1)
    params_per_type.append(params_t)
    
x_type1 = params_per_type[0].reshape(-1, 3)
x_type2 = params_per_type[1].reshape(-1, 3)
x_type3 = params_per_type[2].reshape(-1, 3)
    
# y_type1 = seal.predict(1,x_type1,w_vec)
# y_type2 = seal.predict(2,x_type2,w_vec)
# y_type3 = seal.predict(3,x_type3,w_vec)


# geometry = {
#     'hIn': x_type1[0,0],
#     'hOut': x_type1[0,1],
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
#     'psr': x_type1[0,2],
# }
# _, RDC, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)

# pip install -U pymoo
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair

# ---- 너의 환경에서 이미 준비된 것들 가정 ----
# seals: rotor_import로 얻은 리스트, 각 원소에 SealNet(1~3), Ds, Ls, dp 등이 있음
# groups: {1:[idx...], 2:[idx...], 3:[idx...]} 로 타입별 인덱스 목록
# seal_model: SealFNOModel() 또는 seal_solver로 누설유량을 계산할 함수
# w_vec: 고정된 회전속도 벡터 (신경망/해석기 입력에 필요하면 유지, 아니면 제거)

ts_model = torch.jit.load("net/mlp_leak_20250825_T_120952.pt")




seal_model = SealFNOModel()  # 네가 쓰는 추론기
n_types = 3
n_seal = len(seals)

# 변수는 각 씰당 [hIn(um), hOut(um), psr*10] 형태로 코딩해서 전부 정수로 최적화
# 내부 계산 직전에 m 단위/실수로 변환
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
        # 타입별 누설 합계
        F = np.zeros((pop, n_types), dtype=float)

        # 배치 변환: 씰별 파라미터를 묶어서 타입별 배치 추론
        # X를 [pop, n_seal, 3]으로 재배열하고 실수/물리단위로 스케일링
        X3 = X.reshape(pop, n_seal, 3).astype(float)
        h_in  = X3[:, :, 0] * 1e-6   # -> m
        h_out = X3[:, :, 1] * 1e-6   # -> m
        psr   = X3[:, :, 2] * 1e-1   # -> [0,1.0]

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
            # y = seal_model.predict(t+1, x_type, w_vec)  # [pop*n_k,] 누설유량
            # leak = y.reshape(pop, len(idx)).sum(axis=1)  # [pop,]

            # --- 보수적 경로: 해석기/신경망이 씰별 개별 호출만 될 때 누적 ---
            leak = np.zeros((pop,), dtype=float)
            for j_local, j in enumerate(idx.tolist()):
                Ds = seals[j].Ds
                Ls = seals[j].Ls
                dp = seals[j].dp
                for p in range(pop):
                    geometry = {'hIn':   params_t[p, j_local, 0],
                                'hOut':  params_t[p, j_local, 1],
                                'Ds':    Ds,
                                'Ls':    Ls,
                                'NxSeal': 25}
                    fluid    = {'mu': 1.4e-3, 'rho': 850.0}  # bs_params에서 가져다 써도 됨
                    op_cond  = {'dp': dp, 'w_vec': w_vec, 'psr': params_t[p, j_local, 2]}
                    # seal_solver는 여러 값을 돌려주면 누설유량만 취해라. 아래는 예시 이름.
                    Leak, *_ = seal_solver(geometry=geometry, fluid=fluid, op_conditions=op_cond)
                    # Leak이 스칼라면 그대로, 벡터면 합/평균 선택
                    leak[p] += float(np.sum(Leak))  # 정책에 맞게 합 또는 평균

            F[:, t] = leak

        out["F"] = F

# 정수 도메인용 샘플러/연산자
sampling = IntegerRandomSampling()
crossover = TwoPointCrossover()                   # 정수에서도 적용 가능
mutation = PolynomialMutation(eta=20)             # 이후 Repair로 반올림
repair = RoundRepair()

algorithm = NSGA2(
    pop_size=80,
    sampling=sampling,
    crossover=crossover,
    mutation=mutation,
    eliminate_duplicates=True,
    repair=repair
)

termination = get_termination("n_gen", 150)

problem = SealLeakageProblem()

res = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    save_history=False,
    verbose=True
)

X_opt = res.X         # 파레토 해 집합의 설계변수 (정수)
F_opt = res.F         # 각 해의 [L_type1, L_type2, L_type3]
print("pareto solutions:", X_opt.shape, "objectives:", F_opt.shape)