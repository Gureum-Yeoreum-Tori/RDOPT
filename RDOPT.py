import os
import numpy as np
from import_data import rotor_import, calculate_bearing_loads
from loader_brg_seal import BearingH5Loader, SealFNOModel
from collections import defaultdict
from solver_seal import main_seal_solver as seal_solver
# from solver_rotordyn import BrgModel, SealModel
# from torch.linalg import inv
import torch


## Import data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

w_range = np.array([1500, 5000]) * np.pi / 30
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

n_ele, n_node, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_Ks, mat_Cg, mat_MR, mat_MA, F_mass, F_ex, unb, brgs, seals = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

brgs = calculate_bearing_loads(
    rotor_elements=rotor_elements,
    brgs=brgs,
    F_mass=F_mass,
    F_ex=F_ex,
    n_brg=n_brg,
)
n_dof= mat_M.shape[0]

matrix_params = {
    'mat_M': mat_M,
    'mat_Ks': mat_Ks,
    'mat_Cg': mat_Cg,
    'n_ele': n_ele,
    'n_node': n_node,
    'n_dof': n_dof,
}

n_w = 14
w_vec = np.linspace(w_range[0], w_range[1], n_w)

## predict bearing and seal rdc. single case

brg = BearingH5Loader()
seal = SealFNOModel()


n_pop = 1

brg1_id = np.random.randint(1,55,size=n_pop)[:,None] # bearing_id: 1 ~ 55
brg2_id = np.random.randint(1,55,size=n_pop)[:,None] 
brg1_cr = np.random.randint(10,30,size=n_pop)[:,None] # Cr/D = 10/10000 ~ 30/10000
brg2_cr = np.random.randint(10,30,size=n_pop)[:,None] 
brg_params = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)

K_brg = np.zeros([n_pop,n_w,4,2])
C_brg = np.zeros([n_pop,n_w,4,2])
for i in range(len(brgs)):
    for j in range(n_pop):
        K_, C_, _ = brg.calculate_bearing_coefficients(
            brg_params[j,2*(i-1)].astype(int), 
            brgs[i].Db, 
            brgs[i].Db*brg_params[j,2*(i-1)+1]*1e-4, 
            brgs[i].mu, 
            brgs[i].load, 
            w_vec
        )
        K_brg[j,:, :, i] = np.stack([
                            K_[:,0,0],  # Kxx
                            K_[:,0,1],  # Kxy
                            K_[:,1,0],  # Kyx
                            K_[:,1,1],  # Kyy
                        ], axis=1)
        C_brg[j,:, :, i] = np.stack([
                            C_[:,0,0],  # Kxx
                            C_[:,0,1],  # Kxy
                            C_[:,1,0],  # Kyx
                            C_[:,1,1],  # Kyy
                        ], axis=1)


h_in = np.random.randint(100,500,size=n_pop*n_seal)[:, None]
h_out = np.random.randint(100,500,size=n_pop*n_seal)[:, None]
psr = np.random.randint(0,10,size=n_pop*n_seal)[:, None]

seal_params = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)



















# lb = [100 100 0]; % sealNet data range
# ub = [500 500 10];

groups = defaultdict(list)          # {SealNet: [원본 인덱스,...]}
for i, s in enumerate(seals):
    groups[s.SealNet].append(i)
outputs = [None] * len(seals)  # 원래 순서로 채울 버퍼



this_seal = seals[3]
n_pop = 1200
h_in = np.random.randint(100,500,size=n_pop)[:, None]
h_out = np.random.randint(100,500,size=n_pop)[:, None]
psr = np.random.randint(0,10,size=n_pop)[:, None]
X = np.concatenate([h_in, h_out, psr], axis=1)


fluid = {
    'mu': this_seal.mu,
    'rho': this_seal.rho,
}
import time
t_start = time.time()
for idx in range(n_pop):
    t1 = time.time()
    geometry = {
        'hIn': float(X[idx,0]*1e-6),
        'hOut': float(X[idx,1]*1e-6),
        'Ds': this_seal.Ds,
        'Ls': this_seal.Ls,
        'NxSeal': 25
    }
    op_conditions = {
        'dp': this_seal.dp,
        'psr': float(X[idx,2]*1e-1),
        'w_vec': w_vec,
    }

    _, RDC, _, _, _, _, _ = seal_solver(geometry, fluid, op_conditions)
    
    t2 = time.time()
    print(t2-t1)
t_end = time.time()
elapsed = t_end - t_start
print(elapsed)


RDC_ = RDC[:,2:6]
t_start = time.time()
Y = seal.predict(this_seal.SealNet,X,w_vec) # K k C c
t_end = time.time()
elapsed = t_end - t_start
print(elapsed)
# Y2 = Y.squeeze(0).T
# err = np.abs(RDC_-Y2)/np.abs(RDC_)*100
# print(err)







# for idxs in (1, 2, 3):
#     idx = groups[idxs]
#     n_idx = len(idx)
#     h_in = np.random.randint(100,500,size=n_idx)[:, None]*1e-6
#     h_out = np.random.randint(100,500,size=n_idx)[:, None]*1e-6
#     psr = np.random.randint(0,10,size=n_idx)[:, None]*1e-1
#     X = np.concatenate([h_in, h_out, psr], axis=1)
#     Y = seal.predict(idxs,X,w_vec) # K k C c
#     for k, i in enumerate(idx):
#         outputs[i] = Y[k]









brg = BearingH5Loader()
# bearing_id: 1 ~ 55
brg1_id = np.linspace(1,6,6, dtype=int)[:, None]
brg2_id = np.linspace(2,7,6, dtype=int)[:, None]
brg1_cr = np.linspace(10,20,6)[:, None] # Db * cr / 1000
brg2_cr = np.linspace(12,26,6)[:, None] # Db * cr / 1000
brg_params_ = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)

brg_params = brg_params_[1,:]

K_, C_, _ = brg.calculate_bearing_coefficients(brg_params[0].astype(int), brgs[0].Db, brgs[0].Db*brg_params[1]*1e-4, brgs[0].mu, brgs[0].load, w_vec)



# # SealModel()

# Cr = brgs[0].Db/1000
# bearing_id = 21
# K_, C_ = brg.get_KC_matrix(bearing_id, brgs, w_vec)

# import time
# t_start = time.time()
# for i in range(10000):
#     K_brg, C_brg, S_brg = brg.calculate_bearing_coefficients(bearing_id=bearing_id, Db=brgs[0].Db, Cr=Cr, mu=brgs[0].mu, normal_load=brgs[0].load, omega_vec=w_vec)
# t_end = time.time()
# print(t_end-t_start)

# print(K_brg)


# for i, w in enumerate(w_vec):
#     C = w*mat_Cg
#     A = np.block([
#         [-np.linalg.solve(mat_M, C), -np.linalg.solve(mat_M, mat_Ks)],
#         [np.eye(n_dof), np.zeros((n_dof, n_dof))]
#     ])
#     eigvals_, eigvecs_ = np.linalg.eig(A)
    
    # print(np.imag(eigvals_))

    
    
# A = np.block([
#     [-np.linalg.solve(mat_M, C), -np.linalg.solve(mat_M, K)],
#     [np.eye(Nd), np.zeros((Nd, Nd))]
# ])


# def critspd(matrix_params):
    
#     return cs
    
# if __name__ == "__main__":
    
#     critspd(matrix_params)
    
    
    
    # A = torch.block_diag()  # 없음 -> torch.cat으로 작성
    # A = torch.cat([
    #     torch.cat([-torch.linalg.solve(M, C), -torch.linalg.solve(M, K)], dim=1),
    #     torch.cat([torch.eye(Nd), torch.zeros((Nd, Nd))], dim=1)
    # ], dim=0)
    
    
    
    
    
    
    # # --- 1. 베어링 로더 테스트 ---
    # loader = BearingH5Loader()
    # target_id = 21
    # vec_S = np.array([0.147, 0.101])
    # bearing_params = {
    #         "bearing_id": target_id,
    #         "vec_S": vec_S
    # }
    
    # K_matrices, C_matrices = loader.calculate_nond_RDC(**bearing_params)

    # # 4. 결과 출력
    # if K_matrices is not None and C_matrices is not None:
    #     print("\n계산 결과:")
    #     for i, S in enumerate(vec_S):
    #         print(f"\n--- @ Sommerfeld Number S = {S:.4f} ---")
    #         print("무차원 강성 행렬 K:")
    #         print(K_matrices[i])
    #         print("무차원 감쇠 행렬 C:")
    #         print(C_matrices[i])
    
    # # --- 2. 씰 FNO 모델 로더 테스트 ---
    
    
# print("\n" + "="*50)
# seal_model = SealFNOModel()

# if seal_model.models:
#     # 테스트할 모델 ID 선택
#     test_model_id = 2
    
#     # 배치 예측을 위한 임의의 입력 파라미터 생성 (B=2)
#     # 파라미터 순서: [hIn(um), hOut(um), psr*10]
#     test_params = np.array([
#         [30.0, 25.0, 5.0],  # Case 1
#         [40.0, 40.0, 2.0]   # Case 2
#     ])

#     # 예측할 회전속도 벡터
#     test_w_vec = np.linspace(1500, 5000, 10) * np.pi / 30

#     print(f"\n--- 씰 모델 ID #{test_model_id} 예측 테스트 ---")
#     predicted_rdcs = seal_model.predict(test_model_id, test_params, test_w_vec)

#     if predicted_rdcs is not None:
#         print(f"입력 파라미터 Shape: {test_params.shape}")
#         print(f"예측된 RDC Shape: {predicted_rdcs.shape}")
#         print("RDC 채널 순서: [C, c, K, k]")
#         # 첫 번째 케이스의 K(직접강성) 값 출력
#         print("\nCase 1의 예측된 직접 감쇠(C):")
#         print(predicted_rdcs[0, 0, :])
#         print("\nCase 1의 예측된 연성 감쇠(c):")
#         print(predicted_rdcs[0, 1, :])
#         print("\nCase 1의 예측된 직접 강성(K):")
#         print(predicted_rdcs[0, 2, :])
#         print("\nCase 1의 예측된 연성 강성(k):")
#         print(predicted_rdcs[0, 3, :])
            
# geometry = {
#     'hIn': float(test_params[0,0]*1e-6),
#     'hOut': float(test_params[0,1]*1e-6),
#     'Ds': this_seal.Ds,
#     'Ls': this_seal.Ls,
#     'NxSeal': 25
# }
# fluid = {
#     'mu': this_seal.mu,
#     'rho': this_seal.rho,
# }
# op_conditions = {
#     'dp': this_seal.dp,
#     'psr': float(test_params[0,2]*1e-1),
#     'w_vec': w_vec,
# }

# _, RDC, _, _, _, _, _ = seal_solver(geometry, fluid, op_conditions)

# RDC_ = RDC[:,2:6]


## optimization parameters

# input_brg = [1, 1, 10, 10]
# paretoRatio = 0.4;
# plottt = {'gaplotpareto','gaplotstopping','gaplotrankhist','gaplotspread'};

# nVarBrg = 2*nBrg; % [Brg#1 Cr1/Db*10000 Brg#2 Cr2/Db*10000]
# % intconBrg = 1:2*nBrg;
# lbBrg = [1 1 5 5]; 
# ubBrg = [55 55 20 20];

# nVarSeal = 3*nSeal; % [hIn* hOut*1000 psr*10]
# % intconSeal = 1:3*nSeal;
# lbSeal = [20 20 0]; 
# ubSeal = [50 50 10];