import os
import numpy as np
from import_data import rotor_import, calculate_bearing_loads
from loader_brg_seal import BearingH5Loader, SealFNOModel
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

brg = BearingH5Loader()
seal = SealFNOModel()

Cr = brgs[0].Db/1000
bearing_id = 21
w_vec = np.linspace(w_range[0], w_range[1], 15)


K_brg, C_brg, S_brg = BearingH5Loader.calculate_bearing_coefficients(brg, bearing_id=bearing_id, Db=brgs[0].Db, Cr=Cr, mu=brgs[0].mu, normal_load=brgs[0].load, omega_vec=w_vec)

print(K_brg)


for i, w in enumerate(w_vec):
    C = w*mat_Cg
    A = np.block([
        [-np.linalg.solve(mat_M, C), -np.linalg.solve(mat_M, mat_Ks)],
        [np.eye(n_dof), np.zeros((n_dof, n_dof))]
    ])
    eigvals_, eigvecs_ = np.linalg.eig(A)
    
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
    #     test_model_id = 1 
        
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
    #         print("\nCase 1의 예측된 직접 강성(K):")
    #         print(predicted_rdcs[0, 2, :])
            
            
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