import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# 컨텍스트에 제공된 관련 모듈 임포트
from ng_seal_solver import main_seal_solver
from neuralop.models import FNO


# --- 1. 설정: 모델 경로, 디바이스, 솔버 기본 파라미터 ---
CKPT_PATH = 'net/fno_multihead.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# tapered_seal.py 에서 가져온 솔버 기본 설정
wRange = np.array([1500, 5000]) * np.pi / 30
w_vec = np.linspace(wRange[0], wRange[1] * 1.25, 34) # 학습에 사용된 속도 그리드

# Ds = 0.22
# Ls = 0.02
# NxSeal = 25
# mu = 1.4e-3
# rho = 850
# dp = 850000

Ds = 0.23
Ls = 0.15
NxSeal = 15
mu = 1.4e-3
rho = 850
dp = 16000000 # 20250812_T_113003, seal 3

lb = [20, 20, 0]
ub = [50, 50, 10]
vhIn = np.linspace(lb[0], ub[0], 15)
vhOut = np.linspace(lb[1], ub[1], 15)
vPsr = np.linspace(lb[2], ub[2], 6)

lb = [60, 60, 0]
ub = [80, 80, 10]
vhIn = np.linspace(lb[0], ub[0], 21)
vhOut = np.linspace(lb[1], ub[1], 21)
vPsr = np.linspace(lb[2], ub[2], 11)

lb = [20, 20, 0]
ub = [50, 50, 10]
vhIn = np.linspace(lb[0], ub[0], 31)
vhOut = np.linspace(lb[1], ub[1], 31)
vPsr = np.linspace(lb[2], ub[2], 11)

inputVec = np.array(list(itertools.product(vhIn, vhOut, vPsr)))

# 솔버와 FNO에 대해 실행할 케이스 수를 별도로 설정
num_solver_cases = 5
num_fno_cases = 5000
# print(f"Using device: {DEVICE}")

# --- 2. FNO 모델 아키텍처 정의 (inference_interpolation_fno.py와 동일) ---
class MultiHeadParametricFNO(nn.Module):
    def __init__(self, n_params, param_embedding_dim, fno_modes,
                 fno_hidden_channels, in_channels, n_heads, n_layers, shared_out_channels):
        super().__init__()
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, param_embedding_dim),
            nn.ReLU(),
            nn.Linear(param_embedding_dim, param_embedding_dim)
        )
        self.trunk = FNO(
            n_modes=(fno_modes,),
            hidden_channels=fno_hidden_channels,
            n_layers=n_layers,
            in_channels=in_channels + param_embedding_dim,
            out_channels=shared_out_channels
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(shared_out_channels, shared_out_channels, 1),
                nn.GELU(),
                nn.BatchNorm1d(shared_out_channels),
                nn.Dropout(0.1),
                nn.Conv1d(shared_out_channels, shared_out_channels//2, 1),
                nn.GELU(),
                nn.BatchNorm1d(shared_out_channels//2),
                nn.Dropout(0.1),
                nn.Conv1d(shared_out_channels//2, 1, 1)
            ) for _ in range(n_heads)
        ])

    def forward(self, params, grid):
        pe = self.param_encoder(params)
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)
        x = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)
        feat = self.trunk(x)
        outs = [head(feat) for head in self.heads]
        return torch.cat(outs, dim=1)

# --- 3. 모델 및 스케일러 로드 ---
print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
hp = ckpt['hparams']

model = MultiHeadParametricFNO(
    n_params=hp['n_params'],
    param_embedding_dim=hp['param_embedding_dim'],
    fno_modes=hp['fno_modes'],
    fno_hidden_channels=hp['fno_hidden_channels'],
    in_channels=1,
    n_heads=hp['n_heads'],
    n_layers=hp['n_layers'],
    shared_out_channels=hp['shared_out_channels']
).to(DEVICE).eval()
model.load_state_dict(ckpt['state_dict'], strict=False)

# 추론에 필요한 스케일러 정보 로드
scaler_X_mean = ckpt['scaler_X_mean']
scaler_X_std = ckpt['scaler_X_std']
scalers_y_mean = np.array(ckpt['scalers_y_mean'])
scalers_y_std = np.array(ckpt['scalers_y_std'])
n_rdc_coeffs = hp['n_heads']

print("Model and scalers loaded successfully.")

# --- 4. 검증할 입력 파라미터 선택 ---
# inputVec에서 검증할 케이스의 인덱스를 무작위로 선택합니다.
total_cases = inputVec.shape[0]
# FNO 케이스를 먼저 선택합니다.
fno_test_indices = np.random.choice(total_cases, num_fno_cases, replace=False)
# 솔버 케이스는 FNO 케이스의 부분집합이 되도록 선택합니다.
solver_test_indices = np.random.choice(fno_test_indices, num_solver_cases, replace=False)

# --- 5. 수치해석 솔버 실행 (Ground Truth 생성) 및 시간 측정 ---
print(f"\n--- Running Numerical Solver for {num_solver_cases} cases (sequentially) ---")
solver_results = {} # 결과를 저장할 딕셔너리
solver_start_time = time.perf_counter()

for idx in solver_test_indices:
    current_input = inputVec[idx]
    hIn_val, hOut_val, psr_val_times_10 = current_input
    hIn_solver = Ds * hIn_val / 10000
    hOut_solver = Ds * hOut_val / 10000

    geometry = {'hIn': hIn_solver, 'hOut': hOut_solver, 'Ds': Ds, 'Ls': Ls, 'NxSeal': NxSeal}
    fluid = {'mu': mu, 'rho': rho}
    op_conditions = {'dp': dp, 'psr': psr_val_times_10 / 10, 'w_vec': w_vec}

    _, rdc_solver, _, _, _, _, _ = main_seal_solver(geometry, fluid, op_conditions)

    # FNO 순서에 맞게 솔버 결과를 재정렬합니다: K(4), k(5), C(2), c(3), M(0), m(1)
    reorder_indices = [4, 5, 2, 3, 0, 1]
    rdc_solver_reordered = rdc_solver.T[reorder_indices, :]
    solver_results[idx] = rdc_solver_reordered

solver_end_time = time.perf_counter()
total_solver_time = solver_end_time - solver_start_time
print(f"Total solver time for {num_solver_cases} cases: {total_solver_time:.4f}s")

# --- 6. FNO 모델 추론 실행 (Batch) 및 시간 측정 ---
print(f"\n--- Running FNO Model for {num_fno_cases} cases (in a single batch) ---")
fno_start_time = time.perf_counter()

# 6.1. 모든 검증 케이스에 대한 입력 데이터를 하나의 배치로 준비
input_nond_batch = inputVec[fno_test_indices]
input_scaled_batch = (input_nond_batch - scaler_X_mean) / (scaler_X_std + 1e-9)
params_tensor_batch = torch.tensor(input_scaled_batch, dtype=torch.float32, device=DEVICE)

w_norm = 2 * (w_vec - w_vec.min()) / (w_vec.max() - w_vec.min()) - 1.0
grid_tensor_batch = torch.tensor(w_norm, dtype=torch.float32).view(1, -1, 1).to(DEVICE)
grid_tensor_batch = grid_tensor_batch.repeat(num_fno_cases, 1, 1)

# 6.2. 배치 추론 실행
with torch.no_grad():
    pred_scaled_batch = model(params_tensor_batch, grid_tensor_batch).cpu().numpy()

fno_end_time = time.perf_counter()
total_fno_time = fno_end_time - fno_start_time
print(f"Total FNO inference time for {num_fno_cases} cases: {total_fno_time:.6f}s")

# 6.3. 배치 결과 역스케일링 (Vectorized)
mean_reshaped = scalers_y_mean.reshape(1, n_rdc_coeffs, 1)
std_reshaped = scalers_y_std.reshape(1, n_rdc_coeffs, 1)
pred_orig_batch = pred_scaled_batch * std_reshaped + mean_reshaped

# --- 7. 결과 비교 및 시각화 ---
print(f"\n--- Comparing Solver and FNO results for {num_solver_cases} common cases ---")
rdc_labels = ['K', 'k', 'C', 'c', 'M', 'm']

# FNO 결과에서 솔버 케이스에 해당하는 예측을 찾기 위한 인덱스 매핑
fno_idx_map = {original_idx: i for i, original_idx in enumerate(fno_test_indices)}

# for idx in solver_test_indices:
#     print(f"\n--- Verifying Case (inputVec index: {idx}) ---")

#     # 솔버 결과(Ground Truth)와 FNO 예측 결과 가져오기
#     rdc_solver_truth = solver_results[idx]
#     fno_pred_idx = fno_idx_map[idx]
#     rdc_fno_pred = pred_orig_batch[fno_pred_idx]

#     # 백분율 오차(Percentage Error) 시각화
#     fig, axes = plt.subplots(3, 2, figsize=(15, 12))
#     fig.suptitle(f"Percentage Error: inputVec index {idx}", fontsize=16)
#     axes = axes.flatten()

#     print("Percentage Error Comparison:")
#     for i in range(n_rdc_coeffs):
#         ax = axes[i]
#         truth = rdc_solver_truth[i, :]
#         pred = rdc_fno_pred[i, :]
#         # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더함
#         percentage_error = np.abs((pred - truth) / (truth + 1e-12)) * 100

#         ax.plot(w_vec, percentage_error, 'go-', label='Percentage Error (%)')
#         ax.set_title(f"RDC: {rdc_labels[i]}")
#         ax.set_xlabel('Rotational Speed (rad/s)')
#         ax.set_ylabel('Percentage Error (%)')
#         ax.grid(True); ax.legend()
#         print(f"  - {rdc_labels[i]:<2}: Mean Percentage Error = {np.mean(percentage_error):.2f}%")

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()

# --- 8. 최종 실행 시간 비교 ---
print("\n" + "="*50)
print("           Execution Time Summary")
print("="*50)

avg_solver_time = total_solver_time / num_solver_cases if num_solver_cases > 0 else 0
avg_fno_time = total_fno_time / num_fno_cases if num_fno_cases > 0 else 0

print(f"Numerical Solver ({num_solver_cases} cases, sequential):")
print(f"  - Total time:   {total_solver_time:.4f}s")
print(f"  - Average time: {avg_solver_time:.4f}s per case")

print(f"\nFNO Model ({num_fno_cases} cases, batched):")
print(f"  - Total time:   {total_fno_time:.6f}s")
print(f"  - Average time: {avg_fno_time:.6f}s per case")

if avg_fno_time > 0 and avg_solver_time > 0:
    speedup = avg_solver_time / avg_fno_time
    print(f"\n>> FNO model is approximately {speedup:.2f}x faster than the solver (on a per-case basis).")
print("="*50)
