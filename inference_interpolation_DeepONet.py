# inference_interpolation.py
import os, h5py, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# ----------------- 경로/옵션 -----------------
DATA_DIR   = 'dataset/data/tapered_seal'
MAT_FILE   = os.path.join(DATA_DIR, '20250812_T_113003', 'dataset.mat')
CKPT_PATH  = 'net/DeepONet_multihead.pth'        # state_dict 저장본(.pth)
TS_PATH    = 'net/DeepONet_multihead.pt'          # TorchScript 저장본(.pt)
USE_TS     = True                           # True: TorchScript 경로, False: state_dict 경로
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- 유틸 -----------------
def build_grids(w):
    w = np.asarray(w).ravel()
    w_min, w_max = float(w.min()), float(w.max())
    def norm_w(x):
        return 2.0*(x - w_min)/(w_max - w_min) - 1.0
    return w_min, w_max, norm_w

def inverse_scale_channel(z_scaled, mean, std):
    # z_scaled: [B, L], mean/std: [1] or scalar
    return z_scaled * std + mean

# ----------------- 데이터 로드/전처리 -----------------
with h5py.File(MAT_FILE, 'r') as mat:
    input_nond = np.array(mat.get('inputNond'))          # [nPara, nData]
    w_vec      = np.array(mat['params/wVec'])            # [1, nVel]
    rdc        = np.array(mat.get('RDC'))                # [6, nVel, nData]

n_para, n_data = input_nond.shape
n_rdc_coeffs   = rdc.shape[0]
_, n_vel       = w_vec.shape

X_params   = input_nond.T                                # [nData, nPara]
y_functions = rdc.transpose(2, 0, 1)                    # [nData, 6, nVel]
w = w_vec.squeeze()                                      # [nVel]
wmin, wmax, norm_w = build_grids(w)

# train/val/test split과 스케일러는 원학습과 동일 전략(70/15/15)
indices = np.arange(n_data)
train_size = int(n_data*0.7); val_size = int(n_data*0.15)
train_idx, val_idx, test_idx = np.split(np.random.permutation(indices),
                                        [train_size, train_size+val_size])

# ----------------- 모델 로드 -----------------
if USE_TS:
    # TorchScript: 모델 클래스 필요 없음
    model = torch.jit.load(TS_PATH, map_location='cpu').to(DEVICE).eval()

    # .pth에 저장했던 하이퍼파라미터/스케일러 통계를 같이 쓰고 싶다면 .pth를 부가로 읽음(권장)
    ckpt = torch.load(CKPT_PATH, map_location='cpu',weights_only=False)
    hparams = ckpt.get('hparams', {})
    scaler_X_mean = ckpt['scaler_X_mean']; scaler_X_std = ckpt['scaler_X_std']
    scalers_y_mean = np.array(ckpt['scalers_y_mean']).reshape(n_rdc_coeffs, 1)  # [6,1]
    scalers_y_std  = np.array(ckpt['scalers_y_std']).reshape(n_rdc_coeffs, 1)   # [6,1]
else:
    # state_dict: 아키텍처 재구성 필요 (학습 시 구조와 동일해야 함)
    from neuralop.models import FNO

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
            pe = self.param_encoder(params)                         # [B, emb]
            pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)        # [B, L, emb]
            x  = torch.cat([grid, pe], dim=-1).permute(0,2,1)       # [B, 1+emb, L]
            feat = self.trunk(x)                                    # [B, C, L]
            outs = [head(feat) for head in self.heads]              # list of [B,1,L]
            return torch.cat(outs, dim=1)                           # [B, 6, L]

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE,weights_only=False)
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

    scaler_X_mean = ckpt['scaler_X_mean']; scaler_X_std = ckpt['scaler_X_std']
    scalers_y_mean = np.array(ckpt['scalers_y_mean']).reshape(n_rdc_coeffs, 1)  # [6,1]
    scalers_y_std  = np.array(ckpt['scalers_y_std']).reshape(n_rdc_coeffs, 1)   # [6,1]

# ----------------- 스케일러 구성 -----------------
# X 표준화 (훈련 스케일러 통계 사용)
X_scaled = (X_params - scaler_X_mean) / (scaler_X_std + 1e-12)

# y는 평가 때 역스케일이 필요하므로 통계만 보관
# ----------------- 내삽 쿼리 그리드 준비 -----------------
B_eval  = min(200, len(test_idx))
idx_eval = test_idx[:B_eval]
n_query = 64

wq = np.linspace(wmin, wmax, n_query, dtype=np.float32)
wq_norm = norm_w(wq).astype(np.float32)                 # [-1,1]
grid_q = torch.from_numpy(wq_norm[:, None]).unsqueeze(0)  # [1, Lq, 1]
grid_q = grid_q.to(DEVICE).repeat(B_eval, 1, 1)

start_time = time.time()
# ----------------- 모델 추론 -----------------
with torch.no_grad():
    params_batch = torch.tensor(X_scaled[idx_eval], dtype=torch.float32, device=DEVICE)
    preds_scaled_q = model(params_batch, grid_q).cpu().numpy()     # [B, 6, Lq]
end_time = time.time()


# 역스케일
preds_orig_q = np.empty_like(preds_scaled_q)
for ch in range(n_rdc_coeffs):
    preds_orig_q[:, ch, :] = inverse_scale_channel(
        preds_scaled_q[:, ch, :], scalers_y_mean[ch, 0], scalers_y_std[ch, 0]
    )
print(f"Inference time for {B_eval} samples: {end_time - start_time:.6f} seconds")
print(f"Average per sample: {(end_time - start_time)/B_eval:.6f} seconds")

# ----------------- GT(보간) 생성 후 지표 -----------------
# 전체 GT를 원스케일로 되돌린 후, wq 위치 값은 선형보간으로 획득
y_true_q = np.zeros_like(preds_orig_q)
for ch in range(n_rdc_coeffs):
    # 원본(전체 그리드)로 역스케일
    full_scaled = (y_functions[idx_eval, ch, :]).reshape(B_eval, n_vel)  # 이미 원스케일이면 이 줄만 바꾸면 됨
    # 주의: y_functions가 "원스케일"인 경우 아래 한 줄로 교체
    full_orig = full_scaled  # 만약 y_functions가 이미 원스케일인 경우
    # 만약 y_functions가 "표준화된 값"이었다면:
    # full_orig = full_scaled * scalers_y_std[ch,0] + scalers_y_mean[ch,0]
    for b in range(B_eval):
        y_true_q[b, ch, :] = np.interp(wq, w, full_orig[b])

labels = ['K','k','C','c','M','m']
print("[Interpolation inside wVec range]")
for ch, name in enumerate(labels):
    y_true = y_true_q[:, ch, :].ravel()
    y_pred = preds_orig_q[:, ch, :].ravel()
    mse = mean_squared_error(y_true, y_pred); rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    yr  = y_true.max() - y_true.min()
    rrmse = rmse/(yr + 1e-12)
    mape  = np.mean(np.abs((y_true - y_pred)/(np.abs(y_true) + 1e-12)))
    print(f"[{name}] RMSE={rmse:.6g}, MAE={mae:.6g}, R²={r2:.6f}, rRMSE={100*rrmse:.3f}%, MAPE={100*mape:.3f}%")

#%%
# ----------------- 시각화 -----------------
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mcolors_list = list(mcolors.TABLEAU_COLORS.values())
n_plot = min(5, B_eval)  # 최대 5개만 시각화

rdc_labels = ['K', 'k', 'C', 'c', 'M', 'm']
rdc_units = ['N/m', 'N/m', 'N s/m', 'N s/m', 'kg', 'kg']
    
fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for ch in range(n_rdc_coeffs):
    ax = axes[ch]
    for idx in range(n_plot):
        color = mcolors_list[idx % len(mcolors_list)]
        # GT: w 전체 구간
        ax.plot(
            w, 
            y_functions[idx_eval[idx], ch, :],  # 이미 원스케일 값
            linestyle='-', linewidth=2, color=color,
            label=f"True #{idx_eval[idx]}"
        )
        # Pred: wq 내삽 구간
        ax.plot(
            wq,
            preds_orig_q[idx, ch, :],
            linestyle='--', marker='o', markersize=3, linewidth=1.5, color=color,
            label=f"Pred #{idx_eval[idx]}"
        )
    ax.set_title(labels[ch])
    ax.set_xlabel('Rotational speed [rad/s]')
    ax.set_ylabel(rdc_units[ch] if 'rdc_units' in locals() else labels[ch])
    ax.grid(True)
    ax.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.show()

