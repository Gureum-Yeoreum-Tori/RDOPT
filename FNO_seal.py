#%%
import os
import glob
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import neuralop as nop
from neuralop.models import FNO

# ---- Safe checkpoint loader (handles PyTorch 2.6 weights_only behavior) ----
import torch.serialization as _ts

def load_ckpt_any(path, device):
    """Load a checkpoint robustly.
    Tries weights_only=True first; falls back to allowlisting and weights_only=False if needed.
    Returns a state_dict without '_metadata'.
    """
    # 1) Try safe load
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
        sd = obj.get('state_dict', obj)
    except Exception:
        # 2) Fallback: allowlist known globals then unsafe load
        try:
            # Some older ckpts reference internals like torch._C._nn.gelu
            from torch._C import _nn as _torch_c_nn  # noqa: F401
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals([torch._C._nn.gelu])
            except Exception:
                pass
        except Exception:
            pass
        obj = torch.load(path, map_location=device, weights_only=False)
        sd = obj.get('state_dict', obj)
    # strip metadata key if present
    if isinstance(sd, dict):
        sd.pop('_metadata', None)
    return sd

print("FNO example for predicting seal rotordynamic coefficients.")

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_file = os.path.join(data_dir, '20250812_T_113003', 'dataset.mat')

# 데이터 로딩 및 전처리
with h5py.File(mat_file, 'r') as mat:
    # inputNond: [nPara, nData] -> 형상 파라미터
    input_nond = np.array(mat.get('inputNond'))
    # wVec: [1, nVel] -> 회전 속도 벡터 (좌표 그리드)
    w_vec = np.array(mat['params/wVec'])
    # RDC: [6, nVel, nData] -> 동특성 계수 (타겟 함수)
    rdc = np.array(mat.get('RDC'))

    n_para, n_data = input_nond.shape
    _, n_vel = w_vec.shape
    n_rdc_coeffs = rdc.shape[0] # 6 (Kxx, Kxy, ..., Cxy)

    # 입력 데이터 (X): 형상 파라미터 [nData, nPara]
    X_params = input_nond.T

    # 출력 데이터 (y): 동특성 계수 함수 [nData, nVel, nRDC]
    # FNO는 (batch, channels, grid_points) 형태를 선호하므로, [nData, nRDC, nVel]로 변경합니다.
    y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]

    # 회전 속도 그리드: [nVel, 1] 형태로 준비
    
    w = w_vec.squeeze()                         # [n_vel]
    w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0
    grid = w_norm[:, None]                      # [n_vel, 1]    
    # grid = w_norm.T

# 데이터 스케일링
# 입력 파라미터 스케일링
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_params) 

# 출력 함수 스케일링
 # 각 RDC 계수를 개별적으로 스케일링하여 각 계수의 통계적 특성을 독립적으로 처리합니다.
 # 이는 각 계수를 예측하는 별도의 모델을 학습시킬 때 더 적합합니다.
scalers_y = [StandardScaler() for _ in range(n_rdc_coeffs)]
y_scaled_channels = []
# y_functions shape: [n_data, n_rdc_coeffs, n_vel]
for i in range(n_rdc_coeffs):
    # 각 채널(RDC)의 데이터를 [n_data * n_vel, 1] 형태로 만들어 스케일러에 적용
    channel_data = y_functions[:, i, :].reshape(-1, 1)
    scaled_channel_data = scalers_y[i].fit_transform(channel_data)
    # 원래 형태 [n_data, n_vel]로 복원
    y_scaled_channels.append(scaled_channel_data.reshape(n_data, n_vel))

# 스케일링된 채널들을 다시 하나로 합칩니다. [n_data, n_rdc_coeffs, n_vel]
y_scaled = np.stack(y_scaled_channels, axis=1)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Torch 텐서로 변환
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(X_tensor, y_tensor)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.15)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

batch_size = 2**10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#%%
# 2. Define models (single, multihead, separate)
class ParametricFNO(nn.Module):
    """
    기존: 형상 파라미터를 조건으로 받아 함수를 예측하는 FNO 모델 (단일 네트워크)
    outputs: [B, out_channels, n_vel]
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, out_channels):
        super().__init__()
        self.n_params = n_params
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, param_embedding_dim),
            nn.ReLU(),
            nn.Linear(param_embedding_dim, param_embedding_dim)
        )
        self.fno = FNO(
            n_modes=(fno_modes,),
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels + param_embedding_dim,  # grid(1) + params
            out_channels=out_channels
        )

    def forward(self, params, grid):
        # params: [B, n_params], grid: [B, n_vel, 1]
        pe = self.param_encoder(params)                      # [B, emb]
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)    # [B, n_vel, emb]
        fno_in = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)  # [B, 1+emb, n_vel]
        out = self.fno(fno_in)  # [B, out_channels, n_vel]
        return out

class MultiHeadParametricFNO(nn.Module):
    """
    FNO 본체는 공유하고, 채널별 1x1 Conv1d 헤드를 분리하는 멀티헤드 구조.
    outputs: [B, n_heads(=n_rdc_coeffs), n_vel]
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, n_heads, shared_out_channels=128):
        super().__init__()
        self.n_params = n_params
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, param_embedding_dim), nn.ReLU(), nn.Linear(param_embedding_dim, param_embedding_dim)
        )
        self.trunk = FNO(
            n_modes=(fno_modes,),
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels + param_embedding_dim,
            out_channels=shared_out_channels
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(shared_out_channels, shared_out_channels//2, 1),
                nn.GELU(),
                nn.Conv1d(shared_out_channels//2, 1, 1)
            ) for _ in range(n_heads)
        ])
        # self.heads = nn.ModuleList([nn.Conv1d(shared_out_channels, 1, kernel_size=1) for _ in range(n_heads)])

    def forward(self, params, grid):
        pe = self.param_encoder(params)                       # [B, emb]
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)     # [B, n_vel, emb]
        x = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)   # [B, 1+emb, n_vel]
        feat = self.trunk(x)                                  # [B, Csh, n_vel]
        outs = [head(feat) for head in self.heads]            # each: [B,1,n_vel]
        return torch.cat(outs, dim=1)                         # [B, n_heads, n_vel]

# ----- 학습/평가 모드 선택 -----
# 'single'    : 기존 한 모델이 6채널 동시 예측
# 'multihead' : 본체 공유 + 채널별 헤드 분리
# 'separate'  : 6개 독립 모델(각 1채널)

MODEL_MODE = os.getenv('MODEL_MODE', 'separate')  # 필요시 env로 바꿀 수 있음
print(f"MODEL_MODE = {MODEL_MODE}")

criterion = nop.losses.LpLoss(d=1, p=2)
optimizer = None
epochs = 100
best_val_loss = float('inf')
base_dir = 'net'
os.makedirs(base_dir, exist_ok=True)
fno_modes = min(n_vel//2, 32)   # 32~64 시도
fno_hidden_channels = 256
in_channels = grid_tensor.shape[-1]

#%%
if MODEL_MODE == 'single':
    model = ParametricFNO(
        n_params=n_para,
        param_embedding_dim=64,
        fno_modes=fno_modes,
        fno_hidden_channels=fno_hidden_channels,
        in_channels=in_channels,
        out_channels=n_rdc_coeffs
    ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model_save_path = os.path.join(base_dir, 'fno_seal_best_single.pth')

    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        for params, functions in train_loader:
            params, functions = params.to(device), functions.to(device)
            batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(params, batch_grid)
            loss = criterion(outputs, functions)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * params.size(0)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for params, functions in val_loader:
                params, functions = params.to(device), functions.to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                outputs = model(params, batch_grid)
                val_loss += criterion(outputs, functions).item() * params.size(0)
        train_loss /= len(train_loader.dataset); val_loss /= len(val_loader.dataset)
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'state_dict': model.state_dict()}, model_save_path)
        scheduler.step()

    # --- Evaluate ---
    sd = load_ckpt_any(model_save_path, device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    n_test_samples = len(test_dataset.indices)
    test_params = X_tensor[test_dataset.indices].to(device)
    grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)
    with torch.no_grad():
        predictions_scaled = model(test_params, grid_repeated).cpu().numpy()

elif MODEL_MODE == 'multihead':
    model = MultiHeadParametricFNO(
        n_params=n_para,
        param_embedding_dim=64,
        fno_modes=fno_modes,
        fno_hidden_channels=fno_hidden_channels,
        in_channels=in_channels,
        n_heads=n_rdc_coeffs,
        shared_out_channels=128
    ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model_save_path = os.path.join(base_dir, 'fno_seal_best_multihead.pth')

    for epoch in range(epochs):
        model.train(); train_loss = 0.0
        for params, functions in train_loader:
            params, functions = params.to(device), functions.to(device)
            batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(params, batch_grid)
            loss = criterion(outputs, functions)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * params.size(0)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for params, functions in val_loader:
                params, functions = params.to(device), functions.to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                outputs = model(params, batch_grid)
                val_loss += criterion(outputs, functions).item() * params.size(0)
        train_loss /= len(train_loader.dataset); val_loss /= len(val_loader.dataset)
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'state_dict': model.state_dict()}, model_save_path)
        scheduler.step()

    sd = load_ckpt_any(model_save_path, device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    n_test_samples = len(test_dataset.indices)
    test_params = X_tensor[test_dataset.indices].to(device)
    grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)
    with torch.no_grad():
        predictions_scaled = model(test_params, grid_repeated).cpu().numpy()

elif MODEL_MODE == 'separate':
    # 6개 독립 모델 (각 1 채널)
    models = []
    save_paths = [os.path.join(base_dir, f'fno_seal_best_ch{i}.pth') for i in range(n_rdc_coeffs)]
    for ch in range(n_rdc_coeffs):
        m = ParametricFNO(
            n_params=n_para,
            param_embedding_dim=64,
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            in_channels=in_channels,
            out_channels=1
        ).to(device)
        # opt = optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5)
        opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # 채널별 데이터로더: [B, 1, n_vel]
        y_ch = y_tensor[:, ch, :].unsqueeze(1)
        ds_ch = TensorDataset(X_tensor, y_ch)
        tr_ch, va_ch, te_ch = random_split(ds_ch, [train_size, val_size, test_size])
        tl = DataLoader(tr_ch, batch_size=batch_size, shuffle=True)
        vl = DataLoader(va_ch, batch_size=batch_size, shuffle=False)

        best_ch = float('inf')
        for epoch in range(epochs):
            m.train(); tr_loss = 0.0
            for params, target in tl:
                params, target = params.to(device), target.to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                opt.zero_grad()
                out = m(params, batch_grid)  # [B,1,n_vel]
                loss = criterion(out, target)
                loss.backward(); opt.step()
                tr_loss += loss.item() * params.size(0)
            m.eval(); va_loss = 0.0
            with torch.no_grad():
                for params, target in vl:
                    params, target = params.to(device), target.to(device)
                    batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                    out = m(params, batch_grid)
                    va_loss += criterion(out, target).item() * params.size(0)
            tr_loss /= len(tl.dataset); va_loss /= len(vl.dataset)
            if (epoch+1) % 100 == 0:
                print(f'[ch{ch}] Epoch {epoch+1}/{epochs}, Train {tr_loss:.6f}, Val {va_loss:.6f}')
            if va_loss < best_ch:
                best_ch = va_loss
                torch.save({'state_dict': m.state_dict()}, save_paths[ch])
            sch.step()
        models.append(m)

    # --- Evaluate: 모든 채널 예측 쌓기 ---
    n_test_samples = len(test_dataset.indices)
    test_params = X_tensor[test_dataset.indices].to(device)
    grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)
    preds_list = []
    for ch in range(n_rdc_coeffs):
        m = models[ch]
        sd = load_ckpt_any(save_paths[ch], device)
        m.load_state_dict(sd, strict=False); m.eval()
        with torch.no_grad():
            pred_ch = m(test_params, grid_repeated).cpu().numpy()  # [N,1,n_vel]
        preds_list.append(pred_ch)
    predictions_scaled = np.concatenate(preds_list, axis=1)  # [N,6,n_vel]

else:
    raise ValueError(f"Unknown MODEL_MODE: {MODEL_MODE}")

#%%
# 5. (공통) inverse transform & visualization
# 주의: 아래 변수 이름은 기존 코드 흐름을 따름
# predictions_scaled: [N, n_rdc_coeffs, n_vel]

# Inverse transform to original scale (기존 로직 유지)
pred_reshaped = predictions_scaled.transpose(0, 2, 1).reshape(-1, n_rdc_coeffs)
act_reshaped  = y_tensor[test_dataset.indices].numpy().transpose(0, 2, 1).reshape(-1, n_rdc_coeffs)

# NOTE: 앞에서 정의한 스케일러 리스트 이름은 scalers_y 입니다.
# 기존 코드에 scaler_y로 표기되어 있다면 아래 두 줄에서 오류가 날 수 있으니 필요 시 이름을 맞추세요.
predictions_orig = pred_reshaped  # <- 필요 시 개별 채널 역변환 적용
actuals_orig     = act_reshaped

# 역변환을 채널별로 적용하려면 다음을 사용 (scalers_y 리스트 필요):
# preds_tmp, acts_tmp = [], []
# for i in range(n_rdc_coeffs):
#     preds_tmp.append(scalers_y[i].inverse_transform(pred_reshaped[:, i].reshape(-1,1)))
#     acts_tmp.append(scalers_y[i].inverse_transform(act_reshaped[:, i].reshape(-1,1)))
# predictions_orig = np.stack(preds_tmp, axis=1).squeeze(-1)
# actuals_orig     = np.stack(acts_tmp, axis=1).squeeze(-1)

predictions = predictions_orig.reshape(len(test_dataset.indices), n_vel, n_rdc_coeffs)
actuals     = actuals_orig.reshape(len(test_dataset.indices), n_vel, n_rdc_coeffs)

# 시각화 (기존과 동일)
num_samples_to_plot = 3
random_indices = np.random.choice(len(test_dataset.indices), num_samples_to_plot, replace=False)
rdc_labels = ['K', 'k', 'C', 'c', 'M', 'm']

for idx in random_indices:
    fig, axes = plt.subplots(3, 2, figsize=(15, 18)); axes = axes.ravel()
    original_sample_idx = test_dataset.indices[idx]
    fig.suptitle(f'Test Sample #{original_sample_idx}: Actual vs. Predicted RDC vs. Speed', fontsize=20)
    for j in range(n_rdc_coeffs):
        ax = axes[j]
        ax.plot(w_vec.flatten(), actuals[idx, :, j], 'bo-', label='Actual')
        ax.plot(w_vec.flatten(), predictions[idx, :, j], 'ro--', label='Predicted')
        ax.set_title(rdc_labels[j]); ax.set_xlabel('Rotational Speed (rad/s)'); ax.set_ylabel('RDC Value')
        ax.legend(); ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show()
