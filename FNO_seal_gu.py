#%%
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt

# from torch.serialization import add_safe_globals
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import neuralop as nop
from neuralop.models import FNO
# from neuralop.layers.spectral_convolution import SpectralConv

# add_safe_globals([torch._C._nn.gelu, SpectralConv])

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_file = os.path.join(data_dir, '20250812_T_113003', 'dataset.mat')

# 파라미터 설정
batch_size = 2**16
criterion = nop.losses.LpLoss(d=1, p=2)
epochs = 5000
fno_modes = 32
fno_hidden_channels = 128
n_layers = 3
lr = 1e-3


# 데이터 로딩 및 전처리
with h5py.File(mat_file, 'r') as mat:
    # inputNond: [nPara, nData] 형상 파라미터
    input_nond = np.array(mat.get('inputNond'))
    # wVec: [1, nVel] 회전 속도 벡터 (좌표 그리드)
    w_vec = np.array(mat['params/wVec'])
    # RDC: [6, nVel, nData] 동특성 계수 (타겟 함수)
    rdc = np.array(mat.get('RDC'))

    n_para, n_data = input_nond.shape
    _, n_vel = w_vec.shape
    n_rdc_coeffs = rdc.shape[0] # 6 (K, k, C, c, M, m)

    # 입력 데이터 (X): 형상 파라미터 [nData, nPara]
    X_params = input_nond.T

    # 출력 데이터 (y): 동특성 계수 함수 [nData, nVel, nRDC]
    # FNO는 (batch, channels, grid_points) 형태를 선호하므로 [nData, nRDC, nVel]로 변경이라고 GPT가 그럤다
    y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]

    # 회전 속도 그리드: [nVel, 1]
    
    w = w_vec.squeeze()                         # [n_vel]
    w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0 # normalization
    grid = w_norm[:, None] # [nVel, 1]


# %%
# 데이터 스케일링
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_params) 

scaler_Y = StandardScaler()
channel_data = y_functions[:, 1, :].reshape(-1, 1)
y_scaled = scaler_Y.fit_transform(channel_data).reshape(n_data, n_vel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Torch 텐서로 변환
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
# y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
class ParametricFNO(nn.Module):
    """
    기존: 형상 파라미터를 조건으로 받아 함수를 예측하는 FNO 모델 (단일 네트워크)
    outputs: [B, out_channels, n_vel]
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, out_channels,n_layers):
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
            n_layers=n_layers,
            in_channels=in_channels + param_embedding_dim,
            out_channels=out_channels
        )

    def forward(self, params, grid):
        # params: [B, n_params], grid: [B, n_vel, 1]
        pe = self.param_encoder(params)                      # [B, emb]
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)    # [B, n_vel, emb]
        fno_in = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)  # [B, 1+emb, n_vel]
        out = self.fno(fno_in)  # [B, out_channels, n_vel]
        return out

optimizer = None
best_val_loss = float('inf')
base_dir = 'net'
os.makedirs(base_dir, exist_ok=True)

#%%
model = ParametricFNO(
    n_params=n_para,
    param_embedding_dim=64,
    fno_modes=fno_modes,
    fno_hidden_channels=fno_hidden_channels,
    in_channels=1,
    out_channels=1,
    n_layers=n_layers
    # out_channels=n_rdc_coeffs
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    train_loss /= len(train_dataset); val_loss /= len(val_dataset)
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'state_dict': model.state_dict()}, model_save_path)
    scheduler.step()

# --- Evaluate ---
ckpt = torch.load(model_save_path, map_location=device, weights_only=False)
sd = ckpt.get('state_dict', ckpt)
sd.pop('_metadata', None) 
missing = model.load_state_dict(sd, strict=False)
if missing.unexpected_keys:
    print("unexpected:", missing.unexpected_keys)
if missing.missing_keys:
    print("missing:", missing.missing_keys)
model.eval()

n_test_samples = len(test_dataset.indices)
test_params = X_tensor[test_dataset.indices].to(device)
grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)
with torch.no_grad():
    predictions_scaled = model(test_params, grid_repeated).cpu().numpy()
# %%
# --- Validation on Test Set ---
from sklearn.metrics import mean_squared_error, mean_absolute_error

model.eval()
test_params = X_tensor[test_dataset.indices].to(device)
test_targets = y_tensor[test_dataset.indices].to(device)
grid_repeated = grid_tensor.unsqueeze(0).repeat(len(test_dataset.indices), 1, 1).to(device)

with torch.no_grad():
    preds_scaled = model(test_params, grid_repeated).cpu().numpy()
    targets_scaled = test_targets.cpu().numpy()

# 역스케일링
preds_orig = scaler_Y.inverse_transform(preds_scaled.squeeze(1))
targets_orig = scaler_Y.inverse_transform(targets_scaled.squeeze(1))

# 성능 지표 계산
mse = mean_squared_error(targets_orig.flatten(), targets_orig.flatten())
mae = mean_absolute_error(preds_orig.flatten(), targets_orig.flatten())
rel_lp = criterion(torch.tensor(preds_scaled), torch.tensor(targets_scaled)).item()

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test Relative LpLoss: {rel_lp:.4f}")

# 샘플 시각화
n_plot = 5
plt.figure(figsize=(10, 10))
for i in range(n_plot):
    idx = i
    plt.subplot(n_plot, 1, i+1)
    plt.plot(w, targets_orig[idx], label='True')
    plt.plot(w, preds_orig[idx], '--', label='Predicted')
    if i == 0:
        plt.legend()
plt.xlabel('Velocity')
plt.ylabel('RDC Value')
plt.tight_layout()
plt.show()
# %%
