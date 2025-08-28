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

# add_safe_globals([torch._C._nn.gelu])
# add_safe_globals([nop.layers.spectral_convolution.SpectralConv])

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_file = os.path.join(data_dir, '20250812_T_113003', 'dataset.mat')

# 파라미터 설정
batch_size = 2**10
criterion = nop.losses.LpLoss(d=1, p=2)
epochs = 2000
param_embedding_dim = 64
fno_modes = 4
fno_hidden_channels = 128
n_layers = 3
shared_out_channels = fno_hidden_channels
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
weight_decay=1e-4

import json

hyperparams = {
    "Batch size": batch_size,
    "Parameter embedding dimension": param_embedding_dim,
    "# of FNO modes": fno_modes,
    "# of FNO hidden channels": fno_hidden_channels,
    "# of FNO layers": n_layers,
    "# of shared output channels": shared_out_channels,
    "Learning rate": f"{lr:.1e}"
}

print(json.dumps(hyperparams, indent=2))

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
    # FNO는 (batch, channels, grid_points) 형태를 선호하므로 [nData, nRDC, nVel]로 변경이라고 GPT가 그럔다
    y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]

    # 회전 속도 그리드: [nVel, 1]
    
    w = w_vec.squeeze()                         # [n_vel]
    w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0 # normalization
    grid = w_norm[:, None] # [nVel, 1]


# %%
indices = np.arange(n_data)
train_size = int(n_data*0.7); val_size = int(n_data*0.15)
test_size = n_data - train_size - val_size
train_idx, val_idx, test_idx = np.split(np.random.permutation(indices),
                                        [train_size, train_size+val_size])

scaler_X = StandardScaler().fit(X_params[train_idx])
X_scaled = np.empty_like(X_params, dtype=float)
X_scaled[train_idx] = scaler_X.transform(X_params[train_idx])
X_scaled[val_idx]  = scaler_X.transform(X_params[val_idx])
X_scaled[test_idx] = scaler_X.transform(X_params[test_idx])

scalers_y = [StandardScaler().fit(y_functions[train_idx, i, :].reshape(-1,1))
             for i in range(n_rdc_coeffs)]

y_scaled = np.empty_like(y_functions, dtype=float)
for i in range(n_rdc_coeffs):
    for split_idx in (train_idx, val_idx, test_idx):
        y_scaled[split_idx, i, :] = scalers_y[i].transform(
            y_functions[split_idx, i, :].reshape(-1,1)
        ).reshape(-1, y_functions.shape[-1])

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    
class MultiHeadParametricFNO(nn.Module):
    """
    FNO 본체는 공유하고, 채널별 1x1 Conv1d 헤드를 분리하는 멀티헤드 구조.
    outputs: [B, n_heads(=n_rdc_coeffs), n_vel]
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, n_heads,n_layers, shared_out_channels):
        super().__init__()
        self.n_params = n_params
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
                # depth 2
                nn.Conv1d(shared_out_channels, shared_out_channels // 2, 1),
                nn.GELU(),
                nn.BatchNorm1d(shared_out_channels // 2),
                nn.Dropout(0.1),
                # output
                nn.Conv1d(shared_out_channels // 2, 1, 1)
            ) for _ in range(n_heads)
        ])

    def forward(self, params, grid):
        pe = self.param_encoder(params)                       # [B, emb]
        pe = pe.unsqueeze(1).repeat(1, grid.shape[1], 1)     # [B, n_vel, emb]
        x = torch.cat([grid, pe], dim=-1).permute(0, 2, 1)   # [B, 1+emb, n_vel]
        feat = self.trunk(x)                                  # [B, Csh, n_vel]
        outs = [head(feat) for head in self.heads]            # each: [B,1,n_vel]
        return torch.cat(outs, dim=1)                         # [B, n_heads, n_vel]

optimizer = None
best_val_loss = float('inf')
base_dir = 'test_dir'
os.makedirs(base_dir, exist_ok=True)

#%%

model = MultiHeadParametricFNO(
    n_params=n_para,
    param_embedding_dim=param_embedding_dim,
    fno_modes=fno_modes,
    fno_hidden_channels=fno_hidden_channels,
    in_channels=1,
    n_heads=n_rdc_coeffs,
    n_layers=n_layers,
    shared_out_channels=shared_out_channels
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    train_loss /= len(train_dataset); val_loss /= len(val_dataset)
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'state_dict': model.state_dict()}, model_save_path)
    scheduler.step()
    
#%%
network_path = os.path.join(base_dir, "fno_multihead.pth")
# 저장
ckpt = {
    "state_dict": model.state_dict(),
    "hparams": {
        "param_embedding_dim": param_embedding_dim,
        "fno_modes": fno_modes,
        "fno_hidden_channels": fno_hidden_channels,
        "n_layers": n_layers,
        "shared_out_channels": shared_out_channels,
        "n_params": n_para,
        "n_heads": n_rdc_coeffs,
    },
    # 전처리 스케일러도 같이 저장하면 편함
    "scaler_X_mean": scaler_X.mean_, "scaler_X_std": scaler_X.scale_,
    "scalers_y_mean": [s.mean_.ravel() for s in scalers_y],
    "scalers_y_std":  [s.scale_.ravel() for s in scalers_y],
}
torch.save(ckpt, network_path)
