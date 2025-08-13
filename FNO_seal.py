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

#%%
# FNO 구현을 위해 neuraloperator 라이브러리를 사용합니다.
# 설치: pip install neuraloperator
import neuralop as nop
from neuralop.models import FNO

print("FNO example for predicting seal rotordynamic coefficients.")

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_file = os.path.join(data_dir, '20250812_T_113003', 'dataset.mat')

#%%
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
    grid = w_vec.T

#%%
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
# 2. Define a Parametric FNO model
class ParametricFNO(nn.Module):
    """
    형상 파라미터를 조건으로 받아 함수를 예측하는 FNO 모델
    """
    def __init__(self, n_params, param_embedding_dim, fno_modes, fno_hidden_channels, in_channels, out_channels):
        super(ParametricFNO, self).__init__()
        self.n_params = n_params
        
        # 1. 형상 파라미터를 임베딩하는 MLP
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params, param_embedding_dim),
            nn.ReLU(),
            nn.Linear(param_embedding_dim, param_embedding_dim)
        )
        
        # 2. FNO 모델
        # 입력 채널 = 좌표 채널(1) + 파라미터 임베딩 채널
        self.fno = FNO(
            n_modes=(fno_modes,), 
            hidden_channels=fno_hidden_channels,
            in_channels=in_channels + param_embedding_dim, # grid(1) + params
            out_channels=out_channels
        )

    def forward(self, params, grid):
        # params: [batch, n_params]
        # grid: [batch, n_vel, 1]

        # 파라미터 인코딩 및 브로드캐스팅
        param_embedding = self.param_encoder(params) # [batch, embedding_dim]
        param_broadcast = param_embedding.unsqueeze(1).repeat(1, grid.shape[1], 1) # [batch, n_vel, embedding_dim]

        # FNO 입력을 위해 grid와 결합
        fno_in = torch.cat([grid, param_broadcast], dim=-1) # [batch, n_vel, 1 + embedding_dim]
        
        # FNO는 (batch, channels, grid_points) 형태를 기대하므로 transpose
        fno_in = fno_in.permute(0, 2, 1)

        # FNO 연산
        fno_out = self.fno(fno_in) # [batch, out_channels, n_vel]

        return fno_out


model = ParametricFNO(
    n_params=n_para,
    param_embedding_dim=64,
    fno_modes=4, # 푸리에 모드 개수 (주파수 공간에서 얼마나 많은 모드를 사용할지)
    fno_hidden_channels=128,
    in_channels=1, # 입력은 grid 좌표(1차원)
    out_channels=n_rdc_coeffs # 출력은 RDC 계수(6개)
).to(device)

#%%
# 3. Training setup
criterion = nop.losses.LpLoss(d=1, p=2) # FNO 학습에는 L2 Loss가 효과적
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
epochs = 1000
best_val_loss = float('inf')
model_save_path = 'net/fno_seal_best_model.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

#%%
# 4. Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for params, functions in train_loader:
        params, functions = params.to(device), functions.to(device)
        
        # 모델에 맞는 grid 준비 (batch 차원 추가)
        batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)

        optimizer.zero_grad()
        outputs = model(params, batch_grid)
        loss = criterion(outputs, functions)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * params.size(0)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for params, functions in val_loader:
            params, functions = params.to(device), functions.to(device)
            batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
            outputs = model(params, batch_grid)
            loss = criterion(outputs, functions)
            val_loss += loss.item() * params.size(0)
            
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)

#%%
# 5. Evaluate the best model on the test set and visualize
# strict=False 옵션을 추가하여 모델 state_dict에 포함된 추가 메타데이터 키를 무시하도록 합니다.
# 이는 neuralop과 같은 특정 라이브러리를 사용할 때 발생할 수 있는 호환성 문제를 해결합니다.
model.load_state_dict(torch.load(model_save_path, weights_only=False), strict=False)
model.eval()

test_indices = test_dataset.indices
n_test_samples = len(test_indices)

test_params = X_tensor[test_indices].to(device)
# Get scaled actuals from y_tensor. No need to move to device, will be used on CPU.
actuals_scaled = y_tensor[test_indices].numpy()
grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)

with torch.no_grad():
    predictions_scaled = model(test_params, grid_repeated).cpu().numpy()

# Inverse transform to original scale
# Reshape for scaler: [n_samples, n_channels, n_points] -> [n_samples*n_points, n_channels]
# Note: n_rdc_coeffs is the number of channels
pred_reshaped = predictions_scaled.transpose(0, 2, 1).reshape(-1, n_rdc_coeffs)
act_reshaped = actuals_scaled.transpose(0, 2, 1).reshape(-1, n_rdc_coeffs)

predictions_orig = scaler_y.inverse_transform(pred_reshaped)
actuals_orig = scaler_y.inverse_transform(act_reshaped)

# Reshape back for plotting: [n_samples, n_points, n_channels]
predictions = predictions_orig.reshape(n_test_samples, n_vel, n_rdc_coeffs)
actuals = actuals_orig.reshape(n_test_samples, n_vel, n_rdc_coeffs)

# Visualize results for a few random samples
num_samples_to_plot = 3
random_indices = np.random.choice(n_test_samples, num_samples_to_plot, replace=False)

rdc_labels = ['K', 'k', 'C', 'c', 'M', 'm']

for i in random_indices:
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.ravel()
    original_sample_idx = test_indices[i]
    fig.suptitle(f'Test Sample #{original_sample_idx}: Actual vs. Predicted RDC vs. Speed', fontsize=20)
    
    for j in range(n_rdc_coeffs):
        ax = axes[j]
        ax.plot(w_vec.flatten(), actuals[i, :, j], 'bo-', label='Actual')
        ax.plot(w_vec.flatten(), predictions[i, :, j], 'ro--', label='Predicted')
        ax.set_title(rdc_labels[j])
        ax.set_xlabel('Rotational Speed (rad/s)')
        ax.set_ylabel('RDC Value')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
