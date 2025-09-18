#%%
import os
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
import math


# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)

# 파라미터 설정
batch_size = 2**9
criterion = nn.MSELoss()
epochs = 5000
param_embedding_dim = 2**6
hidden_channels = 2**6
n_layers = 8
shared_out_channels = hidden_channels

lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
weight_decay=1e-6

import json

hyperparams = {
    "Batch size": batch_size,
    "Parameter embedding dimension": param_embedding_dim,
    "# of hidden channels": hidden_channels,
    "# of layers": n_layers,
    "# of shared output channels": shared_out_channels,
    "Learning rate": f"{lr:.1e}",
}

print(json.dumps(hyperparams, indent=2))

class EarlyStopping:
    def __init__(self, patience=100, minimize=True):
        self.best = math.inf if minimize else -math.inf
        self.minimize = minimize
        self.patience = patience
        self.wait = 0
        self.best_state = None

    def step(self, metric, model):
        improved = (metric < self.best) if self.minimize else (metric > self.best)
        if improved:
            self.best = metric
            self.wait = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False  # not early stop
        else:
            self.wait += 1
            return self.wait > self.patience

def build_warmup_cosine(optimizer, total_steps, warmup_steps=500):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    
class MultiHeadDeepONet(nn.Module):

    def __init__(self,
                n_params: int,
                param_embedding_dim: int,
                hidden_channels: int,
                n_heads: int,
                n_layers: int,
                n_basis: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_basis = n_basis


        # --- Branch: params -> coefficients for each head (p bases + bias)
        branch_layers = []
        in_dim = n_params
        # 입력 임베딩
        branch_layers += [nn.Linear(in_dim, param_embedding_dim), nn.GELU()]
        in_dim = param_embedding_dim
        # 히든 블록
        for _ in range(max(1, n_layers - 1)):
            branch_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU()]
            in_dim = hidden_channels
        # 각 head마다 (p + 1)개 계수(편향 포함)
        branch_layers += [nn.Linear(in_dim, n_heads * (n_basis + 1))]
        self.branch = nn.Sequential(*branch_layers)

        # --- Trunk: grid point -> basis functions phi(w) of length p
        trunk_layers = []
        in_dim = 1
        trunk_layers += [nn.Linear(in_dim, param_embedding_dim), nn.GELU()]
        in_dim = param_embedding_dim
        for _ in range(max(1, n_layers - 1)):
            trunk_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU()]
            in_dim = hidden_channels
        trunk_layers += [nn.Linear(in_dim, n_basis)]
        self.trunk = nn.Sequential(*trunk_layers)

    def forward(self, params: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # params: [B, n_params]
        # grid:   [B, L, 1]
        B, L, _ = grid.shape

        # Branch 계수: [B, n_heads, n_basis+1]
        coeff = self.branch(params).view(B, self.n_heads, self.n_basis + 1)

        # Trunk 기저: [B, L, n_basis], 편향항 위해 ones 추가 -> [B, L, n_basis+1]
        phi = self.trunk(grid.reshape(B * L, 1)).view(B, L, self.n_basis)
        ones = torch.ones(B, L, 1, dtype=phi.dtype, device=phi.device)
        phi = torch.cat([phi, ones], dim=-1)  # bias basis

        # 수축: y[b,h,l] = sum_k coeff[b,h,k] * phi[b,l,k]
        y = torch.einsum('bhn,bln->bhl', coeff, phi)  # [B, n_heads, L]
        return y
    
for seal_idx, mat_file in enumerate(mat_files):
# for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    base_dir = 'net'
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, 'deeponet_seal_best_multihead_'+mat_file+'.pth')
    network_path = os.path.join(base_dir, 'deeponet_multihead_'+mat_file+'.pth')
    network_path_ts = os.path.join(base_dir, 'deeponet_multihead_'+mat_file+'.pt')
    loss_hist_path_png = os.path.join('deeponet_multihead_'+mat_file+'.png')

    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # inputNond: [nPara, nData] 형상 파라미터
        # input_nond = np.array(mat.get('inputNond'))
        input_ = np.array(mat.get('input'))
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        # RDC: [6, nVel, nData] 동특성 계수 (타겟 함수)
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:] # no mass
        n_para, n_data = input_.shape
        _, n_vel = w_vec.shape
        n_rdc_coeffs = rdc.shape[0] # 6 

        # 입력 데이터 (X): 형상 파라미터 [nData, nPara]
        X_params = input_.T

        # 출력 데이터 (y): 동특성 계수 함수 [nData, nVel, nRDC]
        # FNO는 (batch, channels, grid_points) 형태를 선호하므로 [nData, nRDC, nVel]로 변경이라고 GPT가 그럔다
        y_functions = rdc.transpose(2, 0, 1) # [nData, nRDC, nVel]
        # 회전 속도 그리드: [nVel, 1]
        w = w_vec.squeeze()                         # [n_vel]
        w_norm = 2 * (w - w.min()) / (w.max()-w.min()) - 1.0 # normalization
        grid = w_norm[:, None] # [nVel, 1]
        # grid = w[:, None] # [nVel, 1]

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

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset   = Subset(dataset, val_idx.tolist())
    test_dataset  = Subset(dataset, test_idx.tolist())

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = None
    best_val_loss = float('inf')
    
    model = MultiHeadDeepONet(
        n_params=n_para,
        param_embedding_dim=param_embedding_dim,
        hidden_channels=hidden_channels,
        n_heads=n_rdc_coeffs,
        n_layers=n_layers,
        n_basis=shared_out_channels,   # 기존 shared_out_channels를 기저 개수로 사용
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    steps_per_epoch = math.ceil(len(train_loader))
    total_steps = steps_per_epoch * epochs
    scheduler = build_warmup_cosine(optimizer, total_steps, warmup_steps=500)

    early = EarlyStopping(patience=100, minimize=True)

    best_val_loss = float('inf')
    start_epoch = 0
    ckpt_best_path = os.path.join(base_dir, f'deeponet_seal_best_multihead_{mat_file}.pth')
    ckpt_last_path = os.path.join(base_dir, f'deeponet_seal_last_multihead_{mat_file}.pth')

    train_losses, val_losses = [], []
    for epoch in range(start_epoch, epochs):
        model.train(); train_loss = 0.0
        n_train = 0
        for params, functions in train_loader:
            params, functions = params.to(device), functions.to(device)
            batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
            outputs = model(params, batch_grid)
            loss = criterion(outputs, functions)
            
            optimizer.zero_grad()
            loss.backward(); 
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * params.size(0)
            n_train += params.size(0)
        train_loss /= n_train
        train_losses.append(train_loss)
        
        model.eval(); val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for params, functions in val_loader:
                params, functions = params.to(device), functions.to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                outputs = model(params, batch_grid)
                val_loss += criterion(outputs, functions).item() * params.size(0)
                n_val    += params.size(0)
        val_loss /= n_val
        val_losses.append(val_loss)

        if (epoch+1) % 100 == 0 or epoch == start_epoch:
            print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                # dataset split indices
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'loss_train': train_losses,
                'loss_val': val_losses,
            }, ckpt_best_path)
    # 저장
    ckpt = {
        "state_dict": model.state_dict(),
        "hparams": {
            "param_embedding_dim": param_embedding_dim,
            "hidden_channels": hidden_channels,
            "n_layers": n_layers,
            "shared_out_channels": shared_out_channels,
            "n_params": n_para,
            "n_heads": n_rdc_coeffs,
        },
        "additional": {
            "w_min": w_min,
            "w_max": w_max,
        },
        # dataset split indices for reproducibility
        "splits": {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        },
        "train_history": {  
            'loss_train': train_losses,
            'loss_val': val_losses,
        },
        # 전처리 스케일러도 같이 저장하면 편함
        "scaler_X_mean": scaler_X.mean_, "scaler_X_std": scaler_X.scale_,
        "scalers_y_mean": [s.mean_.ravel() for s in scalers_y],
        "scalers_y_std":  [s.scale_.ravel() for s in scalers_y],
    }
    torch.save(ckpt, network_path)
    
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.yscale('log', base=10)
    plt.savefig(loss_hist_path_png,dpi=600,bbox_inches="tight")
    plt.show()