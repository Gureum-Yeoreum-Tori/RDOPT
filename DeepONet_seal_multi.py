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
from torch.nn.utils import clip_grad_norm_

# from neuralop.layers.spectral_convolution import SpectralConv

# add_safe_globals([torch._C._nn.gelu])
# add_safe_globals([nop.layers.spectral_convolution.SpectralConv])

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'
# mat_file = '20250812_T_113003' # (K, k, C, c, M, m)
# mat_file = '20250819_T_123001' # (M, m, C, c, K, k) 왜 다시 뒤집혔지
# mat_file = '20250819_T_123831'
# mat_file = '20250819_T_124655'

# mat_files = ('20250819_T_123001', '20250819_T_123831', '20250819_T_124655',)
# mat_files = ('20250825_T_120952',)
# mat_files = ('20250825_T_123550',)
# mat_files = ('20250825_T_125136',)
# mat_files = ('20250825_T_120952','20250825_T_123550','20250825_T_125136',)
# mat_files = ('20250826_T_091719','20250826_T_093534','20250826_T_095326',)

# mat_files = ('20250826_T_091719',)
# mat_files = ('20250826_T_093534',)
# mat_files = ('20250826_T_095326',)

# mat_files = ('20250908_T_154540','20250908_T_155223','20250908_T_155858',)
# mat_files = ('20250908_T_182846','20250908_T_183632','20250908_T_203220',)
# mat_files = ('20250911_T_091324',)
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)

# 파라미터 설정 (기존)
# batch_size = 2**10
# criterion = nn.MSELoss()
# epochs = 5000
# param_embedding_dim = 2**8
# hidden_channels = 2**8
# n_layers = 8
# shared_out_channels = hidden_channels
# p_drop = 0

# lr = 1e-5
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# weight_decay=1e-5

# 파라미터 설정
batch_size = 2**9
# criterion = nn.HuberLoss()
criterion = nn.MSELoss()
epochs = 5000
# param_embedding_dim = 2**6
# hidden_channels = 2**6
# n_layers = 6
param_embedding_dim = 2**6
hidden_channels = 2**6
n_layers = 8
shared_out_channels = hidden_channels
p_drop = 0.0

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
    "p_drop": p_drop,
}

print(json.dumps(hyperparams, indent=2))

#%%
import math
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
                n_basis: int,
                p_drop: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_basis = n_basis
        self.p_drop = p_drop

        # --- Branch: params -> coefficients for each head (p bases + bias)
        branch_layers = []
        in_dim = n_params
        # 입력 임베딩
        branch_layers += [nn.Linear(in_dim, param_embedding_dim), nn.GELU(), nn.Dropout(p_drop)]
        in_dim = param_embedding_dim
        # 히든 블록
        for _ in range(max(1, n_layers - 1)):
            branch_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
            in_dim = hidden_channels
        # 각 head마다 (p + 1)개 계수(편향 포함)
        branch_layers += [nn.Linear(in_dim, n_heads * (n_basis + 1))]
        self.branch = nn.Sequential(*branch_layers)

        # --- Trunk: grid point -> basis functions phi(w) of length p
        trunk_layers = []
        in_dim = 1
        trunk_layers += [nn.Linear(in_dim, param_embedding_dim), nn.GELU(), nn.Dropout(p_drop)]
        in_dim = param_embedding_dim
        for _ in range(max(1, n_layers - 1)):
            trunk_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
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
    loss_hist_path_eps = os.path.join('deeponet_multihead_'+mat_file+'.eps')

    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # inputNond: [nPara, nData] 형상 파라미터
        input_nond = np.array(mat.get('inputNond'))
        # input: [nPara, nData] [*1e6 *1e6 *1e1]
        input_ = np.array(mat.get('input'))
        # wVec: [1, nVel] 회전 속도 벡터 (좌표 그리드)
        w_vec = np.array(mat['params/wVec'])
        w_min = w_vec[0,0]*30/np.pi
        w_max = w_vec[0,-1]*30/np.pi
        # RDC: [6, nVel, nData] 동특성 계수 (타겟 함수)
        rdc = np.array(mat.get('RDC'))
        rdc = rdc[2:6,:,:] # no mass
        # rdc = rdc[2:4,:,:] # no mass
        # rdc = rdc[2:3,:,:] # no mass
        # rdc = rdc[3,:,:][None,:,:] # no mass

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
        p_drop=p_drop
    ).to(device)

    # model = MultiHeadParametricMLP(
    #     n_params=n_para,
    #     param_embedding_dim=param_embedding_dim,
    #     hidden_channels=hidden_channels,
    #     in_channels=1,
    #     n_heads=n_rdc_coeffs,
    #     n_layers=n_layers,
    #     shared_out_channels=shared_out_channels
    # ).to(device)

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

    # --- Resume if last checkpoint exists ---
    # if os.path.exists(ckpt_last_path):
    #     ckpt_last = torch.load(ckpt_last_path, map_location=device, weights_only=False)

    #     # Accept either 'model_state_dict' (our last) or 'state_dict' (older best)
    #     sd = ckpt_last.get('model_state_dict', ckpt_last.get('state_dict', ckpt_last))
    #     if not isinstance(sd, dict):
    #         raise RuntimeError('[Resume] Invalid state dict structure in checkpoint')
    #     # Remove torch serialization metadata that confuses load_state_dict
    #     sd.pop('_metadata', None)

    #     incompatible = model.load_state_dict(sd, strict=False)
    #     if getattr(incompatible, 'missing_keys', None) or getattr(incompatible, 'unexpected_keys', None):
    #         print('[state_dict] missing:', getattr(incompatible, 'missing_keys', []))
    #         print('[state_dict] unexpected:', getattr(incompatible, 'unexpected_keys', []))

    #     # Optimizer / scheduler might be absent in best-only checkpoints
    #     if 'optimizer_state_dict' in ckpt_last:
    #         optimizer.load_state_dict(ckpt_last['optimizer_state_dict'])
    #     if 'scheduler_state_dict' in ckpt_last:
    #         scheduler.load_state_dict(ckpt_last['scheduler_state_dict'])

    #     start_epoch = int(ckpt_last.get('epoch', -1)) + 1
    #     best_val_loss = float(ckpt_last.get('best_val_loss', float('inf')))
    #     print(f"[Resume] Loaded checkpoint at epoch {start_epoch} (best_val={best_val_loss:.6f})")
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
            # clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * params.size(0)
            n_train += params.size(0)
        train_loss /= n_train
        train_losses.append(train_loss)
        # scheduler.step()
        
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

        # Save last (full state for resume)
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'best_val_loss': best_val_loss,
        #     'epoch': epoch,
        #     # dataset split indices
        #     'train_idx': train_idx.tolist(),
        #     'val_idx': val_idx.tolist(),
        #     'test_idx': test_idx.tolist(),
        # }, ckpt_last_path)

        
    
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
            "p_drop": p_drop,
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
    plt.savefig(loss_hist_path_eps,bbox_inches="tight")

    ## save torch script
    plt.show()
    import copy
    model_cpu = copy.deepcopy(model).eval().to("cpu")

    L = int(grid_tensor.shape[0])  # 고정 길이(트레이스 시점에 고정됨)
    dummy_params = torch.zeros(1, int(X_tensor.shape[1]), dtype=torch.float32)  # [B, n_params]
    dummy_grid   = torch.zeros(1, L, 1, dtype=torch.float32)                     # [B, L, 1]

    with torch.no_grad():
        ts = torch.jit.trace(model_cpu, (dummy_params, dummy_grid), check_trace=False)

    ts.save(network_path_ts)

    # # --- Evaluate ---
    # model.eval() # 모델을 추론 모드로 바꿈

    # n_test_samples = len(test_dataset.indices)
    # test_params = X_tensor[test_dataset.indices].to(device)
    # grid_repeated = grid_tensor.unsqueeze(0).repeat(n_test_samples, 1, 1).to(device)

    # with torch.no_grad(): # 예측하는 부분인듯
    #     preds_scaled = model(test_params, grid_repeated).cpu().numpy()
    #     targets_scaled = y_tensor[test_dataset.indices].cpu().numpy()

    # preds_tmp, targets_tmp = [], []
    # for i in range(n_rdc_coeffs):
    #     preds_tmp.append(  scalers_y[i].inverse_transform(preds_scaled[:, i, :]) )
    #     targets_tmp.append(scalers_y[i].inverse_transform(targets_scaled[:, i, :]) )

    # preds_orig   = np.stack(preds_tmp,   axis=1)
    # targets_orig = np.stack(targets_tmp, axis=1)

    # # 샘플 시각화
    # import matplotlib.colors as mcolors
    # mcolors_list = list(mcolors.TABLEAU_COLORS.values())  # HEX 값 리스트
    
    # rdc_labels = ['c',]
    # rdc_units = ['N s/m',]
        
    # # n_plot = 4
    
    # # j = 0  # 보고 싶은 계수 인덱스 (0~n_rdc_coeffs-1)
    # # fig, ax = plt.subplots(figsize=(8, 6))

    # # for idx in range(n_plot):
    # #     color = mcolors_list[idx % len(mcolors_list)]
    # #     # ax.plot(w, targets_orig[idx, j, :], color=color, linestyle='-',
    # #     #         label=f"True, #{test_dataset.indices[idx]}")
    # #     ax.plot(w, preds_orig[idx, j, :], color=color, linestyle='--', marker='o', markersize=3,
    # #             label=f"Pred, #{test_dataset.indices[idx]}")

    # # ax.set_xlabel("Rotational speed [rad/s]")
    # # ax.set_ylabel(rdc_units[j])
    # # ax.set_title(rdc_labels[j])
    # # ax.grid(True)
    # # # 필요하면 legend 추가
    # # ax.legend(ncol=2, fontsize=8)

    # # plt.tight_layout()
    # # plt.show()
    

    # rdc_labels = ['C', 'c', 'K', 'k']
    # rdc_units = ['N s/m', 'N s/m', 'N/m', 'N/m']
        
    # n_plot = 4
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # axes = axes.flatten()  # 2D -> 1D 배열로 변환

    # for j in range(n_rdc_coeffs):
    #     ax = axes[j]
    #     for idx in range(n_plot):
    #         color = mcolors_list[idx % len(mcolors_list)]
    #         ax.plot(w, targets_orig[idx, j, :], color=color, linestyle='-', 
    #                 label=f"True, #{test_dataset.indices[idx]}")
    #         ax.plot(w, preds_orig[idx, j, :], color=color, linestyle='--', marker='o', markersize=3, 
    #                 label=f"Pred, #{test_dataset.indices[idx]}")
    #     ax.set_xlabel('Rotational speed [rad/s]')
    #     ax.set_ylabel(f"{rdc_units[j]}")
    #     ax.set_title(f"{rdc_labels[j]}")
    #     ax.grid(True)
    #     ax.legend()

    # plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    # plt.show()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # for coeff_idx, label in enumerate(rdc_labels):
    #     y_true = np.ravel(targets_orig[:, coeff_idx, :])
    #     y_pred = np.ravel(preds_orig[:, coeff_idx, :])

    #     mse = mean_squared_error(y_true, y_pred)
    #     rmse = np.sqrt(mse)
    #     mae = mean_absolute_error(y_true, y_pred)
    #     r2 = r2_score(y_true, y_pred)
    #     yrng = (y_true.max() - y_true.min())
    #     rrmse = rmse / (yrng + 1e-12)
    #     mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12)))

    #     print(f"[{label}] RMSE: {rmse:.6g}, MAE: {mae:.6g}, "
    #         f"R^2: {r2:.6f}, rRMSE: {100*rrmse:.4f}%, MAPE: {100*mape:.4f}%")

    # # 전체 지표
    # y_true_all = np.ravel(targets_orig)
    # y_pred_all = np.ravel(preds_orig)
    # mse = mean_squared_error(y_true_all, y_pred_all)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_true_all, y_pred_all)
    # r2 = r2_score(y_true_all, y_pred_all)
    # yrng = (y_true_all.max() - y_true_all.min())
    # rrmse = rmse / (yrng + 1e-12)
    # mape = np.mean(np.abs((y_true_all - y_pred_all) / (np.abs(y_true_all) + 1e-12)))

    # print(f"[Overall] RMSE: {rmse:.6g}, MAE: {mae:.6g}, "
    #     f"R^2: {r2:.6f}, rRMSE: {100*rrmse:.4f}%, MAPE: {100*mape:.4f}%")

    # import time

    # test_params = X_tensor[test_dataset.indices].to(device)
    # test_targets = y_tensor[test_dataset.indices].to(device)
    # grid_repeated = grid_tensor.unsqueeze(0).repeat(len(test_dataset.indices), 1, 1).to(device)

    # # 예측 시간 측정
    # with torch.no_grad():
    #     torch.cuda.synchronize()  # GPU 시간 측정 전 동기화
    #     start_time = time.time()

    #     preds_scaled = model(test_params, grid_repeated)

    #     torch.cuda.synchronize()
    #     end_time = time.time()

    # print(f"Inference time for {len(test_dataset)} samples: {end_time - start_time:.6f} seconds")
    # print(f"Average per sample: {(end_time - start_time)/len(test_dataset):.6f} seconds")

    #%%
    from loader_brg_seal import SealDONModel
    model_seal = SealDONModel(device=device)

    X = input_.transpose()
    pop = X.shape[0]
    n_w = w.shape[0]
    rdc_flat  = model_seal.predict(seal_idx+1, X, w).reshape(pop, 1, 4, n_w).squeeze()
    rdc_true = rdc.transpose([2,0,1])
    
    rdc_flat_ = np.ravel(rdc_flat)
    rdc_true_ = np.ravel(rdc_true)
    
    mse = mean_squared_error(rdc_true_, rdc_flat_)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(rdc_true_, rdc_flat_)
    r2 = r2_score(rdc_true_, rdc_flat_)
    yrng = (rdc_true_.max() - rdc_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((rdc_true_ - rdc_flat_) / (np.abs(rdc_true_) + 1e-12)))

    print(f"[Overall] RMSE: {rmse:.6g}, MAE: {mae:.6g}, "
        f"R^2: {r2:.6f}, rRMSE: {100*rrmse:.4f}%, MAPE: {100*mape:.4f}%")


    abs_err = rdc_true-rdc_flat
    rel_err = (rdc_true-rdc_flat)/np.abs(rdc_true)*1e2

    for r in range(4):
        fig = plt.figure()
        
        abs_err_ = abs_err[:,r]
        rel_err_ = rel_err[:,r]
        
        idx_rel_sorted = np.argmax(np.abs(rel_err_),axis=1)
        rows = np.arange(rel_err_.shape[0])
        rels = rel_err_[rows, idx_rel_sorted]
        abss = abs_err_[rows, idx_rel_sorted]
        
        idx_rel_sorted = np.flip(np.argsort(np.abs(rels)))
        relss = rels[idx_rel_sorted]
        absss = abss[idx_rel_sorted]
        
        
        plt.subplot(311)
        plt.plot(absss[:100],'o-')
        plt.title("absoulte error")
        plt.subplot(312)
        plt.plot(relss[:100],'o-')
        plt.ylim([-100, 100])
        plt.title("relative error")
        plt.subplot(313)
        plt.plot(relss[:100],'o-')
        plt.title("relative error")
        plt.tight_layout()
        
        
        
    abs_err = rdc_true-rdc_flat
    rel_err = (rdc_true-rdc_flat)/np.abs(rdc_true)*1e2
    idx_w = -1
    for r in range(4):
        fig = plt.figure()
        rdc_true1 = rdc_true[:,r,idx_w]
        r_e = rel_err[:,r,idx_w]
        a_e = abs_err[:,r,idx_w]
        idx_rel_sorted = np.flip(np.argsort(np.abs(r_e)))
        r_e_sorted = r_e[idx_rel_sorted]
        a_e_sorted = a_e[idx_rel_sorted]

        plt.subplot(411)
        plt.plot(rdc_true1[:100],'o-')
        plt.title("absoulte value")
        plt.subplot(412)
        plt.plot(a_e_sorted[:100],'o-')
        plt.title("absoulte error")
        plt.subplot(413)
        plt.plot(r_e_sorted[:100],'o-')
        plt.ylim([-100, 100])
        plt.title("relative error")
        plt.subplot(414)
        plt.plot(r_e_sorted[:100],'o-')
        plt.title("relative error")
        plt.tight_layout()
        
        
        # plt.subplot(311)
        # plt.plot(abs_err[:,r,idx_w])
        # plt.subplot(312)
        # plt.plot(rel_err[:,r,idx_w])
        # plt.subplot(313)
        # plt.plot(rdc_true[:,r,idx_w])
        # plt.plot(rdc_flat[:,r,idx_w])
        # fig.show()


# %%
