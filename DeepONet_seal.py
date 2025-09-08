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

# 1. Load dataset.mat files
data_dir = 'dataset/data/tapered_seal'

# mat_files = ('20250826_T_091719',)
mat_files = ('20250826_T_093534',)
# mat_files = ('20250826_T_095326',)

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
batch_size = 2**8
criterion = nn.HuberLoss()
criterion = nn.MSELoss()
epochs = 1000
hidden_channels = 2**7
n_layers = 8
p_drop = 0.0

lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
weight_decay=1e-6

import json

hyperparams = {
    "Batch size": batch_size,
    "# of hidden channels": hidden_channels,
    "# of layers": n_layers,
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
    

class SingleHeadDeepONet(nn.Module):

    def __init__(self,
                 n_params: int,
                 hidden_channels: int,
                 n_layers: int,
                 n_basis: int,
                 p_drop: float):
        """
        Classic DeepONet-style factorization for a single target (one RDC).
        y(params, w) = sum_k c_k(params) * phi_k(w) + b
        - Branch outputs n_basis + 1 coefficients (including bias)
        - Trunk outputs n_basis basis values per grid point
        Output shape: [B, L]
        """
        super().__init__()
        self.n_basis = n_basis

        # Branch: params -> coefficients [n_basis + 1]
        branch_layers = []
        in_dim = n_params
        branch_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
        for _ in range(max(1, n_layers - 1)):
            branch_layers += [nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
        branch_layers += [nn.Linear(hidden_channels, n_basis + 1)]
        self.branch = nn.Sequential(*branch_layers)

        # Trunk: grid -> basis [n_basis]
        trunk_layers = []
        in_dim = 1
        trunk_layers += [nn.Linear(in_dim, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
        for _ in range(max(1, n_layers - 1)):
            trunk_layers += [nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.Dropout(p_drop)]
        trunk_layers += [nn.Linear(hidden_channels, n_basis)]
        self.trunk = nn.Sequential(*trunk_layers)

    def forward(self, params: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # params: [B, n_params]
        # grid:   [B, L, 1]
        B, L, _ = grid.shape

        coeff = self.branch(params)  # [B, n_basis+1]
        phi = self.trunk(grid.reshape(B * L, 1)).view(B, L, self.n_basis)  # [B, L, n_basis]
        ones = torch.ones(B, L, 1, dtype=phi.dtype, device=phi.device)
        phi = torch.cat([phi, ones], dim=-1)  # [B, L, n_basis+1]

        # Contraction over basis dimension -> [B, L]
        y = torch.einsum('bn,bln->bl', coeff, phi)
        return y
    
# for seal_idx, mat_file in enumerate(mat_files):
for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    base_dir = 'net'
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, 'deeponet_seal_best_'+mat_file+'.pth')
    network_path = os.path.join(base_dir, 'deeponet_'+mat_file+'.pth')
    network_path_ts = os.path.join(base_dir, 'deeponet_'+mat_file+'.pt')

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
        n_rdc_coeffs = rdc.shape[0] # 4

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

    # Torch 텐서로 변환 (X는 공통)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # --- 각 RDC별로 별도 데이터셋/모델 학습 ---
    n_basis = hidden_channels  # 기본값으로 사용 (필요시 하이퍼파라미터화)

    # 전체 데이터 예측(원 단위)을 저장할 버퍼: [N, nRDC, L]
    preds_orig_all = np.zeros((n_data, n_rdc_coeffs, n_vel), dtype=float)

    for rdc_idx in range(n_rdc_coeffs):
        # y 스케일러 및 스케일 데이터 구성 (shape: [N, L])
        scaler_y = scalers_y[rdc_idx]
        y_i_scaled = np.empty((n_data, n_vel), dtype=float)
        for split_idx in (train_idx, val_idx, test_idx):
            y_i_scaled[split_idx, :] = scaler_y.transform(
                y_functions[split_idx, rdc_idx, :].reshape(-1, 1)
            ).reshape(-1, n_vel)

        y_i_tensor = torch.tensor(y_i_scaled, dtype=torch.float32)

        # 데이터셋 및 로더
        dataset_i = TensorDataset(X_tensor, y_i_tensor)
        from torch.utils.data import Subset
        train_dataset = Subset(dataset_i, train_idx.tolist())
        val_dataset   = Subset(dataset_i, val_idx.tolist())
        test_dataset  = Subset(dataset_i, test_idx.tolist())

        print(f"[RDC {rdc_idx}] Training set size: {len(train_dataset)}")
        print(f"[RDC {rdc_idx}] Validation set size: {len(val_dataset)}")
        print(f"[RDC {rdc_idx}] Test set size: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 모델/최적화기/스케줄러
        model = SingleHeadDeepONet(
            n_params=n_para,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_basis=n_basis,
            p_drop=p_drop
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        steps_per_epoch = math.ceil(len(train_loader))
        total_steps = steps_per_epoch * epochs
        scheduler = build_warmup_cosine(optimizer, total_steps, warmup_steps=500)

        best_val_loss = float('inf')
        start_epoch = 0
        ckpt_best_path = os.path.join(base_dir, f'deeponet_seal_best_rdc{rdc_idx}_{mat_file}.pth')
        ckpt_last_path = os.path.join(base_dir, f'deeponet_seal_last_rdc{rdc_idx}_{mat_file}.pth')

        for epoch in range(start_epoch, epochs):
            model.train(); train_loss = 0.0
            n_train = 0
            for params, functions in train_loader:
                params, functions = params.to(device), functions.to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                outputs = model(params, batch_grid)  # [B, L]
                loss = criterion(outputs, functions)

                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * params.size(0)
                n_train += params.size(0)
            train_loss /= n_train
            
            model.eval(); val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for params, functions in val_loader:
                    params, functions = params.to(device), functions.to(device)
                    batch_grid = grid_tensor.unsqueeze(0).repeat(params.size(0), 1, 1).to(device)
                    outputs = model(params, batch_grid)  # [B, L]
                    val_loss += criterion(outputs, functions).item() * params.size(0)
                    n_val    += params.size(0)
            val_loss /= n_val

            if (epoch+1) % 100 == 0 or epoch == start_epoch:
                print(f'[RDC {rdc_idx}] Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epoch': epoch
                }, ckpt_best_path)

            # Save last (full state for resume)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch
            }, ckpt_last_path)

        # 최종 저장물 (가중치 + 스케일러)
        network_path = os.path.join(base_dir, f'deeponet_rdc{rdc_idx}_{mat_file}.pth')
        network_path_ts = os.path.join(base_dir, f'deeponet_rdc{rdc_idx}_{mat_file}.pt')

        ckpt = {
            "state_dict": model.state_dict(),
            "hparams": {
                "hidden_channels": hidden_channels,
                "n_layers": n_layers,
                "n_params": n_para,
                "n_basis": n_basis,
                "p_drop": p_drop,
                "rdc_idx": rdc_idx,
            },
            "additional": {
                "w_min": w_min,
                "w_max": w_max,
            },
            # 전처리 스케일러 저장
            "scaler_X_mean": scaler_X.mean_, "scaler_X_std": scaler_X.scale_,
            "scaler_y_mean": scaler_y.mean_.ravel(),
            "scaler_y_std":  scaler_y.scale_.ravel(),
        }
        torch.save(ckpt, network_path)

        # TorchScript 저장
        import copy
        model_cpu = copy.deepcopy(model).eval().to("cpu")

        L = int(grid_tensor.shape[0])
        dummy_params = torch.zeros(1, int(X_tensor.shape[1]), dtype=torch.float32)
        dummy_grid   = torch.zeros(1, L, 1, dtype=torch.float32)

        with torch.no_grad():
            ts = torch.jit.trace(model_cpu, (dummy_params, dummy_grid), check_trace=False)
        ts.save(network_path_ts)

        # --- 이 RDC 모델로 전체 데이터 예측 수집 (원 단위) ---
        model.eval()
        with torch.no_grad():
            bs_pred = batch_size
            for s in range(0, n_data, bs_pred):
                e = min(s + bs_pred, n_data)
                params_b = X_tensor[s:e].to(device)
                batch_grid = grid_tensor.unsqueeze(0).repeat(params_b.size(0), 1, 1).to(device)
                preds_scaled = model(params_b, batch_grid).cpu().numpy()  # [B, L]
                preds_orig = scalers_y[rdc_idx].inverse_transform(
                    preds_scaled.reshape(-1, 1)
                ).reshape(-1, n_vel)
                preds_orig_all[s:e, rdc_idx, :] = preds_orig

    #%%
    # --- 통합 지표/시각화 (전체 RDC 스택) ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rdc_true = rdc.transpose([2, 0, 1])  # [N, nRDC, L]
    rdc_flat = preds_orig_all            # [N, nRDC, L]

    rdc_flat_ = np.ravel(rdc_flat)
    rdc_true_ = np.ravel(rdc_true)

    mse = mean_squared_error(rdc_true_, rdc_flat_)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(rdc_true_, rdc_flat_)
    r2 = r2_score(rdc_true_, rdc_flat_)
    yrng = (rdc_true_.max() - rdc_true_.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((rdc_true_ - rdc_flat_) / (np.abs(rdc_true_) + 1e-12)))

    print(f"[Overall] RMSE: {rmse:.6g}, MAE: {mae:.6g}, R^2: {r2:.6f}, rRMSE: {100*rrmse:.4f}%, MAPE: {100*mape:.4f}%")

    # 에러 분포/상위 샘플 시각화
    abs_err = rdc_true - rdc_flat
    rel_err = (rdc_true - rdc_flat) / (np.abs(rdc_true) + 1e-12) * 1e2

    # 1) 각 RDC별, 특정 속도 인덱스에서 상위 오차 샘플 플롯
    idx_w = -1  # 마지막 속도 포인트
    for r in range(n_rdc_coeffs):
        fig = plt.figure()
        rdc_true1 = rdc_true[:, r, idx_w]
        r_e = rel_err[:, r, idx_w]
        a_e = abs_err[:, r, idx_w]
        idx_rel_sorted = np.flip(np.argsort(np.abs(r_e)))
        r_e_sorted = r_e[idx_rel_sorted]
        a_e_sorted = a_e[idx_rel_sorted]

        plt.subplot(411)
        plt.plot(rdc_true1[:100], 'o-')
        plt.title("absolute value")
        plt.subplot(412)
        plt.plot(a_e_sorted[:100], 'o-')
        plt.title("absolute error")
        plt.subplot(413)
        plt.plot(r_e_sorted[:100], 'o-')
        plt.ylim([-100, 100])
        plt.title("relative error (%)")
        plt.subplot(414)
        plt.plot(r_e_sorted[:100], 'o-')
        plt.title("relative error (%)")
        plt.tight_layout()

    # 2) 각 샘플에서 최대 상대오차 지점만 뽑아 정렬해 보기
    for r in range(n_rdc_coeffs):
        fig = plt.figure()
        abs_err_ = abs_err[:, r]
        rel_err_ = rel_err[:, r]

        idx_rel = np.argmax(np.abs(rel_err_), axis=1)
        rows = np.arange(rel_err_.shape[0])
        rels = rel_err_[rows, idx_rel]
        abss = abs_err_[rows, idx_rel]

        idx_sorted = np.flip(np.argsort(np.abs(rels)))
        rels_sorted = rels[idx_sorted]
        abss_sorted = abss[idx_sorted]

        plt.subplot(311)
        plt.plot(abss_sorted[:100], 'o-')
        plt.title("abs error @worst-speed")
        plt.subplot(312)
        plt.plot(rels_sorted[:100], 'o-')
        plt.ylim([-100, 100])
        plt.title("rel error (%) @worst-speed")
        plt.subplot(313)
        plt.plot(rels_sorted[:100], 'o-')
        plt.title("rel error (%) @worst-speed")
        plt.tight_layout()
