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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
# 파라미터 설정
# 파라미터 설정
batch_size = 2**9
criterion = nn.MSELoss()
epochs = 2500
hidden_channels = 2**6
n_layers = 4
p_drop=0.0

lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
weight_decay=1e-6

import json

hyperparams = {
    "Batch size": batch_size,
    "# of hidden channels": hidden_channels,
    "# of layers": n_layers,
    "p_drop": p_drop,
    "Learning rate": f"{lr:.1e}",
}

print(json.dumps(hyperparams, indent=2))

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_channels, n_layers, p_drop):
        super().__init__()

        layers = []
        # 입력층
        layers.append(nn.Linear(in_dim, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p_drop))

        # 히든층
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_drop))

        # 출력층
        layers.append(nn.Linear(hidden_channels, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    base_dir = 'net'
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, 'mlp_leak_best_'+mat_file+'.pth')
    network_path = os.path.join(base_dir, 'mlp_leak_'+mat_file+'.pth')
    network_path_ts = os.path.join(base_dir, 'mlp_leak_'+mat_file+'.pt')

    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # inputNond: [nPara, nData] 형상 파라미터
        # input_nond = np.array(mat.get('inputNond'))
        input_ = np.array(mat.get('input')) # 실차원 기준
        leak = np.array(mat.get('Leak'))
        Leak = leak[6,:].reshape(-1,1)
        # Leak = np.average(leak,0).reshape(-1,1)
        n_data = Leak.shape[0]

        X_params = input_.T

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

    scaler_y = StandardScaler().fit(Leak[train_idx])            # (n_train, 1)
    y_scaled = np.empty_like(Leak, dtype=np.float32)            # (n_data, 1)
    y_scaled[train_idx] = scaler_y.transform(Leak[train_idx])   # OK
    y_scaled[val_idx]   = scaler_y.transform(Leak[val_idx])
    y_scaled[test_idx]  = scaler_y.transform(Leak[test_idx])

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.12)
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

    model = SimpleMLP(
        in_dim=3, 
        out_dim=1, 
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        p_drop=p_drop,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    start_epoch = 0

    hist_train, hist_val = [], []
    for epoch in range(start_epoch, epochs):
        model.train(); train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)                # [B, in], [B,1]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)                                     # [B,1]
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()*xb.size(0)
            n_train += xb.size(0)
        train_loss /= n_train
        
        model.eval(); val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()*xb.size(0)
                n_val    += xb.size(0)
        val_loss /= n_val
        
        hist_train.append(train_loss)
        hist_val.append(val_loss)

        if (epoch+1) % 100 == 0 or epoch == start_epoch:
            print(f'Epoch {epoch+1}/{epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')

        # Save best
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val": best_val_loss,
                "epoch": epoch,
                "p_drop": p_drop,
                "scaler_X_mean": scaler_X.mean_, "scaler_X_std": scaler_X.scale_,
                "scaler_y_mean": scaler_y.mean_, "scaler_y_std": scaler_y.scale_,
                
                # dataset split indices
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist(),
                'test_idx': test_idx.tolist(),
            }, "net/mlp_leak_best.pth")

        scheduler.step()
    
    # 저장
    ckpt = {
        "state_dict": model.state_dict(),
        "hparams": {
            "hidden_channels": hidden_channels,
            "n_layers": n_layers,
            "p_drop": p_drop,
        },
        "splits": {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        },
        # 전처리 스케일러도 같이 저장하면 편함
        "scaler_X_mean": scaler_X.mean_, "scaler_X_std": scaler_X.scale_,
        # "scaler_y_mean": scaler_y.mean_.ravel(), "scaler_y_std": scaler_y.scale_.ravel(),
        "scaler_y_mean": scaler_y.mean_, "scaler_y_std": scaler_y.scale_
    }
    torch.save(ckpt, network_path)
    
    plt.figure()
    plt.plot(hist_train, label='train')
    plt.plot(hist_val,   label='val')
    plt.yscale('log')             # 손실 스케일 차이 크면 로그가 가독성↑
    plt.xlabel('epoch'); plt.ylabel('MSE')
    plt.title('Training / Validation Loss')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout(); plt.show()

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        X_te = X_tensor[test_idx].to(device)
        y_te = y_tensor[test_idx].to(device)
        y_pred_scaled = model(X_te).cpu().numpy()
        y_true_scaled = y_te.cpu().numpy()
        
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)
    
    res = (y_pred - y_true).ravel()
    for k, name in enumerate(['hIn','hOut','psr']):
        plt.figure(); plt.scatter(X_params[test_idx,k], res, s=8, alpha=0.5)
        plt.xlabel(name); plt.ylabel('residual'); plt.grid(True, linestyle=':')
        plt.tight_layout(); plt.show()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    yrng = (y_true.max() - y_true.min())
    rrmse = rmse / (yrng + 1e-12)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12)))
    
    # --- 3) 파리티 플롯 (y_pred vs y_true) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1:1 기준선
    mn = np.nanmin([y_true, y_pred])
    mx = np.nanmax([y_true, y_pred])
    ax[0].plot([mn, mx], [mn, mx], linestyle='--', linewidth=1, label=r'$y_{\mathrm{pred}}=y_{\mathrm{true}}$')

    # 산점도
    ax[0].scatter(y_true[:,-1], y_pred[:,-1], s=12, alpha=0.6, label='val samples')

    # 선형 보정선 y = a x + b (가이드)
    a, b = np.polyfit(y_true[:,-1], y_pred[:,-1], 1)
    ax[0].plot([mn, mx], [a*mn + b, a*mx + b], linewidth=1.2, label=rf'$y={a:.3f}x{b:+.3f}$')

    ax[0].set_xlabel(r'$y_{\mathrm{true}}$')
    ax[0].set_ylabel(r'$y_{\mathrm{pred}}$')
    ax[0].set_title('Parity plot (Validation)')
    ax[0].legend(loc='best')
    ax[0].grid(True, linestyle=':', linewidth=0.6)

    # 지표 텍스트(LaTeX)
    txt = '\n'.join([
        rf'$R^2={r2:.4f}$',
        rf'$\mathrm{{RMSE}}={rmse:.3g}$',
        rf'$\mathrm{{MAE}}={mae:.3g}$',
        rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
        rf'$\mathrm{{MAPE}}={100*mape:.2f}\%$'
    ])
    ax[0].text(0.05, 0.95, txt, transform=ax[0].transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))

    # --- 4) 잔차 히스토그램 (y_pred - y_true) ---
    res = y_pred - y_true
    mu, sigma = np.mean(res), np.std(res)

    ax[1].hist(res, bins=30, alpha=0.8, edgecolor='k')
    ax[1].axvline(0.0, color='k', linestyle='--', linewidth=1)
    ax[1].axvline(mu, color='r', linestyle='-', linewidth=1, label=rf'$\mu={mu:.3g}$')
    ax[1].axvline(mu+2*sigma, color='r', linestyle=':', linewidth=1, label=rf'$\mu\pm2\sigma$')
    ax[1].axvline(mu-2*sigma, color='r', linestyle=':', linewidth=1)

    ax[1].set_xlabel(r'$\mathrm{residual}=y_{\mathrm{pred}}-y_{\mathrm{true}}$')
    ax[1].set_ylabel('count')
    ax[1].set_title('Residual histogram (Validation)')
    ax[1].legend(loc='best')
    ax[1].grid(True, linestyle=':', linewidth=0.6)

    plt.tight_layout()
    plt.show()
    #%%
    # X1 = X_params[0]
    # L1 = Leak[0]
    
    idx_1 = 1241
    
    def pred1(idx_1):
        with torch.no_grad():
            X_te = torch.tensor(scaler_X.transform(X_params[idx_1].reshape(1, -1)), dtype=torch.float32).to(device)
            # y_te = Leak[0].to(device)
            y_pred_scaled = model(X_te).cpu().numpy()
            # y_true_scaled = y_te.cpu().numpy()
            
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        # y_pred = (y_pred_scaled*scaler_y.scale_ + scaler_y.mean_).transpose()
        # y_true = scaler_y.inverse_transform(y_true_scaled)
        
        print(f"numerical: {Leak[idx_1]}")
        print(f"prediction: {y_pred}")
        
        
        
    
    # --- TorchScript export: params -> scalar MLP ---
    import copy
    model_cpu = copy.deepcopy(model).eval().to("cpu")

    in_dim = int(X_tensor.shape[1])                      # 입력 차원
    dummy_x = torch.zeros(1, in_dim, dtype=torch.float32)  # [B=1, in_dim]

    with torch.no_grad():
        ts = torch.jit.trace(model_cpu, dummy_x, check_trace=False)

    ts.save(network_path_ts)  # e.g., 'net/mlp_leak.ts.pt' 같은 경로

    # model = SimpleMLP(
    #     in_dim=3, 
    #     out_dim=1, 
    #     hidden_channels=2**6, 
    #     n_layers=3, 
    #     p_drop=0.1,
    # ).to("cpu")
    # ckpt = torch.load(network_path, map_location="cpu", weights_only=False)
    # sd = ckpt.get("state_dict", ckpt); sd.pop("_metadata", None)
    # model.load_state_dict(sd, strict=False)
    # model.eval().to("cpu")

    # # --- TorchScript trace (fix TracingCheckError by disabling graph re-check) ---
    # L = int(grid_tensor.shape[0])  # 고정 길이(트레이스 시점에 고정됨)
    # dummy_params = torch.zeros(1, int(X_tensor.shape[1]), dtype=torch.float32)  # [B, n_params]
    # dummy_grid   = torch.zeros(1, L, 1, dtype=torch.float32)                     # [B, L, 1]

    # # ts = torch.jit.trace(model, (dummy_params, dummy_grid))

    # with torch.no_grad():
    #     ts = torch.jit.trace(model, (dummy_params, dummy_grid), check_trace=False)

    # ts.save(network_path_ts)
    
    
