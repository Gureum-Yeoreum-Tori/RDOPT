#%%
"""Create validation figures for DeepONet RDC predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.lines as mlines

from model_validation.train import (
    TrainSettings,
    build_model,
    compute_metrics,
    compute_per_head_metrics,
    inverse_scale_rdc,
    load_rdc,
)

figsize_DC = (3.3, 2.06)   

def _default_rcparams():
    plt.rcParams.update({
        "figure.figsize": figsize_DC,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1
    })

DEFAULT_HEAD_LABELS = ("C", "c", "K", "k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="DeepONet checkpoint produced by run_training().")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of sample cases to plot from the selected split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_validation") / "figures",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional metrics JSON path (defaults to output-dir / <stem>_<split>_metrics.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling plot cases.",
    )
    return parser.parse_args()


def restore_scaler(stats: Dict[str, Sequence[float]]) -> StandardScaler:
    scaler = StandardScaler()
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(stats["scale"], dtype=np.float64).reshape(-1)
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = mean.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def restore_scalers_list(stats: Dict[str, Sequence[Sequence[float]]]) -> Sequence[StandardScaler]:
    means = stats.get("mean", [])
    scales = stats.get("scale", [])
    if len(means) != len(scales):
        raise ValueError("Malformed scaler list: mean and scale lengths differ")
    return [restore_scaler({"mean": m, "scale": s}) for m, s in zip(means, scales)]


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _format_r2(value: float, decimals: int = 3) -> str:
    """Floor R^2 to the requested decimal places to avoid showing 1.000."""
    factor = 10 ** decimals
    floored = math.floor(value * factor) / factor
    max_display = 1.0 - 1.0 / factor
    if floored > max_display:
        floored = max_display
    return f"{floored:.{decimals}f}"


def batched_forward(
    model: torch.nn.Module,
    features: np.ndarray,
    grid: torch.Tensor,
    device: torch.device,
    batch_size: int = 2**20,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        xb = torch.from_numpy(features[start:stop]).to(device)
        grid_batch = grid.unsqueeze(0).expand(xb.size(0), -1, -1)
        with torch.no_grad():
            preds = model(xb, grid_batch)
        outputs.append(preds.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_rdc_figure(
    checkpoint_path: Path,
    checkpoint: Dict,
    split: str,
    num_samples: int,
    output_dir: Path,
    metrics_path: Path | None,
    seed: int,
) -> None:
    settings = TrainSettings(**checkpoint["settings"])
    if settings.target != "rdc":
        raise ValueError("The supplied checkpoint was not trained on RDC data.")

    X, Y, grid_norm, grid_raw = load_rdc(settings.data_dir, settings.mat_files, settings.rdc_indices)
    scaler_X = restore_scaler(checkpoint["scalers"]["X"])
    scalers_Y = restore_scalers_list(checkpoint["scalers"]["Y"])

    X_scaled = scaler_X.transform(X).astype(np.float32)

    splits = {name: np.asarray(idx, dtype=np.int32) for name, idx in checkpoint["splits"].items()}
    if split not in splits:
        raise KeyError(f"Split '{split}' not available in checkpoint")
    split_idx = splits[split]
    if split_idx.size == 0:
        raise ValueError(f"Split '{split}' contains no samples")

    device = torch.device("cpu")
    head_names = tuple(settings.head_names) if settings.head_names else DEFAULT_HEAD_LABELS
    model = build_model(
        settings,
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        head_names=head_names,
        trunk_input_dim=grid_norm.shape[-1],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    norm_grid = torch.from_numpy(grid_norm.astype(np.float32)).to(device)
    preds_scaled = batched_forward(model, X_scaled[split_idx], norm_grid, device)
    preds = inverse_scale_rdc(preds_scaled, scalers_Y)
    true = Y[split_idx]

    aggregate = compute_metrics(true, preds)
    per_head = compute_per_head_metrics(true, preds, head_names)
    metrics = {"aggregate": aggregate, "per_head": per_head}

    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = settings.exp_name or checkpoint_path.stem
    if metrics_path is None:
        metrics_path = output_dir / f"{exp_name}_{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, split_idx.size)
    offsets = rng.choice(split_idx.size, size=sample_count, replace=False)

    grid_physical = np.asarray(checkpoint.get("grid", {}).get("original", grid_raw.flatten()), dtype=np.float64)
    rpm = grid_physical * 30.0 / np.pi

    n_heads = true.shape[1]
    fig, axes = plt.subplots(
        sample_count,
        n_heads,
        figsize=(n_heads * 3.2, sample_count * 2.4),
        sharex=True,
    )
    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    line_kwargs = dict(linewidth=1.4)
    for row, offset in enumerate(offsets):
        case_id = int(split_idx[offset])
        for col in range(n_heads):
            ax = axes[row, col]
            ax.plot(rpm, true[offset, col], label="Measured", color="#2E86AB", **line_kwargs)
            ax.plot(rpm, preds[offset, col], label="Predicted", color="#E36F1E", linestyle="--", **line_kwargs)
            ax.grid(True, linestyle=":", linewidth=0.6)
            if row == 0:
                ax.set_title(head_names[col])
            if col == 0:
                ax.set_ylabel(f"Sample #{case_id}")
    for ax in axes[-1, :]:
        ax.set_xlabel("Rotating speed (rpm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))

    figure_path = output_dir / f"{exp_name}_{split}_curves.png"
    fig.savefig(figure_path, dpi=300)


def predict_rdc(
    checkpoint: Dict,
    split: str,
):
    settings = TrainSettings(**checkpoint["settings"])
    if settings.target != "rdc":
        raise ValueError("The supplied checkpoint was not trained on RDC data.")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    X, Y, grid_norm, grid_raw = load_rdc(settings.data_dir, settings.mat_files, settings.rdc_indices)
    scaler_X = restore_scaler(checkpoint["scalers"]["X"])
    scalers_Y = restore_scalers_list(checkpoint["scalers"]["Y"])

    X_scaled = scaler_X.transform(X).astype(np.float32)

    splits = {name: np.asarray(idx, dtype=np.int32) for name, idx in checkpoint["splits"].items()}
    if split not in splits:
        raise KeyError(f"Split '{split}' not available in checkpoint")
    split_idx = splits[split]
    if split_idx.size == 0:
        raise ValueError(f"Split '{split}' contains no samples")

    device = torch.device("cpu")
    head_names = tuple(settings.head_names) if settings.head_names else DEFAULT_HEAD_LABELS
    model = build_model(
        settings,
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        head_names=head_names,
        trunk_input_dim=grid_norm.shape[-1],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    norm_grid = torch.from_numpy(grid_norm.astype(np.float32)).to(device)
    preds_scaled = batched_forward(model, X_scaled[split_idx], norm_grid, device)
    preds = inverse_scale_rdc(preds_scaled, scalers_Y)
    true = Y[split_idx]

    aggregate = compute_metrics(true, preds)
    per_head = compute_per_head_metrics(true, preds, head_names)
    metrics = {"aggregate": aggregate, "per_head": per_head}
    
    return true, preds, metrics, grid_raw

#%%
def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    build_rdc_figure(
        args.checkpoint,
        checkpoint,
        args.split,
        args.num_samples,
        args.output_dir,
        args.metrics_json,
        args.seed,
    )
    print(f"Saved figures to {args.output_dir}")
    if args.metrics_json:
        print(f"Saved metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()









#%%
checkpoints = ('net/tuned_runs/deeponet/deeponet_rdc_20250908_T_182846_tuned.pth',
            'net/tuned_runs/deeponet/deeponet_rdc_20250911_T_091324_tuned.pth',
            'net/tuned_runs/deeponet/deeponet_rdc_20250908_T_183632_tuned.pth',
            'net/tuned_runs/deeponet/deeponet_rdc_20250908_T_203220_tuned.pth',)









# %%

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from time import time as tt
import math
import matplotlib.lines as mlines

ck_path = Path('net/tuned_runs/deeponet/deeponet_rdc_20250908_T_203220_tuned.pth')
checkpoint = load_checkpoint(ck_path)

settings = TrainSettings(**checkpoint["settings"])
X, Y, grid_norm, grid_raw = load_rdc(settings.data_dir, settings.mat_files, settings.rdc_indices)
scaler_X = restore_scaler(checkpoint["scalers"]["X"])
X_scaled = scaler_X.transform(X).astype(np.float32)

splits = {k: np.asarray(v, dtype=np.int32) for k, v in checkpoint["splits"].items()}
tr_idx = splits["train"]
te_idx = splits["test"]

w = grid_raw.reshape(-1)                 # [n_w]
n_w = w.size
rpm = w * 30/np.pi

# (n_case, n_head, n_w)
true_tr = Y[tr_idx]
true_te = Y[te_idx]

def make_flat_io(X_case, Y_case):
    n_case = X_case.shape[0]
    X_flat = np.repeat(X_case, n_w, axis=0)
    w_flat = np.tile(w.reshape(-1,1), (n_case,1))
    X_flat = np.hstack([X_flat, w_flat])                  # [n_case*n_w, n_par+1]
    Y_flat = Y_case.transpose(0,2,1).reshape(-1, Y_case.shape[1])   # [n_case*n_w, n_head]
    return X_flat, Y_flat

X_tr_flat, y_tr_flat = make_flat_io(X_scaled[tr_idx], true_tr)
X_te_flat, y_te_flat = make_flat_io(X_scaled[te_idx], true_te)


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

head_names = tuple(settings.head_names) if settings.head_names else DEFAULT_HEAD_LABELS
model = build_model(settings, input_dim=X.shape[1], output_dim=Y.shape[1],
                    head_names=head_names, trunk_input_dim=grid_norm.shape[-1])
model.load_state_dict(checkpoint["state_dict"])
model.eval().to(device)
grid_t = torch.from_numpy(grid_norm.astype(np.float32)).to(device)

# t0 = tt()
# with torch.no_grad():
#     preds_scaled = batched_forward(model, X_scaled[te_idx], grid_t, device)
# preds_dn = inverse_scale_rdc(preds_scaled, restore_scalers_list(checkpoint["scalers"]["Y"]))
# t1 = tt()


# X_big = X_scaled.repeat(10,axis=0)
# t0 = tt()
# with torch.no_grad():
#     preds_scaled = batched_forward(model, X_big, grid_t, device)
# preds_dn = inverse_scale_rdc(preds_scaled, restore_scalers_list(checkpoint["scalers"]["Y"]))
# t1 = tt()
# print(f"[DeepONet] forward_time = {t1 - t0:.4f}s")


# 모델들 (트리 제외 스케일러 적용)
models = [
    ("Ridge", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])),
    ("SVR",   Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.01))])),
    ("KRR",   Pipeline([("scaler", StandardScaler()), ("reg", KernelRidge(kernel="rbf", alpha=1e-2, gamma=None))])),
    ("RF",    RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=0)),
]

from sklearn.base import clone
from joblib import dump, load

head_order = ("C","c","K","k")  # 고정
trained = {}                    # {"Ridge": [mdl_C, mdl_c, mdl_K, mdl_k], ...}

for mname, mdl in models:
    per_head = []
    head_models = []
    for j, name in enumerate(head_order):
        mdl_j = clone(mdl)                           # 헤드별 독립 모델
        t0 = tt()
        mdl_j.fit(X_tr_flat, y_tr_flat[:, j])
        t1 = tt()
        print(f"{mname} rdc_{j} fit time = {t1 - t0:.4f}s")
        head_models.append(mdl_j)
        
        t0 = tt()
        yhat = mdl_j.predict(X_te_flat).reshape(-1, n_w)   # [n_case, n_w]
        t1 = tt()
        print(f"{mname} rdc_{j} predict time = {t1 - t0:.4f}s")
        metric_ = compute_metrics(true_te[:, j, :], yhat)
        rmse = metric_['rmse']
        mae = metric_['mae']
        mape = metric_['mape']
        r2 = metric_['r2']
        rrmse = metric_['rrmse']
        per_head.append((rmse, mae, r2, rrmse))
        # print(f"[{mname}][{name}] RMSE={rmse:.3g}  MAE={mae:.3g}  MAPE={mape:.2f}%  R^2={r2:.4f}  rRMSE={100*rrmse:.2f}%")

    trained[mname] = head_models

#%%
def predict_baseline(trained_family, X_flat, n_case, n_w):
    """trained_family: [mdl_C, mdl_c, mdl_K, mdl_k]"""
    yh_list = []
    for mdl in trained_family:
        yh = mdl.predict(X_flat).reshape(n_case, n_w)
        yh_list.append(yh)
    # (n_case, 4, n_w) 순서로 스택
    return np.stack(yh_list, axis=1)








#%%
checkpoints = ('net/tuned_runs/deeponet/deeponet_rdc_20250908_T_203220_tuned.pth',)
_default_rcparams()
ck_path = Path(checkpoints[0])
checkpoint = load_checkpoint(ck_path)
figsize_F = (6.8,4.2)
ylabels = ['K $(N/m)$', 'k $(N/m)$',f'C ($N \\cdot s/m$)', f'c ($N \\cdot s/m$)']

true, preds, metrics, w_vec = predict_rdc(
    checkpoint,
    split='test'
    )
yt = true[:,[2,3,0,1,]] # n_data, RDC (K, k, C, c), n_w
yp = preds[:,[2,3,0,1,]] # n_data, RDC, n_w
rpm = w_vec.squeeze()*30/np.pi
plot_idx = [276, 140, 89, 244, 35]

fig,axes = plt.subplots(2,2,figsize=figsize_F)
axes = axes.ravel()

label_wrong = ('K' ,'k', 'C', 'c')
label_right = ('C', 'c','K' ,'k')
# mapping = dict(zip(label_wrong, label_right))
# for old in label_wrong:
#     new = mapping[old]
#     rmse = metrics['per_head'][new]['rmse']
#     print(new, rmse)

for i in range(4):
    ax = axes[i]
    ax.plot(rpm,yt[plot_idx,i].T, 'o',
                    alpha=0.7, markerfacecolor='none', markersize=4)
    ax.set_prop_cycle(None)
    ax.plot(rpm,yp[plot_idx,i].T, '-')
    ax.set_xlabel("Rotating speed (rpm)")
    ax.set_ylabel(ylabels[i])
    
    metric = metrics['per_head'][label_right[i]]
    r2 = metric['r2']
    r2_disp = _format_r2(r2)
    rrmse = metric['rrmse']
    mae = metric['mae']
    mape = metric['mape']
    
    txt = '\n'.join([
        rf'$R^2={r2_disp}$',
        # rf'$\mathrm{{RMSE}}={rmse:.3g}$',
        # rf'$\mathrm{{MAE}}={mae:.3g}$',
        rf'$\mathrm{{MAPE}}={mape:.2f}\%$',
        rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
    ])
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0.5))

legend_handles = [
    mlines.Line2D([], [], color='k', marker='o', linestyle='None', markerfacecolor='none', label='True value', markersize=4),
    mlines.Line2D([], [], color='k', linestyle='-', label='DeepONet'),
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=len(legend_handles),
    bbox_to_anchor=(0.5, -0.08),
)
fig.tight_layout()
fig.savefig("val_rdc.png",dpi=600,bbox_inches="tight")


#%%

n_case_te = X_scaled[te_idx].shape[0]
y_ridge = predict_baseline(trained["Ridge"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
y_svf = predict_baseline(trained["SVR"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
y_krr = predict_baseline(trained["KRR"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
y_rf = predict_baseline(trained["RF"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
yp = preds[:,[2,3,0,1,]] # n_data, RDC, n_w

#%%
plot_idx = np.random.randint(0,len(te_idx),3)
plot_idx = [276, 140, 89, 244, 35, 252]
plot_idx = plot_idx[-1]
figsize_F_s = (6.8,3.2)
fig,axes = plt.subplots(2,2,figsize=figsize_F_s)
axes = axes.ravel()



label_wrong = ('K' ,'k', 'C', 'c')
label_right = ('C', 'c','K' ,'k')
linestyle_tuple = [
    ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 5))),
    ('densely dotted',        (0, (1, 1))),

    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),

    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),

    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())  # HEX 값 리스트

for i in range(4):
    ax = axes[i]
    # ax.plot(rpm,yt[plot_idx,i].T, 'o',alpha=0.7, markerfacecolor='none', markersize=4)
    ax.plot(rpm,yt[plot_idx,i].T, '-',color='k')
    # ax.set_prop_cycle(None)
    # ax.plot(rpm,yp[plot_idx,i].T, '-')
    # ax.set_prop_cycle(None)
    # ax.set_prop_cycle(None)
    ax.plot(rpm,y_rf[plot_idx,i].T, 'o', markersize=6, markerfacecolor='none',color=colors[0])
    # ax.plot(rpm,y_svf[plot_idx,i].T, '+', markersize=6,color=colors[1])
    # ax.plot(rpm,y_ridge[plot_idx,i].T, '^', markersize=6,color=colors[4])
    # ax.set_prop_cycle(None)
    ax.plot(rpm,y_krr[plot_idx,i].T, 'p', markersize=5,color=colors[2])
    ax.plot(rpm,yp[plot_idx,i].T, '*', markersize=5,markeredgewidth=1,color=colors[3])
    ax.set_xlabel("Rotating speed (rpm)")
    ax.set_ylabel(ylabels[i])
    
    metric = metrics['per_head'][label_right[i]]
    r2 = metric['r2']
    r2_disp = _format_r2(r2)
    rrmse = metric['rrmse']
    mae = metric['mae']
    mape = metric['mape']
    
    # txt = '\n'.join([
    #     rf'$R^2={r2_disp}$',
    #     # rf'$\mathrm{{RMSE}}={rmse:.3g}$',
    #     # rf'$\mathrm{{MAE}}={mae:.3g}$',
    #     rf'$\mathrm{{MAPE}}={mape:.2f}\%$',
    #     rf'$\mathrm{{rRMSE}}={100*rrmse:.2f}\%$',
    # ])
    # ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
        # bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0.5))

legend_handles = [
    mlines.Line2D([], [], color='k', linestyle='-', label='True value'),
    mlines.Line2D([], [], color=colors[3], marker='*', linestyle='None', label='DeepONet', markersize=6),
    mlines.Line2D([], [], color=colors[0], marker='o', linestyle='None', markerfacecolor='none', label='Random Forest', markersize=6),
    # mlines.Line2D([], [], color=colors[1], marker='+', linestyle='None', label='SVR', markersize=6),
    # mlines.Line2D([], [], color=colors[4], marker='^', linestyle='None', label='Ridge', markersize=6),
    mlines.Line2D([], [], color=colors[2], marker='p', linestyle='None', label='Kernel Ridge', markersize=4),
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=len(legend_handles),
    bbox_to_anchor=(0.5, -0.06),
)
fig.tight_layout()
fig.savefig("val_rdc_reg.png",dpi=600,bbox_inches="tight")

X[te_idx[plot_idx]]
# %%
y_ridge = predict_baseline(trained["Ridge"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] # K, k, C, c
y_svf = predict_baseline(trained["SVR"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
y_krr = predict_baseline(trained["KRR"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 
y_rf = predict_baseline(trained["RF"], X_te_flat, n_case_te, n_w)[:,[2,3,0,1,]] 

#%%
y_ridge_ = y_ridge[:,[2,3,0,1]]
y_svf_ = y_svf[:,[2,3,0,1]]
y_krr_ = y_krr[:,[2,3,0,1]]
y_rf_ = y_rf[:,[2,3,0,1]]

yh = [] 
yh.append(y_ridge_)
yh.append(y_svf_)
yh.append(y_krr_)
yh.append(y_rf_)

results = {}

for (mname, _), y_pred in zip(models, yh):
    per_head = {}
    for j, name in enumerate(head_order):   # ("C","c","K","k")
        metrics_j = compute_metrics(true_te[:, j, :], y_pred[:, j, :])
        per_head[name] = metrics_j
    results[mname] = {"per_head": per_head}

# %%

def make_flat_i(X_case):
    n_case = X_case.shape[0]
    X_flat = np.repeat(X_case, n_w, axis=0)
    w_flat = np.tile(w.reshape(-1,1), (n_case,1))
    X_flat = np.hstack([X_flat, w_flat]) 
    return X_flat

X_100 = X_scaled[np.random.randint(0,len(X_scaled),100)]
orders = [10,30]
# for order in range(1,20):
for _, order in enumerate(orders):
    X_big = X_100.repeat(order,axis=0)
    t0 = tt()
    with torch.no_grad():
        preds_scaled = batched_forward(model, X_big, grid_t, device)
    preds_dn = inverse_scale_rdc(preds_scaled, restore_scalers_list(checkpoint["scalers"]["Y"]))
    t1 = tt()
    print(f"[DeepONet] forward_time = {t1 - t0:.4f}s")

    X_flat = make_flat_i(X_big)
    t0 = tt()
    y_krr = predict_baseline(trained["KRR"], X_flat, len(X_big), n_w)[:,[2,3,0,1,]] 
    t1 = tt()
    print(f'KRR prediction time = {t1 - t0:.4f}s')
    
    t0 = tt()
    y_krr = predict_baseline(trained["RF"], X_flat, len(X_big), n_w)[:,[2,3,0,1,]] 
    t1 = tt()
    print(f'RF prediction time = {t1 - t0:.4f}s')
# %%
