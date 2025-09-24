#%%
"""Generate parity plots and metrics for a trained leakage MLP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import matplotlib.axes as axs
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from model_validation.train import (
    TrainSettings,
    build_model,
    compute_metrics,
    load_leak,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a leakage MLP checkpoint produced by run_training().")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split used for plotting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the parity plot image (defaults to checkpoint stem + _parity.png).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional path to store computed metrics as JSON.",
    )
    return parser.parse_args()

def restore_scaler(stats: Dict[str, list]) -> StandardScaler:
    scaler = StandardScaler()
    mean = np.asarray(stats["mean"], dtype=np.float64).reshape(-1)
    scale = np.asarray(stats["scale"], dtype=np.float64).reshape(-1)
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = mean.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


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


def build_leak_parity(
    checkpoint: Dict,
    split: str,
    output_path: Path,
    metrics_path: Path | None,
) -> None:
    settings = TrainSettings(**checkpoint["settings"])
    if settings.target != "leak":
        raise ValueError("The supplied checkpoint was not trained on leakage data.")

    X, y = load_leak(settings.data_dir, settings.mat_files, settings.leak_index)

    scaler_X = restore_scaler(checkpoint["scalers"]["X"])
    scaler_y = restore_scaler(checkpoint["scalers"]["y"])
    X_scaled = scaler_X.transform(X).astype(np.float32)
    y_scaled = scaler_y.transform(y).astype(np.float32)

    splits = {name: np.asarray(idx, dtype=np.int32) for name, idx in checkpoint["splits"].items()}
    if split not in splits:
        raise KeyError(f"Split '{split}' not available in checkpoint")
    split_idx = splits[split]

    device = torch.device("cpu")
    model = build_model(settings, input_dim=X.shape[1], output_dim=y.shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X_scaled[split_idx]).to(device)
        preds_scaled = model(xb).cpu().numpy()

    y_pred = scaler_y.inverse_transform(preds_scaled)
    y_true = y[split_idx]

    metrics = compute_metrics(
        y_true.reshape(y_true.shape[0], 1, 1),
        y_pred.reshape(y_pred.shape[0], 1, 1),
    )

    if metrics_path is None:
        metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    mn = float(np.min([y_true.min(), y_pred.min()]))
    mx = float(np.max([y_true.max(), y_pred.max()]))
    
    _default_rcparams()
    fig, ax = plt.subplots()
    ax.scatter(y_true[:, 0], y_pred[:, 0], s=18, alpha=0.6, label=f"{split.title()} samples")
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", linewidth=0.8, label="y = x")
    ax.set_xlabel("Measured $\\dot{m}$")
    ax.set_ylabel("Predicted $\\dot{m}$")
    ax.set_title(f"Leakage parity ({split})")
    r2_display = _format_r2(metrics["r2"], decimals=3)
    text = "\n".join(
        [
            f"RMSE: {metrics['rmse']:.4g}",
            f"MAE:  {metrics['mae']:.4g}",
            f"R^2:  {r2_display}",
            f"rRMSE: {metrics['rrmse']*100:.2f}%",
        ]
    )
    ax.text(0.05, 0.05, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5))
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def predict_and_draw(
    checkpoint: Dict,
    split: str,
    ax: axs.Axes
) -> None:
    settings = TrainSettings(**checkpoint["settings"])
    if settings.target != "leak":
        raise ValueError("The supplied checkpoint was not trained on leakage data.")

    X, y = load_leak(settings.data_dir, settings.mat_files, settings.leak_index)

    scaler_X = restore_scaler(checkpoint["scalers"]["X"])
    scaler_y = restore_scaler(checkpoint["scalers"]["y"])
    X_scaled = scaler_X.transform(X).astype(np.float32)
    y_scaled = scaler_y.transform(y).astype(np.float32)

    splits = {name: np.asarray(idx, dtype=np.int32) for name, idx in checkpoint["splits"].items()}
    if split not in splits:
        raise KeyError(f"Split '{split}' not available in checkpoint")
    split_idx = splits[split]

    device = torch.device("cpu")
    model = build_model(settings, input_dim=X.shape[1], output_dim=y.shape[1])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X_scaled[split_idx]).to(device)
        preds_scaled = model(xb).cpu().numpy()

    y_pred = scaler_y.inverse_transform(preds_scaled)
    y_true = y[split_idx]

    metrics = compute_metrics(
        y_true.reshape(y_true.shape[0], 1, 1),
        y_pred.reshape(y_pred.shape[0], 1, 1),
    )


    mn = float(np.min([y_true.min(), y_pred.min()]))
    mx = float(np.max([y_true.max(), y_pred.max()]))
    
    import matplotlib.colors as mcolors
    mcolors_list = list(mcolors.TABLEAU_COLORS.values())
    
    _default_rcparams()
    # fig, ax = plt.subplots()
    # ax.scatter(y_true[:, 0], y_pred[:, 0], s=18, alpha=0.6, label=f"{split.title()} samples")
    ax.scatter(y_true[:, 0], y_pred[:, 0], s=14, label="data", facecolors="none", edgecolors=mcolors_list[0], linewidth=0.5)
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", linewidth=0.8, label="y = x")
    ax.set_xlabel("Measured $\\dot{m}$")
    ax.set_ylabel("Predicted $\\dot{m}$")
    # ax.axis('equal')
    # ax.set_title(f"Leakage parity ({split})")
    r2_display = _format_r2(metrics["r2"], decimals=5)
    r2_display = _format_r2(metrics["r2"], decimals=5)
    text = "\n".join(
        [
            f"RMSE: {metrics['rmse']:.4g}",
            f"MAE:  {metrics['mae']:.4g}",
            f"$R^2$:  {r2_display}",
            f"rRMSE: {metrics['rrmse']*100:.2f}%",
        ]
    )
    r2 = metrics["r2"]
    rmse = metrics['rmse']
    # ax.text(0.05, 0.05, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            # bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5))
    # ax.grid(True, linestyle=":", linewidth=0.6)
    # ax.legend()
    return ax, r2, rmse


#%%
def main() -> None:
    args = parse_args()
    # checkpoint = load_checkpoint(args.checkpoint)
    # output_path = args.output or args.checkpoint.with_name(args.checkpoint.stem + "_parity.png")
    # metrics_path = args.metrics_json
    # build_leak_parity(checkpoint, args.split, output_path, metrics_path)
    # print(f"Saved parity plot to {output_path}")
    # if metrics_path:
    #     print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()

#%%
# args = parse_args()
checkpoints = ('net/tuned_runs/mlp/mlp_leak_20250908_T_182846_tuned.pth',
            'net/tuned_runs/mlp/mlp_leak_20250911_T_091324_tuned.pth',
            'net/tuned_runs/mlp/mlp_leak_20250908_T_183632_tuned.pth',
            'net/tuned_runs/mlp/mlp_leak_20250908_T_203220_tuned.pth',)
fig,axes = plt.subplots(2,2,figsize=(6,4))
axes = axes.ravel()
for i, ck_path in enumerate(checkpoints):
    ck_path = Path(ck_path)
    checkpoint = load_checkpoint(ck_path)
    axes[i]=predict_and_draw(checkpoint = checkpoint, split= 'test', ax= axes[i])

fig.tight_layout()


# plt.plot(checkpoint['history']['train'])
# plt.plot(checkpoint['history']['val'])

# %%
checkpoints = ('net/tuned_runs/mlp/mlp_leak_20250908_T_203220_tuned.pth',)
# ck_path = Path(ck_path)
ck_path = Path(checkpoints[0])
checkpoint = load_checkpoint(ck_path)
figsize_SC_ss = (6, 2.2)

fig,axes = plt.subplots(1,3,figsize=figsize_SC_ss)
axes = axes.ravel()
ra = np.random.random(3)
ra = np.sort(ra)
axes[0], r2, rmse =predict_and_draw(checkpoint = checkpoint, split= 'train', ax= axes[0])
axes[0].set_title("Train")

# r22 = r2-ra[0]*1e-4
# r2_display = _format_r2(r22, decimals=4)
r2_display = _format_r2(r2, decimals=4)
text = "\n".join(
    [
        f"RMSE: {rmse:.3g}",
        f"$R^2$:  {r2_display}",
    ]
)
axes[0].text(0.95, 0.05, text, transform=axes[0].transAxes, va="bottom", ha="right", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0.5))

axes[1], r2, rmse=predict_and_draw(checkpoint = checkpoint, split= 'val', ax= axes[1])
axes[1].set_title("Valid")

# r22 = r2-ra[1]*1e-4
# r2_display = _format_r2(r22, decimals=4)
r2_display = _format_r2(r2, decimals=4)

text = "\n".join(
    [
        f"RMSE: {rmse:.3g}",
        f"$R^2$:  {r2_display}",
    ]
)
axes[1].text(0.95, 0.05, text, transform=axes[1].transAxes, va="bottom", ha="right", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0.5))



axes[2], r2, rmse=predict_and_draw(checkpoint = checkpoint, split= 'test', ax= axes[2])
axes[2].set_title("Test")

# r22 = r2-ra[2]*1e-4
# r2_display = _format_r2(r22, decimals=4)
r2_display = _format_r2(r2, decimals=4)

text = "\n".join(
    [
        f"RMSE: {rmse:.3g}",
        f"$R^2$:  {r2_display}",
    ]
)
axes[2].text(0.95, 0.05, text, transform=axes[2].transAxes, va="bottom", ha="right", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0.5))


axes[2].legend()

for i in range(3):
    # axes[i].set_xlim(0,30)
    # axes[i].set_ylim(0,30)
    axes[i].set_xticks(np.arange(0, 30+0.1, 10))
    axes[i].set_yticks(np.arange(0, 30+0.1, 10))

fig.tight_layout()
fig.savefig("val_mlp.png",dpi=600,bbox_inches="tight")

# %%
