"""Generate parity plots and metrics for a trained leakage MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(y_true[:, 0], y_pred[:, 0], s=18, alpha=0.6, label=f"{split.title()} samples")
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", linewidth=0.8, label="y = x")
    ax.set_xlabel("Measured $\\dot{m}$")
    ax.set_ylabel("Predicted $\\dot{m}$")
    ax.set_title(f"Leakage parity ({split})")
    text = "\n".join(
        [
            f"RMSE: {metrics['rmse']:.4g}",
            f"MAE:  {metrics['mae']:.4g}",
            f"R^2:  {metrics['r2']:.4f}",
            f"rRMSE: {metrics['rrmse']*100:.2f}%",
        ]
    )
    ax.text(0.05, 0.05, text, transform=ax.transAxes, va="bottom", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5))
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    output_path = args.output or args.checkpoint.with_name(args.checkpoint.stem + "_parity.png")
    metrics_path = args.metrics_json
    build_leak_parity(checkpoint, args.split, output_path, metrics_path)
    print(f"Saved parity plot to {output_path}")
    if metrics_path:
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
