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

from model_validation.train import (
    TrainSettings,
    build_model,
    compute_metrics,
    compute_per_head_metrics,
    inverse_scale_rdc,
    load_rdc,
)

DEFAULT_HEAD_LABELS = ("K", "k", "C", "c")


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


def batched_forward(
    model: torch.nn.Module,
    features: np.ndarray,
    grid: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
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
