#%%
# model_validation/example_load_seal.py
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from model_validation.models import DeepONet
from model_validation.train import (
    TrainSettings,
    compute_metrics,
    compute_per_head_metrics,
    load_rdc,
    split_indices,
)


def load_deeponet_bundle(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    settings = TrainSettings(**ckpt["settings"])
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x_stats = ckpt["scalers"]["X"]
    x_mean = torch.tensor(x_stats["mean"], dtype=torch.float32, device=device)
    x_scale = torch.tensor(x_stats["scale"], dtype=torch.float32, device=device)

    y_stats = ckpt["scalers"]["Y"]
    y_mean = torch.tensor(y_stats["mean"], dtype=torch.float32, device=device)
    y_scale = torch.tensor(y_stats["scale"], dtype=torch.float32, device=device)

    out_dim, _ = y_mean.shape
    model = DeepONet(
        input_dim=x_mean.numel(),
        output_dim=out_dim,
        param_embd_dim=settings.param_embedding_dim,
        branch_layers=settings.branch_layers,
        trunk_layers=settings.trunk_layers,
        latent_dim=settings.n_basis,
        activation=settings.activation,
        dropout=settings.dropout,
        trunk_input_dim=1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    grid = ckpt.get("grid", {})
    grid_original = torch.tensor(grid.get("original", []), dtype=torch.float32, device=device)
    grid_normalized = torch.tensor(grid.get("normalized", []), dtype=torch.float32, device=device)
    return {
        "model": model,
        "device": device,
        "settings": settings,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "grid_original": grid_original.view(-1),
        "grid_normalized": grid_normalized.view(-1, 1) if grid_normalized.numel() else None,
    }


def normalize_features(x_raw: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x_raw - mean) / scale


def renormalize_grid(w_query: Sequence[float], grid_original: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(w_query, dtype=torch.float32, device=grid_original.device)
    w_min = grid_original.min()
    w_max = grid_original.max()
    w_norm = 2 * (w - w_min) / (w_max - w_min) - 1
    return w_norm.unsqueeze(-1)


def denormalize_batch(
    preds_scaled: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> np.ndarray:
    target_mean = mean.unsqueeze(0)
    target_scale = scale.unsqueeze(0)
    restored = preds_scaled * target_scale + target_mean
    return restored.detach().cpu().numpy()


def denormalize_heads(
    pred_scaled: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    head_names: Sequence[str] | None,
) -> Dict[str, np.ndarray]:
    names = list(head_names) if head_names else [f"head_{i}" for i in range(pred_scaled.size(0))]
    pred = pred_scaled * scale + mean
    return {name: pred[idx].detach().cpu().numpy() for idx, name in enumerate(names)}


def visualize_cases(
    w_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_names: Sequence[str],
    test_indices: Sequence[int],
    n_plot: int,
    out_dir: Optional[Path],
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib optional
        print("[warn] matplotlib is not installed; skip plotting.")
        return

    if n_plot <= 0:
        return

    n_cases = len(test_indices)
    if n_cases == 0:
        print("[warn] No test samples available for plotting.")
        return

    n_plot = min(n_plot, n_cases)
    head_labels = list(head_names)
    w_axis = np.asarray(w_axis).reshape(-1)

    save_dir = None
    if out_dir is not None:
        save_dir = Path(out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for local_idx in range(n_plot):
        dataset_idx = int(test_indices[local_idx])
        true_vals = y_true[local_idx]
        pred_vals = y_pred[local_idx]

        fig, axes = plt.subplots(len(head_labels), 1, sharex=True, figsize=(8, 3 * len(head_labels)))
        if len(head_labels) == 1:
            axes = [axes]

        for head_idx, ax in enumerate(axes):
            ax.plot(w_axis, true_vals[head_idx], label="true", color="tab:blue")
            ax.plot(w_axis, pred_vals[head_idx], label="pred", color="tab:orange", linestyle="--")
            ax.set_ylabel(head_labels[head_idx])
            ax.grid(True, alpha=0.3)
            if local_idx == 0 and head_idx == 0:
                ax.legend()

        axes[-1].set_xlabel("Angular speed (rad/s)")
        fig.suptitle(f"Test sample {dataset_idx}")
        fig.tight_layout()

        if save_dir is not None:
            out_path = save_dir / f"test_{dataset_idx:05d}.png"
            fig.savefig(out_path, dpi=150)
            print(f"[info] Saved plot for test index {dataset_idx} -> {out_path}")

        if show:
            plt.show()

        plt.close(fig)


def predict_rdc(
    checkpoint: str | Path,
    x_raw: Sequence[float],
    w_query: Sequence[float],
) -> Dict[str, np.ndarray]:
    bundle = load_deeponet_bundle(checkpoint)
    device = bundle["device"]

    branch_in = torch.tensor(x_raw, dtype=torch.float32, device=device)
    branch_norm = normalize_features(branch_in, bundle["x_mean"], bundle["x_scale"]).unsqueeze(0)

    trunk_norm = renormalize_grid(w_query, bundle["grid_original"]).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = bundle["model"](branch_norm, trunk_norm).squeeze(0)

    return denormalize_heads(pred_scaled, bundle["y_mean"], bundle["y_scale"], bundle["settings"].head_names)


def predict_on_test_split(
    checkpoint: Path,
    n_plot: int = 3,
    plot_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    bundle = load_deeponet_bundle(checkpoint)
    settings = bundle["settings"]
    device = bundle["device"]

    X, Y, grid_norm, grid_raw = load_rdc(settings.data_dir, settings.mat_files, settings.rdc_indices)
    train_idx, val_idx, test_idx = split_indices(X.shape[0], settings.seed)

    branch_raw = torch.tensor(X[test_idx], dtype=torch.float32, device=device)
    branch_norm = normalize_features(branch_raw, bundle["x_mean"], bundle["x_scale"])

    if bundle["grid_normalized"] is not None:
        grid_tensor = bundle["grid_normalized"].to(device)
    else:
        grid_tensor = renormalize_grid(grid_raw.flatten(), bundle["grid_original"]).to(device)
    trunk_batch = grid_tensor.unsqueeze(0).expand(branch_norm.size(0), -1, -1)

    with torch.no_grad():
        preds_scaled = bundle["model"](branch_norm, trunk_batch)

    preds = denormalize_batch(preds_scaled, bundle["y_mean"], bundle["y_scale"])
    y_true = Y[test_idx]

    if settings.head_names:
        head_names = list(settings.head_names)
    else:
        head_names = [f"head_{idx}" for idx in range(preds.shape[1])]

    metrics_all = compute_metrics(y_true, preds)
    metrics_per_head = compute_per_head_metrics(y_true, preds, head_names)

    first_idx = test_idx[0]
    sample_pred = preds_scaled[0]
    sample_denorm = denormalize_heads(sample_pred, bundle["y_mean"], bundle["y_scale"], head_names)
    print(f"Test samples: {len(test_idx)} (first index {first_idx})")
    for head, values in sample_denorm.items():
        print(f"[{head}] first sample prediction -> shape {values.shape}, first 5 values: {values[:5]}")

    print("Overall metrics (test split):")
    for name, value in metrics_all.items():
        print(f"  {name}: {value:.6f}")

    print("Per-head metrics (test split):")
    for head, stats in metrics_per_head.items():
        stats_str = ", ".join(f"{k}={v:.6f}" for k, v in stats.items())
        print(f"  {head}: {stats_str}")

    grid_tensor = bundle["grid_original"]
    if grid_tensor.numel():
        w_axis = grid_tensor.cpu().numpy()
    else:
        w_axis = grid_raw.flatten()

    visualize_cases(
        w_axis=w_axis,
        y_true=y_true,
        y_pred=preds,
        head_names=head_names,
        test_indices=test_idx,
        n_plot=n_plot,
        out_dir=plot_dir,
        show=show_plots,
    )

    return metrics_all, metrics_per_head


def main() -> None:
    parser = ArgumentParser(description="Load a DeepONet checkpoint and evaluate on the test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("net/optuna/deeponet/best_trial.pth"),
        help="Path to the saved DeepONet checkpoint.",
    )
    parser.add_argument(
        "--plot-cases",
        type=int,
        default=3,
        help="Number of test cases to visualise (default: 3).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save plots into this directory instead of only showing them.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively (useful together with --plot-dir).",
    )
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"[warn] Ignoring unrecognized arguments: {unknown}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    show = not args.no_show
    predict_on_test_split(
        checkpoint=args.checkpoint,
        n_plot=args.plot_cases,
        plot_dir=args.plot_dir,
        show_plots=show,
    )


if __name__ == "__main__":
    main()

# %%
