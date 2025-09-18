from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .data import build_scalers, create_dataloaders, create_kfold_indices, prepare_datasets, apply_scaler, SealDataset
from .train import (
    _flatten_numpy,
    _make_subset,
    _parse_loss_weights,
    build_loss,
    build_model,
    compute_loss,
    forward_model,
)
from .utils import (
    compute_metrics,
    configure_experiment,
    configure_logging,
    get_device,
    load_config,
    run_baselines,
    save_json,
    set_seed,
)

plt.switch_backend("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained seal models")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kfold", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    metrics_cfg = config.get("metrics", {})

    seed = config.get("seed", 42)
    set_seed(seed)

    exp_name = args.exp_name or config.get("experiment", "default_experiment")
    save_dir = Path(args.save_dir or config.get("logging", {}).get("save_dir", "runs"))
    paths = configure_experiment(save_dir, exp_name)
    logger = configure_logging(paths.logs, f"eval_{exp_name}")

    device = get_device(args.device)

    kfold = args.kfold or int(data_cfg.get("kfold", 0))

    if kfold > 1:
        data_cfg["split"] = False
    train_dataset, val_dataset, test_dataset, scalers, features, targets, ids, weights, w_grid = prepare_datasets(
        data_cfg,
        paths.root,
        seed,
    )

    target_names = [str(name) for name in data_cfg.get("targets", [])]
    if not target_names:
        target_names = [f"target_{idx}" for idx in range(train_dataset.targets.shape[-1])]

    if kfold > 1:
        folds = create_kfold_indices(features.shape[0], kfold, seed)
        if args.fold >= len(folds):
            raise ValueError(f"Fold {args.fold} out of range")
        train_idx, val_idx = folds[args.fold]
        train_dataset = _make_subset(features, targets, ids, weights, w_grid, train_idx)
        val_dataset = _make_subset(features, targets, ids, weights, w_grid, val_idx)
        scalers = build_scalers(data_cfg.get("scaler", "none"))
        apply_scaler(scalers, train_dataset, [val_dataset])

    loaders = create_dataloaders(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle_train=False,
    )

    model = build_model(model_cfg, training_cfg, target_names, train_dataset, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prediction_names = list(getattr(model, "head_names", target_names))

    loss_fn = build_loss(training_cfg.get("loss", "mse"))
    loss_weights = _parse_loss_weights(training_cfg.get("loss_weights"), target_names, model)
    metric_names = metrics_cfg.get("report", ["rmse", "mae", "mape", "r2"])

    all_metrics: Dict[str, Dict[str, float]] = {}
    prediction_rows: List[Dict[str, Any]] = []

    for split_name in ["train", "val", "test"]:
        if split_name not in loaders:
            continue
        split_metrics, rows = evaluate_loader(
            model,
            loaders[split_name],
            loss_fn,
            loss_weights,
            metric_names,
            target_names,
            device,
            split_name,
            args.fold,
        )
        all_metrics[split_name] = split_metrics
        prediction_rows.extend(rows)
        logger.info("%s metrics: %s", split_name.capitalize(), json.dumps(split_metrics, indent=2))

    save_json(paths.root / "metrics_eval.json", all_metrics)

    if prediction_rows:
        df = pd.DataFrame(prediction_rows)
        df.to_parquet(paths.predictions / "predictions.parquet", index=False)
        create_plots(df, prediction_names, paths.figures)

    baseline_cfg = config.get("baselines", {})
    if baseline_cfg.get("run", False) and test_dataset is not None and train_dataset.targets.ndim == 2:
        baseline_results = run_baselines(
            train_dataset.features,
            train_dataset.targets,
            test_dataset.features,
            test_dataset.targets,
            baseline_cfg,
            target_names,
            metric_names,
            logger,
        )
        save_json(paths.root / "baselines_eval.json", baseline_results)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    loss_weights: Mapping[str, float],
    metric_names: Sequence[str],
    target_names: Sequence[str],
    device: torch.device,
    split: str,
    fold: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    head_order = list(getattr(model, "head_names", target_names))
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            w = batch.get("w")
            if w is not None:
                w = w.to(device)
            with autocast(enabled=False):
                outputs = forward_model(model, x, w)
                loss, tensors = compute_loss(
                    outputs,
                    y,
                    loss_fn,
                    loss_weights,
                    None,
                    target_names,
                    head_order,
                )
            losses.append(loss.item())
            preds.append(_flatten_numpy(tensors["pred"]))
            trues.append(_flatten_numpy(tensors["true"]))
            rows.extend(
                build_prediction_rows(
                    batch,
                    tensors["pred"],
                    tensors["true"],
                    head_order,
                    split,
                    fold,
                    sample_index,
                )
            )
            sample_index += len(batch["x"])
    preds_np = np.concatenate(preds, axis=0)
    trues_np = np.concatenate(trues, axis=0)
    metrics = compute_metrics(trues_np, preds_np, metric_names, head_order or target_names)
    metrics["loss"] = float(np.mean(losses))
    return metrics, rows


def build_prediction_rows(
    batch: Mapping[str, Any],
    pred_tensor: torch.Tensor,
    true_tensor: torch.Tensor,
    head_order: Sequence[str],
    split: str,
    fold: int,
    start_index: int,
) -> List[Dict[str, Any]]:
    ids = batch.get("id")
    w = batch.get("w")
    preds = pred_tensor.detach().cpu().numpy()
    trues = true_tensor.detach().cpu().numpy()
    rows: List[Dict[str, Any]] = []
    if preds.ndim == 3:
        batch_size, grid, heads = preds.shape
        for i in range(batch_size):
            sample_id = ids[i] if ids else f"{split}_{start_index + i}"
            for g in range(grid):
                row: Dict[str, Any] = {
                    "id": sample_id,
                    "split": split,
                    "fold": fold,
                }
                if w is not None:
                    w_value = w[i, g].cpu().numpy()
                    row["w"] = float(w_value) if np.isscalar(w_value) else w_value.tolist()
                for idx, name in enumerate(head_order or [f"target_{k}" for k in range(heads)]):
                    row[f"y_true_{name}"] = float(trues[i, g, idx])
                    row[f"y_pred_{name}"] = float(preds[i, g, idx])
                rows.append(row)
    else:
        batch_size, heads = preds.shape
        for i in range(batch_size):
            sample_id = ids[i] if ids else f"{split}_{start_index + i}"
            row = {
                "id": sample_id,
                "split": split,
                "fold": fold,
            }
            for idx, name in enumerate(head_order or [f"target_{k}" for k in range(heads)]):
                row[f"y_true_{name}"] = float(trues[i, idx])
                row[f"y_pred_{name}"] = float(preds[i, idx])
            rows.append(row)
    return rows


def create_plots(df: pd.DataFrame, target_names: Sequence[str], figure_dir: Path) -> None:
    if df.empty:
        return
    ensure_columns = [f"y_true_{name}" for name in target_names]
    if not all(col in df.columns for col in ensure_columns):
        return
    test_df = df[df["split"] == "test"] if "split" in df.columns else df
    for name in target_names:
        parity_path = figure_dir / f"parity_{name}.png"
        plt.figure(figsize=(5, 5))
        plt.scatter(test_df[f"y_true_{name}"], test_df[f"y_pred_{name}"], s=10, alpha=0.6)
        min_val = min(test_df[f"y_true_{name}"].min(), test_df[f"y_pred_{name}"].min())
        max_val = max(test_df[f"y_true_{name}"].max(), test_df[f"y_pred_{name}"].max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Parity plot - {name}")
        plt.tight_layout()
        plt.savefig(parity_path)
        plt.close()

    error_path = figure_dir / "error_hist.png"
    plt.figure(figsize=(6, 4))
    for name in target_names:
        errors = test_df[f"y_pred_{name}"] - test_df[f"y_true_{name}"]
        plt.hist(errors, bins=40, alpha=0.5, label=name)
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")
    plt.title("Error histogram")
    if len(target_names) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(error_path)
    plt.close()

    if "w" in test_df.columns:
        sample_ids = test_df["id"].unique().tolist()
        random.shuffle(sample_ids)
        for sample_id in sample_ids[:3]:
            sample_df = test_df[test_df["id"] == sample_id]
            if sample_df.empty:
                continue
            first_w = sample_df.iloc[0]["w"]
            if not np.isscalar(first_w):
                continue
            curve_path = figure_dir / f"curve_{sample_id}.png"
            plt.figure(figsize=(6, 4))
            sorted_df = sample_df.sort_values("w")
            for name in target_names:
                plt.plot(sorted_df["w"], sorted_df[f"y_true_{name}"], label=f"True {name}")
                plt.plot(sorted_df["w"], sorted_df[f"y_pred_{name}"], linestyle="--", label=f"Pred {name}")
            plt.xlabel("w")
            plt.ylabel("Response")
            plt.title(f"Prediction curves - {sample_id}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(curve_path)
            plt.close()


if __name__ == "__main__":
    main()
