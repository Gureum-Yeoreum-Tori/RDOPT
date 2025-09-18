import argparse
import itertools
import json
import sys

import numpy as np
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
    sys.path.append(str(ROOT.parent))
    from model_validation.train import DEFAULTS, DEFAULT_MAT_FILES, TrainSettings, run_training
else:
    from .train import DEFAULTS, DEFAULT_MAT_FILES, TrainSettings, run_training


def first_value(values: Sequence, default):
    if values is None:
        return default
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return values if values is not None else default


def to_list(values: Sequence, fallback) -> List:
    if values is None:
        return [fallback]
    if isinstance(values, (list, tuple)):
        return list(values) if values else [fallback]
    return [values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--model", choices=["deeponet", "mlp"], default="deeponet")
    parser.add_argument("--target", choices=["rdc", "leak"], default="rdc")
    parser.add_argument("--mat-files", nargs="*", default=list(DEFAULT_MAT_FILES))
    parser.add_argument("--data-dir", default="dataset/data/tapered_seal")
    parser.add_argument("--leak-index", type=int)
    parser.add_argument("--rdc-indices", nargs="*", type=int)
    parser.add_argument("--hidden-channels", nargs="*", type=int)
    parser.add_argument("--param-embedding", nargs="*", type=int)
    parser.add_argument("--n-basis", nargs="*", type=int)
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--dropout", nargs="*", type=float)
    parser.add_argument("--lr", nargs="*", type=float)
    parser.add_argument("--weight-decay", nargs="*", type=float)
    parser.add_argument("--batch-size", nargs="*", type=int)
    parser.add_argument("--epochs", nargs="*", type=int)
    parser.add_argument("--warmup", nargs="*", type=int)
    parser.add_argument("--patience", nargs="*", type=int)
    parser.add_argument("--grad-clip", nargs="*", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--exp-prefix")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--baseline-alpha", type=float)
    parser.add_argument("--json-log")
    return parser.parse_args()


def build_base_settings(args: argparse.Namespace) -> TrainSettings:
    base = DEFAULTS[args.model]
    mat_files = tuple(args.mat_files) if args.mat_files else DEFAULT_MAT_FILES
    rdc_indices = tuple(args.rdc_indices) if args.rdc_indices else (2, 3, 4, 5)
    return TrainSettings(
        model=args.model,
        target=args.target,
        data_dir=args.data_dir,
        mat_files=mat_files,
        leak_index=args.leak_index if args.leak_index is not None else 6,
        rdc_indices=rdc_indices,
        batch_size=first_value(args.batch_size, base["batch_size"]),
        epochs=first_value(args.epochs, base["epochs"]),
        lr=first_value(args.lr, base["lr"]),
        weight_decay=first_value(args.weight_decay, base["weight_decay"]),
        hidden_channels=base["hidden_channels"],
        param_embedding_dim=base["param_embedding_dim"],
        n_layers=base["n_layers"],
        dropout=base["dropout"],
        n_basis=base["n_basis"],
        warmup=first_value(args.warmup, base["warmup"]),
        patience=first_value(args.patience, base["patience"]),
        grad_clip=first_value(args.grad_clip, base["grad_clip"]),
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir or "net",
        exp_name=None,
        baseline_alpha=args.baseline_alpha if args.baseline_alpha is not None else 1.0,
    )


def build_grid(args: argparse.Namespace, base: TrainSettings) -> Dict[str, List]:
    if args.model == "deeponet":
        return {
            "hidden_channels": to_list(args.hidden_channels, base.hidden_channels),
            "param_embedding_dim": to_list(args.param_embedding, base.param_embedding_dim),
            "n_layers": to_list(args.layers, base.n_layers),
            "n_basis": to_list(args.n_basis, base.n_basis),
            "dropout": to_list(args.dropout, base.dropout),
            "lr": to_list(args.lr, base.lr),
            "weight_decay": to_list(args.weight_decay, base.weight_decay),
            "batch_size": to_list(args.batch_size, base.batch_size),
            "epochs": to_list(args.epochs, base.epochs),
            "warmup": to_list(args.warmup, base.warmup),
            "patience": to_list(args.patience, base.patience),
            "grad_clip": to_list(args.grad_clip, base.grad_clip),
        }
    return {
        "hidden_channels": to_list(args.hidden_channels, base.hidden_channels),
        "n_layers": to_list(args.layers, base.n_layers),
        "dropout": to_list(args.dropout, base.dropout),
        "lr": to_list(args.lr, base.lr),
        "weight_decay": to_list(args.weight_decay, base.weight_decay),
        "batch_size": to_list(args.batch_size, base.batch_size),
        "epochs": to_list(args.epochs, base.epochs),
        "patience": to_list(args.patience, base.patience),
        "grad_clip": to_list(args.grad_clip, base.grad_clip),
    }


def cast_param(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.model == "deeponet" and args.target != "rdc":
        raise ValueError("DeepONet tuning requires the rdc target")
    if args.model == "mlp" and args.target != "leak":
        raise ValueError("MLP tuning requires the leak target")
    base_settings = build_base_settings(args)
    grid = build_grid(args, base_settings)
    order = list(grid.keys())
    exp_prefix = args.exp_prefix or f"{args.model}_{args.target}"
    results: List[Dict] = []
    for trial_idx, values in enumerate(itertools.product(*(grid[key] for key in order)), start=1):
        trial_params = dict(zip(order, values))
        trial_settings = replace(base_settings, **trial_params)
        trial_settings.exp_name = f"{exp_prefix}_trial{trial_idx:03d}"
        outcome = run_training(trial_settings)
        val_rmse = outcome["metrics"]["val"]["rmse"]
        record = {
            "trial": trial_idx,
            "params": {k: cast_param(v) for k, v in trial_params.items()},
            "val_rmse": float(val_rmse),
            "metrics": outcome["metrics"],
            "checkpoint": outcome["checkpoint"],
        }
        results.append(record)
        print(json.dumps(record))
        if args.max_trials and trial_idx >= args.max_trials:
            break
    ranked = sorted(results, key=lambda item: item["val_rmse"]) if results else []
    summary = {"results": results, "ranked": ranked}
    print(json.dumps(summary, indent=2))
    if args.json_log:
        Path(args.json_log).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
