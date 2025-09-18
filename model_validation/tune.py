import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import optuna

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
    parser = argparse.ArgumentParser(description="Optuna-based hyperparameter tuning")
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
    parser.add_argument("--baseline-alpha", type=float)
    parser.add_argument("--json-log")
    parser.add_argument("--study-name")
    parser.add_argument("--storage")
    parser.add_argument("--sampler", choices=["tpe", "random", "cmaes"], default="tpe")
    parser.add_argument("--pruner", choices=["none", "median", "hyperband"], default="none")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--timeout", type=float)
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


def build_space(args: argparse.Namespace, base: TrainSettings) -> Dict[str, List]:
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


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    lower = name.lower()
    if lower == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if lower == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def make_pruner(name: str) -> Optional[optuna.pruners.BasePruner]:
    lower = name.lower()
    if lower == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=1)
    if lower == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return None


def suggest_trial_params(trial: optuna.Trial, search_space: Dict[str, List]) -> Dict[str, object]:
    updates: Dict[str, object] = {}
    for key, values in search_space.items():
        options = [cast_param(v) for v in values]
        if len(options) == 1:
            updates[key] = options[0]
            continue
        updates[key] = trial.suggest_categorical(key, options)
    return updates


def run_optuna(
    base_settings: TrainSettings,
    search_space: Dict[str, List],
    sampler: optuna.samplers.BaseSampler,
    pruner: Optional[optuna.pruners.BasePruner],
    n_trials: Optional[int],
    timeout: Optional[float],
    study_name: Optional[str],
    storage: Optional[str],
    exp_prefix: Optional[str],
    json_log: Optional[str],
) -> optuna.Study:
    if base_settings.model == "deeponet" and base_settings.target != "rdc":
        raise ValueError("DeepONet tuning requires the rdc target")
    if base_settings.model == "mlp" and base_settings.target != "leak":
        raise ValueError("MLP tuning requires the leak target")

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )

    prefix = exp_prefix or f"{base_settings.model}_{base_settings.target}"

    def objective(trial: optuna.Trial) -> float:
        updates = suggest_trial_params(trial, search_space)
        trial_settings = replace(base_settings, **updates)
        trial_settings.exp_name = f"{prefix}_trial{trial.number:03d}"
        outcome = run_training(trial_settings)
        val_rmse = float(outcome["metrics"]["val"]["rmse"])
        trial.set_user_attr("metrics", outcome["metrics"])
        trial.set_user_attr("checkpoint", outcome["checkpoint"])
        trial.set_user_attr("params_full", updates)
        print(json.dumps({
            "trial": trial.number,
            "params": updates,
            "val_rmse": val_rmse,
            "checkpoint": outcome["checkpoint"],
        }))
        return val_rmse

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    summary = {
        "best_trial": {
            "number": study.best_trial.number if study.best_trial is not None else None,
            "value": study.best_value if study.best_trial is not None else None,
            "params": study.best_params if study.best_trial is not None else None,
            "user_attrs": study.best_trial.user_attrs if study.best_trial is not None else None,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "state": str(t.state),
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    print(json.dumps(summary, indent=2))
    if json_log:
        Path(json_log).write_text(json.dumps(summary, indent=2))
    return study


def main() -> None:
    args = parse_args()
    base_settings = build_base_settings(args)
    search_space = build_space(args, base_settings)
    sampler = make_sampler(args.sampler, base_settings.seed)
    pruner = make_pruner(args.pruner)
    n_trials = args.n_trials if args.n_trials is not None else 20
    run_optuna(
        base_settings=base_settings,
        search_space=search_space,
        sampler=sampler,
        pruner=pruner,
        n_trials=n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
        exp_prefix=args.exp_prefix,
        json_log=args.json_log,
    )


if __name__ == "__main__":
    main()
