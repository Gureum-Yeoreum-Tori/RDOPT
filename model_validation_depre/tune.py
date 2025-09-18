from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import optuna
import torch
from optuna.pruners import MedianPruner
from torch.cuda.amp import GradScaler

from .data import build_scalers, create_dataloaders, create_kfold_indices, prepare_datasets, apply_scaler
from .train import (
    _make_subset,
    _parse_loss_weights,
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
    run_epoch,
)
from .utils import (
    ExperimentPaths,
    best_metric,
    configure_experiment,
    configure_logging,
    get_device,
    is_higher_better,
    load_config,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kfold", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config, args.override)
    training_cfg = base_config.get("training", {})
    logging_cfg = base_config.get("logging", {})
    data_cfg = base_config.get("data", {})
    metrics_cfg = base_config.get("metrics", {})

    seed = base_config.get("seed", 42)
    set_seed(seed)

    exp_name = args.exp_name or f"tune_{base_config.get('experiment', 'default')}"
    save_dir = Path(args.save_dir or logging_cfg.get("save_dir", "runs"))
    paths = configure_experiment(save_dir, exp_name)
    logger = configure_logging(paths.logs, f"tune_{exp_name}")

    primary_metric = metrics_cfg.get("primary", "r2")
    direction = "maximize" if is_higher_better(primary_metric) else "minimize"

    study_path = paths.root / "tuning.db"
    study = optuna.create_study(
        study_name=exp_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction=direction,
        pruner=MedianPruner(),
    )

    device = get_device(args.device)
    logger.info("Starting tuning with %d trials", args.n_trials)

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        apply_trial_suggestions(trial, config)

        local_paths = ExperimentPaths(
            root=paths.root / f"trial_{trial.number}",
            checkpoints=(paths.root / f"trial_{trial.number}" / "checkpoints"),
            logs=(paths.root / f"trial_{trial.number}" / "logs"),
            figures=(paths.root / f"trial_{trial.number}" / "figures"),
            predictions=(paths.root / f"trial_{trial.number}" / "predictions"),
        )
        for directory in [local_paths.root, local_paths.checkpoints, local_paths.logs, local_paths.figures, local_paths.predictions]:
            directory.mkdir(parents=True, exist_ok=True)

        trial_logger = configure_logging(local_paths.logs, f"trial_{trial.number}")
        training_cfg_local = config.setdefault("training", {})
        data_cfg_local = config.setdefault("data", {})
        metrics_cfg_local = config.setdefault("metrics", {})
        model_cfg_local = config.setdefault("model", {})

        kfold = args.kfold or int(data_cfg_local.get("kfold", 0))
        if kfold > 1:
            data_cfg_local["split"] = False
        train_dataset, val_dataset, test_dataset, scalers, features, targets, ids, weights, w_grid = prepare_datasets(
            data_cfg_local,
            local_paths.root,
            seed,
        )

        target_names = [str(name) for name in data_cfg_local.get("targets", [])]
        if not target_names:
            target_names = [f"target_{idx}" for idx in range(train_dataset.targets.shape[-1])]

        if kfold > 1:
            folds = create_kfold_indices(features.shape[0], kfold, seed)
            fold_idx = args.fold if args.fold < len(folds) else 0
            train_idx, val_idx = folds[fold_idx]
            train_dataset = _make_subset(features, targets, ids, weights, w_grid, train_idx)
            val_dataset = _make_subset(features, targets, ids, weights, w_grid, val_idx)
            scalers = build_scalers(data_cfg_local.get("scaler", "none"))
            apply_scaler(scalers, train_dataset, [val_dataset])

        loaders = create_dataloaders(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset,
            batch_size=int(training_cfg_local.get("batch_size", 64)),
        )

        model = build_model(model_cfg_local, training_cfg_local, target_names, train_dataset, device)
        optimizer = build_optimizer(training_cfg_local, model)
        scheduler = build_scheduler(training_cfg_local, optimizer, len(loaders["train"]), primary_metric)
        loss_fn = build_loss(training_cfg_local.get("loss", "mse"))
        loss_weights = _parse_loss_weights(training_cfg_local.get("loss_weights"), target_names, model)
        scaler = GradScaler(enabled=device.type == "cuda" and training_cfg_local.get("mixed_precision", False))

        epochs = int(training_cfg_local.get("epochs", 100))
        grad_clip = float(training_cfg_local.get("grad_clip", 0.0) or 0.0)
        metric_names = metrics_cfg_local.get("report", ["rmse", "mae", "mape", "r2"])
        best_score = None
        patience_cfg = training_cfg_local.get("early_stopping", {})
        patience = int(patience_cfg.get("patience", 0))
        min_delta = float(patience_cfg.get("min_delta", 0.0))
        no_improve = 0

        metric_value = None
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_fn,
                loss_weights,
                metric_names,
                target_names,
                device,
                scaler,
                training_cfg_local.get("mixed_precision", False),
                grad_clip,
                training=True,
            )
            val_metrics = None
            if "val" in loaders:
                val_metrics = run_epoch(
                    model,
                    loaders["val"],
                    optimizer,
                    loss_fn,
                    loss_weights,
                    metric_names,
                    target_names,
                    device,
                    scaler,
                    training_cfg_local.get("mixed_precision", False),
                    grad_clip,
                    training=False,
                )
                metric_value = val_metrics.get(primary_metric.lower())
            else:
                metric_value = train_metrics.get(primary_metric.lower())

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and metric_value is not None:
                scheduler.step(metric_value)

            trial.report(metric_value or 0.0, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if metric_value is not None and best_metric(best_score, metric_value, primary_metric):
                best_score = metric_value
                no_improve = 0
                checkpoint_state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best": best_score,
                    "config": config,
                }
                if scheduler:
                    checkpoint_state["scheduler"] = scheduler.state_dict()
                save_checkpoint(checkpoint_state, local_paths.checkpoints / "best.ckpt", trial_logger)
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience and best_score is not None and metric_value is not None:
                if abs(metric_value - best_score) < min_delta:
                    break

        fallback = metric_value if metric_value is not None else 0.0
        final_metric = best_score if best_score is not None else fallback
        save_json(local_paths.root / "metrics.json", {"best": final_metric})
        with open(local_paths.root / "config.json", "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
        return float(final_metric)

    study.optimize(objective, n_trials=args.n_trials)
    logger.info("Study finished. Best trial: %s", study.best_trial.number)
    best_trial_dir = paths.root / f"trial_{study.best_trial.number}" / "checkpoints"
    best_checkpoint = best_trial_dir / "best.ckpt"
    if best_checkpoint.exists():
        target_checkpoint = paths.checkpoints / "tune_best.ckpt"
        target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        target_checkpoint.write_bytes(best_checkpoint.read_bytes())
    best_config_path = paths.root / f"trial_{study.best_trial.number}" / "config.json"
    if best_config_path.exists():
        (paths.root / "best_trial_config.json").write_bytes(best_config_path.read_bytes())


def apply_trial_suggestions(trial: optuna.Trial, config: Dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})

    training["lr"] = trial.suggest_float("training.lr", 1e-5, 1e-2, log=True)
    training["batch_size"] = trial.suggest_categorical("training.batch_size", [128, 256, 512, 1024])
    training["weight_decay"] = trial.suggest_float("training.weight_decay", 1e-8, 1e-2, log=True)
    training["optimizer"] = trial.suggest_categorical("training.optimizer", ["adam", "adamw", "sgd"])
    training["scheduler"] = trial.suggest_categorical("training.scheduler", ["none", "one-cycle", "cosine", "plateau"])
    training["loss"] = trial.suggest_categorical("training.loss", ["mse", "l1", "huber"])
    training["grad_clip"] = trial.suggest_float("training.grad_clip", 0.0, 2.0)

    model_type = (training.get("model") or model_cfg.get("type") or "MLP").lower()
    activation = trial.suggest_categorical("model.activation", ["relu", "gelu", "tanh"])
    model_cfg["activation"] = activation
    model_cfg["dropout"] = trial.suggest_float("model.dropout", 0.0, 0.4)
    model_cfg["layernorm"] = trial.suggest_categorical("model.layernorm", [False, True])
    model_cfg["init"] = trial.suggest_categorical("model.init", ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"])

    hidden_layers = []
    num_layers = trial.suggest_int("model.hidden_layers", 1, 4)
    for idx in range(num_layers):
        hidden_layers.append(trial.suggest_int(f"model.hidden_width_{idx}", 64, 512))
    model_cfg["hidden_layers"] = hidden_layers

    if model_type in {"multiheadmlp"}:
        model_cfg["head_dim"] = trial.suggest_int("model.head_dim", 1, 4)

    if model_type in {"deeponet", "multiheaddeeponet"}:
        deeponet = model_cfg.setdefault("deeponet", {})
        branch_layers = []
        branch_len = trial.suggest_int("deeponet.branch_depth", 1, 3)
        for idx in range(branch_len):
            branch_layers.append(trial.suggest_int(f"deeponet.branch_width_{idx}", 64, 512))
        trunk_layers = []
        trunk_len = trial.suggest_int("deeponet.trunk_depth", 1, 3)
        for idx in range(trunk_len):
            trunk_layers.append(trial.suggest_int(f"deeponet.trunk_width_{idx}", 64, 512))
        deeponet["branch_layers"] = branch_layers
        deeponet["trunk_layers"] = trunk_layers
        deeponet["latent_dim"] = trial.suggest_int("deeponet.latent_dim", 32, 256)

    if data_cfg.get("kfold", 0):
        training["epochs"] = trial.suggest_int("training.epochs", 50, 200)
    else:
        training["epochs"] = trial.suggest_int("training.epochs", 50, 400)


if __name__ == "__main__":
    main()
