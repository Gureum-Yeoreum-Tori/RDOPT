from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import (
    SealDataset,
    apply_scaler,
    build_scalers,
    create_dataloaders,
    create_kfold_indices,
    prepare_datasets,
)
from .models import DeepONet, MLP, MultiHeadDeepONet, MultiHeadMLP
from .utils import (
    ExperimentPaths,
    best_metric,
    compute_metrics,
    configure_experiment,
    configure_logging,
    count_parameters,
    ensure_dir,
    get_device,
    is_higher_better,
    load_config,
    run_baselines,
    save_checkpoint,
    save_json,
    set_seed,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal model training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--kfold", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    training_cfg = config.setdefault("training", {})
    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})
    metrics_cfg = config.setdefault("metrics", {})
    logging_cfg = config.setdefault("logging", {})

    seed = args.seed or config.get("seed", 42)
    set_seed(seed)

    exp_name = args.exp_name or config.get("experiment", "default_experiment")
    save_dir = Path(args.save_dir or logging_cfg.get("save_dir", "runs"))
    paths = configure_experiment(save_dir, exp_name)
    logger = configure_logging(paths.logs, exp_name)

    logger.info("Configuration:\n%s", json.dumps(config, indent=2))

    device = get_device(args.device)

    mixed_precision = (
        training_cfg.get("mixed_precision", False)
        if args.mixed_precision is None
        else args.mixed_precision
    )

    kfold = args.kfold or int(data_cfg.get("kfold", 0))
    target_names = [str(name) for name in data_cfg.get("targets", [])]

    if kfold > 1:
        data_cfg["split"] = False
    train_dataset, val_dataset, test_dataset, scalers, features, targets, ids, weights, w_grid = prepare_datasets(
        data_cfg,
        paths.root,
        seed,
    )

    if not target_names:
        target_names = [f"target_{idx}" for idx in range(train_dataset.targets.shape[-1])]

    if kfold > 1:
        folds = create_kfold_indices(features.shape[0], kfold, seed)
        if args.fold >= len(folds):
            raise ValueError(f"Fold {args.fold} out of range")
        train_idx, val_idx = folds[args.fold]
        train_dataset = _make_subset(
            features,
            targets,
            ids,
            weights,
            w_grid,
            train_idx,
        )
        val_dataset = _make_subset(
            features,
            targets,
            ids,
            weights,
            w_grid,
            val_idx,
        )
        scalers = build_scalers(data_cfg.get("scaler", "none"))
        apply_scaler(scalers, train_dataset, [val_dataset])
        ensure_dir(paths.root)
    elif train_dataset is None:
        raise RuntimeError("Failed to build training dataset")

    batch_size = int(training_cfg.get("batch_size", 64))
    loaders = create_dataloaders(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        batch_size=batch_size,
    )

    model = build_model(model_cfg, training_cfg, target_names, train_dataset, device)
    logger.info("Model: %s", model)
    logger.info("Trainable parameters: %d", count_parameters(model))

    optimizer = build_optimizer(training_cfg, model)
    scheduler = build_scheduler(training_cfg, optimizer, len(loaders["train"]), primary_metric)
    loss_fn = build_loss(training_cfg.get("loss", "mse"))
    loss_weights = _parse_loss_weights(training_cfg.get("loss_weights"), target_names, model)

    scaler = GradScaler(enabled=mixed_precision and device.type == "cuda")
    start_epoch = 1
    best_score: Optional[float] = None
    primary_metric = metrics_cfg.get("primary", "r2")
    patience_cfg = training_cfg.get("early_stopping", {})
    patience = int(patience_cfg.get("patience", 0))
    min_delta = float(patience_cfg.get("min_delta", 0.0))
    no_improve = 0

    if args.resume:
        start_epoch, best_score = load_state(args.resume, model, optimizer, scheduler, scaler, device)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    epochs = int(training_cfg.get("epochs", 100))
    grad_clip = float(training_cfg.get("grad_clip", 0.0) or 0.0)

    metric_names = metrics_cfg.get("report", ["rmse", "mae", "mape", "r2"])

    for epoch in range(start_epoch, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)
        train_stats = run_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_fn,
            loss_weights,
            metric_names,
            target_names,
            device,
            scaler,
            mixed_precision,
            grad_clip,
            training=True,
        )
        logger.info("Train metrics: %s", json.dumps(train_stats, indent=2))

        val_stats: Optional[Dict[str, float]] = None
        if "val" in loaders:
            val_stats = run_epoch(
                model,
                loaders["val"],
                optimizer,
                loss_fn,
                loss_weights,
                metric_names,
                target_names,
                device,
                scaler,
                mixed_precision,
                grad_clip,
                training=False,
            )
            logger.info("Validation metrics: %s", json.dumps(val_stats, indent=2))
            metric_value = val_stats.get(primary_metric.lower())
        else:
            metric_value = train_stats.get(primary_metric.lower())

        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau) and metric_value is not None:
            scheduler.step(metric_value)

        improved = metric_value is not None and best_metric(best_score, metric_value, primary_metric)
        current_best = best_score if best_score is not None else metric_value
        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "best": current_best,
        }
        if scheduler:
            checkpoint_state["scheduler"] = scheduler.state_dict()

        save_checkpoint(checkpoint_state, paths.checkpoints / "last.ckpt", logger)

        if improved:
            best_score = metric_value
            no_improve = 0
            checkpoint_state["best"] = best_score
            save_checkpoint(checkpoint_state, paths.checkpoints / "best.ckpt", logger)
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience and best_score is not None and metric_value is not None:
            if abs(metric_value - best_score) < min_delta:
                logger.info("Early stopping triggered")
                break

    if "test" in loaders:
        test_stats = run_epoch(
            model,
            loaders["test"],
            optimizer,
            loss_fn,
            loss_weights,
            metric_names,
            target_names,
            device,
            scaler,
            mixed_precision,
            grad_clip,
            training=False,
        )
        logger.info("Test metrics: %s", json.dumps(test_stats, indent=2))
        save_json(paths.root / "metrics_test.json", test_stats)

    baseline_cfg = config.get("baselines", {})
    if baseline_cfg.get("run", False) and test_dataset is not None:
        if train_dataset.targets.ndim > 2:
            logger.info("Skipping baselines for operator data")
        else:
            x_train = train_dataset.features
            y_train = train_dataset.targets
            x_test = test_dataset.features
            y_test = test_dataset.targets
            baseline_results = run_baselines(
                x_train,
                y_train,
                x_test,
                y_test,
                baseline_cfg,
                target_names,
                metrics_cfg.get("report", ["rmse", "mae", "mape", "r2"]),
                logger,
            )
            save_json(paths.root / "baselines.json", baseline_results)


def build_model(
    model_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    target_names: Sequence[str],
    train_dataset: SealDataset,
    device: torch.device,
) -> nn.Module:
    model_type = (
        training_cfg.get("model")
        or model_cfg.get("type")
        or model_cfg.get("name")
        or "MLP"
    ).lower()
    input_dim = model_cfg.get("input_dim") or train_dataset.features.shape[1]
    output_dim = model_cfg.get("output_dim", len(target_names) or 1)
    activation = model_cfg.get("activation", "relu")
    hidden_layers = model_cfg.get("hidden_layers", [128, 128])
    dropout = float(model_cfg.get("dropout", 0.0) or 0.0)
    layernorm = bool(model_cfg.get("layernorm", False))
    init = model_cfg.get("init", "xavier_uniform")

    if model_type == "mlp":
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
        )
    elif model_type == "multiheadmlp":
        head_names = model_cfg.get("heads") or target_names
        if len(head_names) <= 1:
            model = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                dropout=dropout,
                layernorm=layernorm,
            )
        else:
            model = MultiHeadMLP(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                head_names=head_names,
                head_dim=model_cfg.get("head_dim", 1),
                activation=activation,
                dropout=dropout,
                layernorm=layernorm,
            )
    elif model_type == "deeponet":
        deeponet_cfg = model_cfg.get("deeponet", {})
        trunk_input_dim = (
            train_dataset.w_grid.shape[-1]
            if getattr(train_dataset, "w_grid", None) is not None
            else deeponet_cfg.get("trunk_input_dim", 1)
        )
        if getattr(train_dataset, "w_grid", None) is None:
            raise ValueError("DeepONet requires w_grid data")
        model = DeepONet(
            input_dim=input_dim,
            output_dim=output_dim,
            branch_layers=deeponet_cfg.get("branch_layers", [128, 128]),
            trunk_layers=deeponet_cfg.get("trunk_layers", [128, 128]),
            latent_dim=deeponet_cfg.get("latent_dim", 128),
            activation=activation,
            dropout=dropout,
            trunk_input_dim=trunk_input_dim,
        )
    elif model_type == "multiheaddeeponet":
        deeponet_cfg = model_cfg.get("deeponet", {})
        head_names = model_cfg.get("heads") or target_names
        if getattr(train_dataset, "w_grid", None) is None:
            raise ValueError("MultiHeadDeepONet requires w_grid data")
        trunk_input_dim = train_dataset.w_grid.shape[-1]
        model = MultiHeadDeepONet(
            input_dim=input_dim,
            head_names=head_names,
            branch_layers=deeponet_cfg.get("branch_layers", [128, 128]),
            trunk_layers=deeponet_cfg.get("trunk_layers", [128, 128]),
            latent_dim=deeponet_cfg.get("latent_dim", 128),
            activation=activation,
            dropout=dropout,
            trunk_input_dim=trunk_input_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    initialize_weights(model, init)
    return model


def initialize_weights(model: nn.Module, init: str) -> None:
    init = init.lower()

    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif init == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_init)


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module) -> Optimizer:
    optimizer_name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0) or 0.0)
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    cfg: Mapping[str, Any],
    optimizer: Optimizer,
    steps_per_epoch: int,
    primary_metric: str,
) -> Optional[_LRScheduler]:
    scheduler_name = cfg.get("scheduler", "none").lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "one-cycle":
        max_lr = float(cfg.get("lr", 1e-3))
        epochs = int(cfg.get("epochs", 100))
        return OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    if scheduler_name == "cosine":
        epochs = int(cfg.get("epochs", 100))
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "plateau":
        mode = "max" if is_higher_better(primary_metric) else "min"
        return ReduceLROnPlateau(optimizer, mode=mode)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "l1":
        return nn.L1Loss(reduction="none")
    if name == "huber":
        return nn.SmoothL1Loss(reduction="none")
    raise ValueError(f"Unsupported loss: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    loss_weights: Dict[str, float],
    metric_names: Sequence[str],
    target_names: Sequence[str],
    device: torch.device,
    scaler: GradScaler,
    mixed_precision: bool,
    grad_clip: float,
    training: bool,
) -> Dict[str, float]:
    model.train(training)
    progress = tqdm(loader, leave=False)
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for batch in progress:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        w = batch.get("w")
        if w is not None:
            w = w.to(device)
        sample_weight = batch.get("weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(device)
        with torch.set_grad_enabled(training):
            with autocast(enabled=mixed_precision):
                outputs = forward_model(model, x, w)
                loss, per_head = compute_loss(
                    outputs,
                    y,
                    loss_fn,
                    loss_weights,
                    sample_weight,
                    target_names,
                    getattr(model, "head_names", target_names),
                )
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(loss.item())
        preds.append(_flatten_numpy(per_head["pred"]))
        trues.append(_flatten_numpy(per_head["true"]))
    preds_np = np.concatenate(preds, axis=0)
    trues_np = np.concatenate(trues, axis=0)
    metric_targets = list(getattr(model, "head_names", target_names or ["target"]))
    metrics = compute_metrics(
        y_true=trues_np,
        y_pred=preds_np,
        metric_names=metric_names,
        target_names=metric_targets,
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def forward_model(model: nn.Module, x: Tensor, w: Optional[Tensor]) -> Any:
    if isinstance(model, (DeepONet, MultiHeadDeepONet)):
        if w is None:
            raise ValueError("DeepONet variants require w input")
        return model(x, w)
    return model(x)


def compute_loss(
    outputs: Any,
    targets: Tensor,
    loss_fn: nn.Module,
    loss_weights: Mapping[str, float],
    sample_weight: Optional[Tensor],
    target_names: Sequence[str],
    head_order: Sequence[str],
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if isinstance(outputs, dict):
        target_map = {name: idx for idx, name in enumerate(target_names)}
        losses: List[Tensor] = []
        true_tensors: List[Tensor] = []
        merged = []
        for name in head_order:
            if name not in outputs:
                continue
            pred = outputs[name]
            target_idx = target_map.get(name)
            if target_idx is None:
                raise ValueError(f"Target {name} not found in data targets")
            target = targets[..., target_idx : target_idx + 1]
            loss = loss_fn(pred, target)
            if sample_weight is not None:
                loss = loss * sample_weight.view(sample_weight.shape[0], *([1] * (loss.dim() - 1)))
            loss = loss.mean()
            weight = loss_weights.get(name, 1.0)
            losses.append(loss * weight)
            merged.append(pred)
            true_tensors.append(target)
        total = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=targets.device)
        pred_tensor = torch.cat(merged, dim=-1) if merged else torch.zeros_like(targets)
        true_tensor = torch.cat(true_tensors, dim=-1) if true_tensors else targets
        return total, {"pred": pred_tensor.detach(), "true": true_tensor.detach()}
    loss = loss_fn(outputs, targets)
    if sample_weight is not None:
        expand_dims = (1,) * (loss.dim() - 1)
        loss = loss * sample_weight.view(sample_weight.shape[0], *expand_dims)
    total = loss.mean()
    return total, {"pred": outputs.detach(), "true": targets.detach()}


def _parse_loss_weights(
    weights_cfg: Optional[Any],
    target_names: Sequence[str],
    model: nn.Module,
) -> Dict[str, float]:
    if weights_cfg is None:
        if isinstance(model, (MultiHeadMLP, MultiHeadDeepONet)):
            return {name: 1.0 for name in getattr(model, "head_names", target_names)}
        return {name: 1.0 for name in target_names}
    if isinstance(weights_cfg, Mapping):
        return {str(k): float(v) for k, v in weights_cfg.items()}
    if isinstance(weights_cfg, (list, tuple)):
        return {
            name: float(weights_cfg[idx])
            for idx, name in enumerate(target_names)
            if idx < len(weights_cfg)
        }
    raise ValueError("Invalid loss_weights configuration")


def _make_subset(
    x: np.ndarray,
    y: np.ndarray,
    ids: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    w_grid: Optional[np.ndarray],
    indices: np.ndarray,
) -> SealDataset:
    return SealDataset(
        features=x[indices],
        targets=y[indices],
        ids=None if ids is None else ids[indices],
        weights=None if weights is None else weights[indices],
        w_grid=None if w_grid is None else w_grid[indices],
    )


def load_state(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, Optional[float]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)) + 1, checkpoint.get("best", None)


def _flatten_numpy(tensor: Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim > 2:
        return array.reshape(-1, array.shape[-1])
    if array.ndim == 1:
        return array[:, None]
    return array


if __name__ == "__main__":
    main()
