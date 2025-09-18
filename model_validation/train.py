import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parent
if __package__ is None or __package__ == "":
    sys.path.append(str(ROOT.parent))
    from model_validation.models import MultiHeadDeepONet, MLP
else:
    from .models import MultiHeadDeepONet, MLP


DEFAULT_MAT_FILES = (
    "20250908_T_182846",
    "20250911_T_091324",
    "20250908_T_183632",
    "20250908_T_203220",
)

DEFAULTS = {
    "deeponet": {
        "batch_size": 512,
        "epochs": 5000,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "hidden_layers": [64, 64, 64, 64, 64, 64, 64, 64],  # 8 layers
        "param_embedding_dim": 64,
        "dropout": 0.0,
        "n_basis": 64,
        "warmup": 500,
        "patience": 100,
        "grad_clip": 0.0,
    },
    "mlp": {
        "batch_size": 256,
        "epochs": 3000,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "hidden_layers": [64, 64, 64, 64],  # 4 layers
        "param_embedding_dim": 0,
        "dropout": 0.0,
        "n_basis": 0,
        "warmup": 0,
        "patience": 100,
        "grad_clip": 1.0,
    },
}

@dataclass
class TrainSettings:
    model: str
    target: str
    data_dir: str
    mat_files: Sequence[str]
    leak_index: int = 6
    rdc_indices: Sequence[int] = (4, 5, 2, 3)
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    activation: str = "relu"
    hidden_layers: Sequence[int] = [64, 64, 64, 64]
    param_embedding_dim: int = 64
    dropout: float = 0.0
    n_basis: int = 64
    warmup: int = 0
    patience: int = 0
    grad_clip: float = 0.0
    seed: int = 42
    device: Optional[str] = None
    out_dir: str = "valid_net"
    exp_name: Optional[str] = None
    baseline_alpha: float = 1.0
    init: str = "xavier_uniform"
    layernorm: bool = False
    head_names: Sequence[str] = ('K','k','C','c',)


class EarlyStopping:
    def __init__(self, patience: int, minimize: bool = True) -> None:
        self.patience = max(0, patience)
        self.minimize = minimize
        self.best = math.inf if minimize else -math.inf
        self.wait = 0
        self.state: Optional[Dict[str, Tensor]] = None

    def update(self, metric: float, model: nn.Module) -> bool:
        improved = (metric < self.best) if self.minimize else (metric > self.best)
        if improved:
            self.best = metric
            self.wait = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait > self.patience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal dataset training")
    parser.add_argument("--model", choices=["deeponet", "mlp", "baseline"], default="deeponet")
    parser.add_argument("--target", choices=["rdc", "leak"], default="rdc")
    parser.add_argument("--mat-files", nargs="*", default=list(DEFAULT_MAT_FILES))
    parser.add_argument("--data-dir", default="dataset/data/tapered_seal")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-layers", nargs="*", type=int)   # 리스트로 받음
    parser.add_argument("--param-embedding-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--n-basis", type=int)
    parser.add_argument("--leak-index", type=int)
    parser.add_argument("--rdc-indices", nargs="*", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--out-dir")
    parser.add_argument("--exp-name")
    parser.add_argument("--baseline-alpha", type=float)
    parser.add_argument("--head-names", default=('K','k','C','c',))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(spec: Optional[str]) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_rdc(data_dir: str, mat_files: Sequence[str], rdc_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    grid_norm: Optional[np.ndarray] = None
    grid_raw: Optional[np.ndarray] = None
    for name in mat_files:
        path = os.path.join(data_dir, name, "dataset.mat")
        with h5py.File(path, "r") as mat:
            inputs = np.array(mat["input"], dtype=np.float32)
            rdc = np.array(mat["RDC"], dtype=np.float32)
            w_vec = np.array(mat["params/wVec"], dtype=np.float32).squeeze()
        sel = rdc[list(rdc_indices), :, :]
        features.append(inputs.T)
        targets.append(sel.transpose(2, 0, 1))
        if grid_norm is None:
            w = w_vec
            w_norm = 2.0 * (w - w.min()) / (w.max() - w.min()) - 1.0
            grid_norm = w_norm[:, None].astype(np.float32)
            grid_raw = w[:, None].astype(np.float32)
    if not features:
        raise RuntimeError("No data loaded for RDC")
    X = np.concatenate(features, axis=0)
    Y = np.concatenate(targets, axis=0)
    return X, Y, grid_norm, grid_raw


def load_leak(data_dir: str, mat_files: Sequence[str], leak_index: int) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for name in mat_files:
        path = os.path.join(data_dir, name, "dataset.mat")
        with h5py.File(path, "r") as mat:
            inputs = np.array(mat["input"], dtype=np.float32)
            leak = np.array(mat["Leak"], dtype=np.float32)
        features.append(inputs.T)
        targets.append(leak[leak_index, :].reshape(-1, 1))
    if not features:
        raise RuntimeError("No data loaded for Leak")
    X = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    return X, y


def split_indices(n: int, seed: int, splits: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_end = int(n * splits[0])
    val_end = train_end + int(n * splits[1])
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def scale_inputs(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X[train_idx])
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, scaler


def scale_targets_rdc(Y: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, List[StandardScaler]]:
    n_heads = Y.shape[1]
    scalers: List[StandardScaler] = []
    Y_scaled = np.empty_like(Y, dtype=np.float32)
    for head in range(n_heads):
        scaler = StandardScaler().fit(Y[train_idx, head, :].reshape(-1, 1))
        transformed = scaler.transform(Y[:, head, :].reshape(-1, 1)).reshape(Y.shape[0], Y.shape[2])
        Y_scaled[:, head, :] = transformed.astype(np.float32)
        scalers.append(scaler)
    return Y_scaled, scalers


def scale_targets_leak(y: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(y[train_idx])
    y_scaled = scaler.transform(y).astype(np.float32)
    return y_scaled, scaler


def make_loaders(dataset: TensorDataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, batch_size: int) -> Dict[str, DataLoader]:
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())
    test_dataset = Subset(dataset, test_idx.tolist())
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int, warmup_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if warmup_steps <= 0 or steps_per_epoch == 0:
        return None
    total_steps = epochs * steps_per_epoch
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return scheduler


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], grid: Optional[Tensor], grad_clip: float) -> float:
    model.train()
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if grid is not None:
            batch_grid = grid.unsqueeze(0).expand(xb.size(0), -1, -1)
            preds = model(xb, batch_grid)
        else:
            preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(1, count)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, grid: Optional[Tensor]) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if grid is not None:
                batch_grid = grid.unsqueeze(0).expand(xb.size(0), -1, -1)
                preds = model(xb, batch_grid)
            else:
                preds = model(xb)
            loss = criterion(preds, yb)
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / max(1, count)


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, grid: Optional[Tensor]) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            if grid is not None:
                batch_grid = grid.unsqueeze(0).expand(xb.size(0), -1, -1)
                preds = model(xb, batch_grid)
            else:
                preds = model(xb)
            outputs.append(preds.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def inverse_scale_rdc(arr: np.ndarray, scalers: Sequence[StandardScaler]) -> np.ndarray:
    n_samples, n_heads, n_points = arr.shape
    restored = np.empty_like(arr)
    for head, scaler in enumerate(scalers):
        restored[:, head, :] = scaler.inverse_transform(arr[:, head, :].reshape(-1, 1)).reshape(n_samples, n_points)
    return restored


def inverse_scale_leak(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(arr)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_flat = y_true.reshape(y_true.shape[0], -1)
    pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    mse = mean_squared_error(true_flat, pred_flat)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(true_flat, pred_flat)
    r2 = r2_score(true_flat, pred_flat)
    rng = np.max(true_flat) - np.min(true_flat)
    rrmse = rmse / (rng + 1e-12)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "rrmse": float(rrmse)}


def compute_per_head_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    per_head: Dict[str, Dict[str, float]] = {}
    n_heads = y_true.shape[1]
    for head in range(n_heads):
        metrics = compute_metrics(y_true[:, head, :][:, :, None], y_pred[:, head, :][:, :, None])
        per_head[f"head_{head}"] = metrics
    return per_head


def serialize_scaler(scaler: StandardScaler) -> Dict[str, List[float]]:
    return {"mean": scaler.mean_.astype(float).tolist(), "scale": scaler.scale_.astype(float).tolist()}


def serialize_scalers_list(scalers: Sequence[StandardScaler]) -> Dict[str, List[List[float]]]:
    means = [sc.mean_.astype(float).reshape(-1).tolist() for sc in scalers]
    scales = [sc.scale_.astype(float).reshape(-1).tolist() for sc in scalers]
    return {"mean": means, "scale": scales}


# def run_baseline(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, scalers_y: Union[Sequence[StandardScaler], StandardScaler], target: str, alpha: float) -> Dict[str, Dict[str, Dict[str, float]]]:
#     X_train = features[train_idx]
#     X_val = features[val_idx]
#     X_test = features[test_idx]
#     y_train = targets[train_idx]
#     y_val = targets[val_idx]
#     y_test = targets[test_idx]

#     if target == "rdc":
#         reshaped = y_train.reshape(y_train.shape[0], -1)
#         model = MultiOutputRegressor(Ridge(alpha=alpha))
#         model.fit(X_train, reshaped)
#         pred_train = model.predict(X_train).reshape(y_train.shape)
#         pred_val = model.predict(X_val).reshape(y_val.shape)
#         pred_test = model.predict(X_test).reshape(y_test.shape)
#         true_train = inverse_scale_rdc(y_train, scalers_y)
#         true_val = inverse_scale_rdc(y_val, scalers_y)
#         true_test = inverse_scale_rdc(y_test, scalers_y)
#         pred_train_orig = inverse_scale_rdc(pred_train, scalers_y)
#         pred_val_orig = inverse_scale_rdc(pred_val, scalers_y)
#         pred_test_orig = inverse_scale_rdc(pred_test, scalers_y)
#         metrics = {
#             "train": compute_metrics(true_train, pred_train_orig),
#             "val": compute_metrics(true_val, pred_val_orig),
#             "test": compute_metrics(true_test, pred_test_orig),
#         }
#         metrics["per_head"] = {
#             "train": compute_per_head_metrics(true_train, pred_train_orig),
#             "val": compute_per_head_metrics(true_val, pred_val_orig),
#             "test": compute_per_head_metrics(true_test, pred_test_orig),
#         }
#     else:
#         ridge = Ridge(alpha=alpha)
#         ridge.fit(X_train, y_train)
#         pred_train = ridge.predict(X_train)
#         pred_val = ridge.predict(X_val)
#         pred_test = ridge.predict(X_test)
#         true_train = inverse_scale_leak(y_train, scalers_y)
#         true_val = inverse_scale_leak(y_val, scalers_y)
#         true_test = inverse_scale_leak(y_test, scalers_y)
#         pred_train_orig = inverse_scale_leak(pred_train, scalers_y)
#         pred_val_orig = inverse_scale_leak(pred_val, scalers_y)
#         pred_test_orig = inverse_scale_leak(pred_test, scalers_y)
#         metrics = {
#             "train": compute_metrics(true_train, pred_train_orig),
#             "val": compute_metrics(true_val, pred_val_orig),
#             "test": compute_metrics(true_test, pred_test_orig),
#         }
#     return metrics


def save_checkpoint(path: Path, model: nn.Module, settings: TrainSettings, scalers: Dict[str, Union[Dict[str, List[float]], Dict[str, List[List[float]]]]], splits: Dict[str, List[int]], history: Dict[str, List[float]], metrics: Dict[str, Dict[str, float]], grid_info: Optional[Dict[str, List[float]]]) -> None:
    payload = {
        "model": settings.model,
        "target": settings.target,
        "state_dict": model.state_dict(),
        "settings": asdict(settings),
        "scalers": scalers,
        "splits": splits,
        "history": history,
        "metrics": metrics,
        "grid": grid_info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_settings(args: argparse.Namespace) -> TrainSettings:
    base = DEFAULTS[args.model]
    settings = TrainSettings(
        model=args.model,
        target=args.target,
        data_dir=args.data_dir,
        mat_files=tuple(args.mat_files) if args.mat_files else DEFAULT_MAT_FILES,
        leak_index=args.leak_index if args.leak_index is not None else 6,
        rdc_indices=tuple(args.rdc_indices) if args.rdc_indices else (2, 3, 4, 5),
        batch_size=args.batch_size or base["batch_size"],
        epochs=args.epochs or base["epochs"],
        lr=args.lr or base["lr"],
        weight_decay=args.weight_decay if args.weight_decay is not None else base["weight_decay"],
        hidden_channels=args.hidden_channels or base["hidden_channels"],
        param_embedding_dim=args.param_embedding or base["param_embedding_dim"],
        n_layers=args.layers or base["n_layers"],
        dropout=args.dropout if args.dropout is not None else base["dropout"],
        n_basis=args.n_basis or base["n_basis"],
        warmup=args.warmup if args.warmup is not None else base["warmup"],
        patience=args.patience if args.patience is not None else base["patience"],
        grad_clip=args.grad_clip if args.grad_clip is not None else base["grad_clip"],
        seed=args.seed or 42,
        device=args.device,
        out_dir=args.out_dir or "net",
        exp_name=args.exp_name,
        baseline_alpha=args.baseline_alpha if args.baseline_alpha is not None else 1.0,
    )
    return settings

#TODO: 수정된 models.py에 맞춰서 model building, run training, evaluation, tuning 스크립트 수정해줘
def build_model(settings: TrainSettings) -> nn.Module:
    
    set_seed(settings.seed)
    device = get_device(settings.device)

    if settings.model == "mlp" and settings.target != "leak":
        raise ValueError("MLP training supports the leak target only")
    if settings.model == "deeponet" and settings.target != "rdc":
        raise ValueError("DeepONet training supports the rdc target only")

    model_type = settings.model.lower()
    
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

def run_training(settings: TrainSettings) -> Dict[str, Union[float, str, Dict[str, Dict[str, float]]]]:
    set_seed(settings.seed)
    device = get_device(settings.device)

    if settings.model == "mlp" and settings.target != "leak":
        raise ValueError("MLP training supports the leak target only")
    if settings.model == "deeponet" and settings.target != "rdc":
        raise ValueError("DeepONet training supports the rdc target only")
    
    
    

    if settings.target == "rdc":
        X, Y, grid_norm, grid_raw = load_rdc(settings.data_dir, settings.mat_files, settings.rdc_indices)
        train_idx, val_idx, test_idx = split_indices(X.shape[0], settings.seed)
        X_scaled, scaler_X = scale_inputs(X, train_idx)
        Y_scaled, scalers_y = scale_targets_rdc(Y, train_idx)
        tensor_X = torch.from_numpy(X_scaled)
        tensor_Y = torch.from_numpy(Y_scaled)
        dataset = TensorDataset(tensor_X, tensor_Y)
        loaders = make_loaders(dataset, train_idx, val_idx, test_idx, settings.batch_size)
        criterion = nn.MSELoss()
        model = MultiHeadDeepONet(
            n_params=X.shape[1],
            param_embedding_dim=settings.param_embedding_dim,
            hidden_channels=settings.hidden_channels,
            head_names=(),
            n_heads=Y.shape[1],
            n_layers=settings.n_layers,
            n_basis=settings.n_basis,
            p_drop=settings.dropout,
        ).to(device)
        grid_tensor = torch.from_numpy(grid_norm).to(device)
        grid_info = {
            "normalized": grid_norm.flatten().astype(float).tolist(),
            "original": grid_raw.flatten().astype(float).tolist(),
        }
        scalers_serialized = {
            "X": serialize_scaler(scaler_X),
            "Y": serialize_scalers_list(scalers_y),
        }
    else:
        X, y = load_leak(settings.data_dir, settings.mat_files, settings.leak_index)
        train_idx, val_idx, test_idx = split_indices(X.shape[0], settings.seed)
        X_scaled, scaler_X = scale_inputs(X, train_idx)
        y_scaled, scaler_y = scale_targets_leak(y, train_idx)
        tensor_X = torch.from_numpy(X_scaled)
        tensor_y = torch.from_numpy(y_scaled)
        dataset = TensorDataset(tensor_X, tensor_y)
        loaders = make_loaders(dataset, train_idx, val_idx, test_idx, settings.batch_size)
        criterion = nn.MSELoss()
        if settings.model == "mlp":
            model = MLP(
                in_dim=X.shape[1],
                out_dim=y.shape[1],
                hidden_channels=settings.hidden_channels,
                n_layers=settings.n_layers,
                p_drop=settings.dropout,
            ).to(device)
        else:
            model = MLP(
                in_dim=X.shape[1],
                out_dim=y.shape[1],
                hidden_channels=settings.hidden_channels,
                n_layers=settings.n_layers,
                p_drop=settings.dropout,
            ).to(device)
        grid_tensor = None
        grid_info = None
        scalers_serialized = {
            "X": serialize_scaler(scaler_X),
            "y": serialize_scaler(scaler_y),
        }

    if settings.model == "baseline":
        metrics = run_baseline(
            tensor_X.numpy(),
            tensor_Y.numpy() if settings.target == "rdc" else tensor_y.numpy(),
            train_idx,
            val_idx,
            test_idx,
            scalers_y if settings.target == "rdc" else scaler_y,
            settings.target,
            settings.baseline_alpha,
        )
        return {"checkpoint": "", "metrics": metrics, "best_val_loss": metrics["val"]["rmse"]}
    
    hyperparams = {
        "Batch size": settings.batch_size,
        "Parameter embedding dimension": settings.param_embedding_dim,
        "# of hidden channels": settings.hidden_channels,
        "# of layers": settings.n_layers,
        "# of shared output channels": settings.param_embedding_dim,
        "Learning rate": f"{settings.lr:.1e}",
        "p_drop": settings.dropout,
    }

    print(json.dumps(hyperparams, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    scheduler = build_scheduler(optimizer, settings.epochs, len(loaders["train"]), settings.warmup)
    early = EarlyStopping(settings.patience, minimize=True) if settings.patience > 0 else None

    history = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(settings.epochs):
        train_loss = train_epoch(model, loaders["train"], device, criterion, optimizer, scheduler, grid_tensor, settings.grad_clip)
        val_loss = eval_epoch(model, loaders["val"], device, criterion, grid_tensor)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{settings.epochs}, Train {train_loss:.6f}, Val {val_loss:.6f}')
        if val_loss < best_val:
            best_val = val_loss
        if early is not None:
            stop = early.update(val_loss, model)
            if stop:
                break

    if early and early.state is not None:
        model.load_state_dict(early.state)

    train_eval_loader = DataLoader(loaders["train"].dataset, batch_size=settings.batch_size, shuffle=False)
    val_eval_loader = DataLoader(loaders["val"].dataset, batch_size=settings.batch_size, shuffle=False)
    test_eval_loader = DataLoader(loaders["test"].dataset, batch_size=settings.batch_size, shuffle=False)

    train_preds = collect_predictions(model, train_eval_loader, device, grid_tensor)
    val_preds = collect_predictions(model, val_eval_loader, device, grid_tensor)
    test_preds = collect_predictions(model, test_eval_loader, device, grid_tensor)

    if settings.target == "rdc":
        y_train = tensor_Y[train_idx].numpy()
        y_val = tensor_Y[val_idx].numpy()
        y_test = tensor_Y[test_idx].numpy()
        train_true = inverse_scale_rdc(y_train, scalers_y)
        val_true = inverse_scale_rdc(y_val, scalers_y)
        test_true = inverse_scale_rdc(y_test, scalers_y)
        train_pred = inverse_scale_rdc(train_preds, scalers_y)
        val_pred = inverse_scale_rdc(val_preds, scalers_y)
        test_pred = inverse_scale_rdc(test_preds, scalers_y)
        per_head = {
            "train": compute_per_head_metrics(train_true, train_pred),
            "val": compute_per_head_metrics(val_true, val_pred),
            "test": compute_per_head_metrics(test_true, test_pred),
        }
    else:
        y_train = tensor_y[train_idx].numpy()
        y_val = tensor_y[val_idx].numpy()
        y_test = tensor_y[test_idx].numpy()
        train_true = inverse_scale_leak(y_train, scaler_y)
        val_true = inverse_scale_leak(y_val, scaler_y)
        test_true = inverse_scale_leak(y_test, scaler_y)
        train_pred = inverse_scale_leak(train_preds, scaler_y)
        val_pred = inverse_scale_leak(val_preds, scaler_y)
        test_pred = inverse_scale_leak(test_preds, scaler_y)
        per_head = {}

    metrics = {
        "train": compute_metrics(train_true, train_pred),
        "val": compute_metrics(val_true, val_pred),
        "test": compute_metrics(test_true, test_pred),
    }
    if per_head:
        metrics["per_head"] = per_head

    out_dir = Path(settings.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = settings.exp_name or f"{settings.model}_{settings.target}_{settings.mat_files[0]}"
    ckpt_path = out_dir / f"{exp_name}.pth"

    scalers_info = scalers_serialized
    splits = {
        "train": train_idx.astype(int).tolist(),
        "val": val_idx.astype(int).tolist(),
        "test": test_idx.astype(int).tolist(),
    }
    save_checkpoint(ckpt_path, model.cpu(), settings, scalers_info, splits, history, metrics, grid_info)

    return {
        "checkpoint": str(ckpt_path),
        "metrics": metrics,
        "best_val_loss": best_val,
    }


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    result = run_training(settings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
