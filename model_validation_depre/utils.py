from __future__ import annotations

import ast
import json
import logging
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import yaml
from scipy.interpolate import LinearNDInterpolator, griddata
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

METRIC_EPS = 1e-8


@dataclass
class ExperimentPaths:
    root: Path
    checkpoints: Path
    logs: Path
    figures: Path
    predictions: Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_experiment(save_dir: Path, exp_name: str) -> ExperimentPaths:
    """Create standard experiment folders."""
    root = ensure_dir(save_dir / exp_name)
    return ExperimentPaths(
        root=root,
        checkpoints=ensure_dir(root / "checkpoints"),
        logs=ensure_dir(root / "logs"),
        figures=ensure_dir(root / "figures"),
        predictions=ensure_dir(root / "predictions"),
    )


def set_seed(seed: int) -> None:
    """Seed all major RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve compute device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_logging(log_dir: Path, exp_name: str) -> logging.Logger:
    """Set up file and console logging."""
    ensure_dir(log_dir)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def parse_override(value: str) -> Any:
    """Parse CLI override values using YAML/AST fallback."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def apply_overrides(config: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    """Apply dotted key overrides into nested dictionaries."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}")
        key, value = override.split("=", 1)
        parsed_value = parse_override(value)
        keys = key.split(".")
        cursor: MutableMapping[str, Any] = config
        for part in keys[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                cursor[part] = {}
            cursor = cursor[part]  # type: ignore[assignment]
        cursor[keys[-1]] = parsed_value


def load_config(path: Optional[str], overrides: Sequence[str]) -> Dict[str, Any]:
    """Load YAML config and apply overrides."""
    config: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    apply_overrides(config, overrides)
    return config


def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    """Store mapping as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def save_pickle(path: Path, obj: Any) -> None:
    """Persist Python object via pickle."""
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: Path) -> Any:
    """Load pickle file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error with epsilon guard."""
    denom = np.clip(np.abs(y_true), METRIC_EPS, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


METRIC_FUNCS = {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "r2": r2,
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: Sequence[str],
    target_names: Sequence[str],
) -> Dict[str, float]:
    """Compute metrics overall and per target."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    target_count = y_true.shape[1]
    metrics: Dict[str, float] = {}
    for name in metric_names:
        func = METRIC_FUNCS[name.lower()]
        per_target: List[float] = []
        for idx in range(target_count):
            score = func(y_true[:, idx], y_pred[:, idx])
            suffix = target_names[idx] if idx < len(target_names) else f"target_{idx}"
            metrics[f"{name.lower()}_{suffix}"] = score
            per_target.append(score)
        metrics[name.lower()] = float(np.mean(per_target))
    return metrics


def is_higher_better(metric_name: str) -> bool:
    """Identify optimization direction."""
    return metric_name.lower() in {"r2"}


def best_metric(initial: Optional[float], candidate: float, metric_name: str) -> bool:
    """Return True if candidate improves the metric."""
    if initial is None:
        return True
    if is_higher_better(metric_name):
        return candidate > initial
    return candidate < initial


def interpolation_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Apply interpolation baseline depending on dimensionality."""
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        preds = [
            interpolation_baseline(x_train, y_train[:, idx], x_eval, method)
            for idx in range(y_train.shape[1])
        ]
        return np.stack(preds, axis=1)
    dims = x_train.shape[1]
    y_flat = np.squeeze(y_train)
    if dims <= 2:
        interpolator = LinearNDInterpolator(x_train, y_flat)
        preds = interpolator(x_eval)
        if np.isnan(preds).any():
            preds = griddata(x_train, y_flat, x_eval, method=method)
        return np.asarray(preds)
    knn = KNeighborsRegressor(n_neighbors=min(5, len(x_train)))
    knn.fit(x_train, y_flat)
    preds_knn = knn.predict(x_eval)
    return np.asarray(preds_knn)


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: Mapping[str, Any],
    target_names: Sequence[str],
    metric_names: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate classical baselines."""
    results: Dict[str, Dict[str, float]] = {}
    if not config.get("run", False):
        return results

    baselines: Dict[str, Any] = {}
    if config.get("linear", True):
        baselines["linear"] = LinearRegression()
    if config.get("ridge_alpha") is not None:
        baselines["ridge"] = Ridge(alpha=float(config.get("ridge_alpha", 1.0)))
    if config.get("random_forest", True):
        baselines["random_forest"] = RandomForestRegressor(
            n_estimators=int(config.get("rf_estimators", 200)),
            random_state=int(config.get("seed", 0)),
            n_jobs=-1,
        )
    if config.get("knn", True):
        baselines["knn"] = KNeighborsRegressor(n_neighbors=int(config.get("knn_k", 5)))
    if config.get("xgboost", False) and xgb is not None:
        baselines["xgboost"] = xgb.XGBRegressor(
            n_estimators=int(config.get("xgb_estimators", 500)),
            learning_rate=float(config.get("xgb_lr", 0.05)),
            max_depth=int(config.get("xgb_max_depth", 6)),
            subsample=float(config.get("xgb_subsample", 0.8)),
            colsample_bytree=float(config.get("xgb_colsample", 0.8)),
            objective="reg:squarederror",
            n_jobs=1,
            random_state=int(config.get("seed", 0)),
        )

    y_train_flat = y_train if y_train.ndim == 1 else y_train
    y_test_flat = y_test if y_test.ndim == 1 else y_test

    for name, model in baselines.items():
        fitted = _fit_baseline(model, x_train, y_train_flat)
        preds = fitted.predict(x_test)
        metrics = compute_metrics(
            y_true=y_test_flat,
            y_pred=preds,
            metric_names=metric_names,
            target_names=target_names,
        )
        results[name] = metrics
        if logger:
            logger.info("Baseline %s metrics: %s", name, json.dumps(metrics, indent=2))

    if config.get("interpolation", True):
        preds = interpolation_baseline(x_train, y_train_flat, x_test)
        metrics = compute_metrics(
            y_test_flat,
            preds,
            metric_names,
            target_names,
        )
        results["interpolation"] = metrics
        if logger:
            logger.info("Baseline interpolation metrics: %s", json.dumps(metrics, indent=2))

    return results


def _fit_baseline(model: Any, x: np.ndarray, y: np.ndarray) -> Any:
    """Fit baseline model target-wise if needed."""
    if y.ndim == 1 or y.shape[1] == 1:
        model.fit(x, np.squeeze(y))
        return model
    models = []
    for idx in range(y.shape[1]):
        clone_model = clone(model)
        clone_model.fit(x, y[:, idx])
        models.append(clone_model)
    return _MultiOutputWrapper(models)


class _MultiOutputWrapper:
    """Wrapper to mimic predict for per-target estimators."""

    def __init__(self, models: Sequence[Any]):
        self.models = models

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = [model.predict(x) for model in self.models]
        return np.stack(preds, axis=1)


def save_checkpoint(
    state: Mapping[str, Any],
    path: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Persist training state."""
    torch.save(state, path)
    if logger:
        logger.info("Checkpoint saved to %s", path)
