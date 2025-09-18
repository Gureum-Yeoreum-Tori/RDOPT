from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .utils import save_pickle


@dataclass
class ScalerBundle:
    feature: Optional[Any]
    target: Optional[Any]


@dataclass
class DatasetBundle:
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]
    scalers: ScalerBundle


class SealDataset(Dataset):
    """Torch dataset for seal regression tasks."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        ids: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        w_grid: Optional[np.ndarray] = None,
    ) -> None:
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.ids = ids
        self.weights = weights.astype(np.float32) if weights is not None else None
        self.w_grid = w_grid.astype(np.float32) if w_grid is not None else None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = {
            "x": torch.from_numpy(self.features[idx]),
            "y": torch.from_numpy(self.targets[idx]),
        }
        if self.ids is not None:
            sample["id"] = int(self.ids[idx])
        if self.weights is not None:
            sample["weight"] = torch.tensor(self.weights[idx])
        if self.w_grid is not None:
            sample["w"] = torch.from_numpy(self.w_grid[idx])
        return sample


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate handling optional metadata."""
    features = torch.stack([item["x"] for item in batch])
    targets = torch.stack([item["y"] for item in batch])
    result: Dict[str, Any] = {"x": features, "y": targets}
    if "w" in batch[0]:
        w_shapes = {tuple(item["w"].shape) for item in batch}
        if len(w_shapes) != 1:
            raise ValueError("Inconsistent w_grid shapes in batch")
        result["w"] = torch.stack([item["w"] for item in batch])
    if "weight" in batch[0]:
        result["weight"] = torch.stack([item["weight"] for item in batch])
    if "id" in batch[0]:
        result["id"] = [item["id"] for item in batch]
    return result


def load_table(data_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Load tabular data from csv, h5, or mat."""
    if data_cfg.get("path_csv"):
        return pd.read_csv(data_cfg["path_csv"])
    if data_cfg.get("path_h5"):
        table = data_cfg.get("table")
        if table:
            return pd.read_hdf(data_cfg["path_h5"], key=table)
        with h5py.File(data_cfg["path_h5"], "r") as handle:
            return _h5_to_frame(handle)
    if data_cfg.get("path_mat"):
        mat = loadmat(data_cfg["path_mat"])
        return _dict_to_frame(mat)
    raise ValueError("No supported data path provided")


def _h5_to_frame(handle: h5py.File) -> pd.DataFrame:
    payload: Dict[str, Any] = {}
    for key, value in handle.items():
        if isinstance(value, h5py.Dataset):
            payload[key] = np.array(value)
    return _dict_to_frame(payload)


def _dict_to_frame(data: Mapping[str, Any]) -> pd.DataFrame:
    cleaned = {
        key: np.array(value)
        for key, value in data.items()
        if not key.startswith("__")
    }
    columns: Dict[str, np.ndarray] = {}
    sample_count: Optional[int] = None
    for key, value in cleaned.items():
        arr = np.array(value)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            columns[key] = arr
            sample_count = arr.shape[0]
            continue
        if arr.ndim == 2:
            if sample_count is None:
                sample_count = arr.shape[0]
            if arr.shape[0] == sample_count:
                columns[key] = arr
            elif arr.shape[1] == sample_count:
                columns[key] = arr.T
            else:
                columns[key] = arr
        else:
            columns[key] = arr
    frame = pd.DataFrame()
    for key, value in columns.items():
        if value.ndim == 1:
            frame[key] = value
        else:
            for idx in range(value.shape[1]):
                frame[f"{key}_{idx}"] = value[:, idx]
    return frame


def prepare_arrays(
    frame: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    id_column: Optional[str],
    weight_column: Optional[str],
    w_column: Optional[str],
    expand_w: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract numpy arrays from frame with optional expansions."""
    x = frame.loc[:, list(features)].to_numpy(dtype=np.float32)
    y = frame.loc[:, list(targets)].to_numpy(dtype=np.float32)
    ids = frame[id_column].to_numpy() if id_column and id_column in frame else None
    weights = (
        frame[weight_column].to_numpy(dtype=np.float32)
        if weight_column and weight_column in frame
        else None
    )
    w_grid: Optional[np.ndarray] = None
    if w_column and w_column in frame:
        raw = frame[w_column]
        parsed_values: List[np.ndarray] = []
        stackable = True
        for value in raw:
            if isinstance(value, (list, np.ndarray)):
                parsed = np.array(value, dtype=np.float32)
            elif isinstance(value, str):
                try:
                    parsed = np.array(json.loads(value), dtype=np.float32)
                except json.JSONDecodeError:
                    parsed = np.array(ast.literal_eval(value), dtype=np.float32)
            else:
                parsed = np.array([value], dtype=np.float32)
            if parsed.ndim == 0:
                parsed = parsed.reshape(1)
            parsed_values.append(parsed)
        first_shape = parsed_values[0].shape
        for item in parsed_values:
            if item.shape != first_shape:
                stackable = False
                break
        if stackable and len(first_shape) >= 1:
            w_grid = np.stack(parsed_values)
        else:
            scalars = [float(np.squeeze(item)) for item in parsed_values]
            w_grid = np.asarray(scalars, dtype=np.float32)[:, None]
        if w_grid.ndim == 2:
            w_grid = w_grid[..., None]
        if expand_w:
            x, y, ids, weights, w_grid = _expand_by_w_grid(x, y, ids, weights, w_grid)
    return x, y, ids, weights, w_grid


def _expand_by_w_grid(
    x: np.ndarray,
    y: np.ndarray,
    ids: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    w: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if w is None:
        return x, y, ids, weights, w
    if w.ndim == 3:
        samples, grid, w_dim = w.shape
        x_expanded = np.repeat(x, grid, axis=0)
        if y.ndim == 3:
            y_expanded = y.reshape(samples * grid, -1)
        else:
            y_expanded = np.repeat(y, grid, axis=0)
        w_expanded = w.reshape(samples * grid, w_dim)
        ids_expanded = np.repeat(ids, grid, axis=0) if ids is not None else None
        weights_expanded = np.repeat(weights, grid, axis=0) if weights is not None else None
        return x_expanded, y_expanded, ids_expanded, weights_expanded, w_expanded
    return x, y, ids, weights, w


def build_scalers(variant: str) -> ScalerBundle:
    """Instantiate scalers from config flag."""
    variant = (variant or "none").lower()
    if variant == "standard":
        return ScalerBundle(StandardScaler(), None)
    if variant == "minmax":
        return ScalerBundle(MinMaxScaler(), None)
    return ScalerBundle(None, None)


def split_datasets(
    x: np.ndarray,
    y: np.ndarray,
    ids: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    w_grid: Optional[np.ndarray],
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[SealDataset, Optional[SealDataset], Optional[SealDataset]]:
    indices = np.arange(x.shape[0])
    test_ratio = float(test_size)
    val_ratio = float(val_size)
    if test_ratio > 0:
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
        )
    else:
        idx_train, idx_test = indices, np.array([], dtype=int)
    if val_ratio > 0:
        idx_train, idx_val = train_test_split(
            idx_train,
            test_size=val_ratio / max(1.0 - test_ratio, 1e-6),
            random_state=seed,
            shuffle=True,
        )
    else:
        idx_val = np.array([], dtype=int)

    train = SealDataset(
        features=x[idx_train],
        targets=y[idx_train],
        ids=None if ids is None else ids[idx_train],
        weights=None if weights is None else weights[idx_train],
        w_grid=None if w_grid is None else w_grid[idx_train],
    )
    val = None
    if idx_val.size > 0:
        val = SealDataset(
            features=x[idx_val],
            targets=y[idx_val],
            ids=None if ids is None else ids[idx_val],
            weights=None if weights is None else weights[idx_val],
            w_grid=None if w_grid is None else w_grid[idx_val],
        )
    test = None
    if idx_test.size > 0:
        test = SealDataset(
            features=x[idx_test],
            targets=y[idx_test],
            ids=None if ids is None else ids[idx_test],
            weights=None if weights is None else weights[idx_test],
            w_grid=None if w_grid is None else w_grid[idx_test],
        )
    return train, val, test


def apply_scaler(bundle: ScalerBundle, train: SealDataset, others: Iterable[Optional[SealDataset]]) -> None:
    """Fit scaler on train and transform datasets in-place."""
    if bundle.feature is None:
        return
    scaler = bundle.feature
    scaler.fit(train.features)
    train.features = scaler.transform(train.features).astype(np.float32)
    for subset in others:
        if subset is None:
            continue
        subset.features = scaler.transform(subset.features).astype(np.float32)


def save_scalers(bundle: ScalerBundle, path: Path) -> None:
    payload = {
        "feature": bundle.feature,
        "target": bundle.target,
    }
    save_pickle(path, payload)


def create_dataloaders(
    train: SealDataset,
    val: Optional[SealDataset],
    test: Optional[SealDataset],
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    loaders = {
        "train": DataLoader(
            train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    }
    if val is not None:
        loaders["val"] = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    if test is not None:
        loaders["test"] = DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    return loaders


def create_kfold_indices(n_samples: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(np.arange(n_samples))]


def prepare_datasets(
    data_cfg: Mapping[str, Any],
    exp_dir: Path,
    seed: int,
) -> Tuple[SealDataset, Optional[SealDataset], Optional[SealDataset], ScalerBundle, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    frame = load_table(data_cfg)
    x, y, ids, weights, w_grid = prepare_arrays(
        frame,
        data_cfg.get("features", []),
        data_cfg.get("targets", []),
        data_cfg.get("id_column"),
        data_cfg.get("weight_column"),
        data_cfg.get("w_column"),
        expand_w=bool(data_cfg.get("expand_w", False)),
    )
    scalers = build_scalers(data_cfg.get("scaler", "none"))
    if data_cfg.get("split", True):
        train, val, test = split_datasets(
            x,
            y,
            ids,
            weights,
            w_grid,
            val_size=float(data_cfg.get("val_size", 0.1)),
            test_size=float(data_cfg.get("test_size", 0.1)),
            seed=seed,
        )
        apply_scaler(scalers, train, [val, test])
        scaler_path = exp_dir / "scaler.pkl"
        save_scalers(scalers, scaler_path)
        return train, val, test, scalers, x, y, ids, weights, w_grid
    dataset = SealDataset(
        features=x,
        targets=y,
        ids=ids,
        weights=weights,
        w_grid=w_grid,
    )
    return dataset, None, None, scalers, x, y, ids, weights, w_grid
