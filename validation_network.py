#%%
import json
from typing import Optional
from model_validation.train import DEFAULT_MAT_FILES, TrainSettings, run_training

BASE_TRAIN_SETTINGS = TrainSettings(
    model="deeponet",
    target="rdc",
    data_dir="dataset/data/tapered_seal",
    mat_files=DEFAULT_MAT_FILES,
    leak_index=6,
    rdc_indices=(2, 3, 4, 5),
    batch_size=512,
    epochs=5000,
    lr=1e-4,
    weight_decay=1e-6,
    hidden_channels=64,
    param_embedding_dim=64,
    n_layers=8,
    dropout=0.0,
    n_basis=64,
    warmup=500,
    patience=0,
    grad_clip=0.0,
    seed=42,
    device=None,
    out_dir="net",
    exp_name="deeponet_rdc_manual",
    baseline_alpha=1.0,
)


def run(settings: Optional[TrainSettings] = None) -> dict:
    active = settings or BASE_TRAIN_SETTINGS
    return run_training(active)

result = run()
print(json.dumps(result, indent=2))

#%%
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from model_validation.train import DEFAULTS, DEFAULT_MAT_FILES, TrainSettings, run_training


BASE_TUNE_SETTINGS = TrainSettings(
    model="deeponet",
    target="rdc",
    data_dir="dataset/data/tapered_seal",
    mat_files=DEFAULT_MAT_FILES,
    leak_index=6,
    rdc_indices=(2, 3, 4, 5),
    batch_size=DEFAULTS["deeponet"]["batch_size"],
    epochs=DEFAULTS["deeponet"]["epochs"],
    lr=DEFAULTS["deeponet"]["lr"],
    weight_decay=DEFAULTS["deeponet"]["weight_decay"],
    hidden_channels=DEFAULTS["deeponet"]["hidden_channels"],
    param_embedding_dim=DEFAULTS["deeponet"]["param_embedding_dim"],
    n_layers=DEFAULTS["deeponet"]["n_layers"],
    dropout=DEFAULTS["deeponet"]["dropout"],
    n_basis=DEFAULTS["deeponet"]["n_basis"],
    warmup=DEFAULTS["deeponet"]["warmup"],
    patience=DEFAULTS["deeponet"]["patience"],
    grad_clip=DEFAULTS["deeponet"]["grad_clip"],
    seed=42,
    device=None,
    out_dir="net",
    exp_name=None,
    baseline_alpha=1.0,
)

PARAM_GRID: Dict[str, Iterable] = {
    "hidden_channels": [64, 128],
    "param_embedding_dim": [64, 128],
    "n_layers": [6, 8],
    "n_basis": [64, 96],
    "dropout": [0.0, 0.1],
    "lr": [1e-4, 5e-5],
    "weight_decay": [1e-6, 5e-6],
    "batch_size": [256, 512],
    "warmup": [250, 500],
    "patience": [80, 120],
    "grad_clip": [0.0, 1.0],
}

EXP_PREFIX = "deeponet_rdc_grid"
MAX_TRIALS: Optional[int] = None
JSON_LOG: Optional[Path] = None


def cast_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_grid(
    settings: TrainSettings,
    grid: Dict[str, Iterable],
    max_trials: Optional[int] = None,
    exp_prefix: Optional[str] = None,
    json_log: Optional[Path] = None,
) -> Tuple[List[Dict], Dict]:
    if settings.model == "deeponet" and settings.target != "rdc":
        raise ValueError("DeepONet tuning requires the rdc target")
    if settings.model == "mlp" and settings.target != "leak":
        raise ValueError("MLP tuning requires the leak target")
    order = list(grid.keys())
    results: List[Dict] = []
    prefix = exp_prefix or f"{settings.model}_{settings.target}"
    for trial_idx, values in enumerate(product(*(grid[key] for key in order)), start=1):
        trial_params = dict(zip(order, values))
        trial_settings = replace(settings, **trial_params)
        trial_settings.exp_name = f"{prefix}_trial{trial_idx:03d}"
        outcome = run_training(trial_settings)
        val_rmse = outcome["metrics"]["val"]["rmse"]
        record = {
            "trial": trial_idx,
            "params": {k: cast_value(v) for k, v in trial_params.items()},
            "val_rmse": float(val_rmse),
            "metrics": outcome["metrics"],
            "checkpoint": outcome["checkpoint"],
        }
        results.append(record)
        print(json.dumps(record))
        if max_trials and trial_idx >= max_trials:
            break
    ranked = sorted(results, key=lambda item: item["val_rmse"]) if results else []
    summary = {"results": results, "ranked": ranked}
    print(json.dumps(summary, indent=2))
    if json_log:
        json_log.write_text(json.dumps(summary, indent=2))
    return results, summary


run_grid(
    settings=BASE_TUNE_SETTINGS,
    grid=PARAM_GRID,
    max_trials=MAX_TRIALS,
    exp_prefix=EXP_PREFIX,
    json_log=JSON_LOG,
)
