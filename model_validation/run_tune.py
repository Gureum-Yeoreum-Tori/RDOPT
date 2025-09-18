from pathlib import Path
from typing import Dict, Iterable, Optional

from model_validation.train import DEFAULTS, DEFAULT_MAT_FILES, TrainSettings
from model_validation.tune import make_pruner, make_sampler, run_optuna


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

PARAM_SPACE: Dict[str, Iterable] = {
    "hidden_channels": [64, 96, 128],
    "param_embedding_dim": [64, 96, 128],
    "n_layers": [6, 7, 8],
    "n_basis": [64, 96],
    "dropout": [0.0, 0.05, 0.1],
    "lr": [5e-5, 1e-4, 2e-4],
    "weight_decay": [1e-6, 5e-6, 1e-5],
    "batch_size": [256, 384, 512],
    "warmup": [250, 500, 750],
    "patience": [80, 100, 120],
    "grad_clip": [0.0, 0.5, 1.0],
}

SAMPLER_NAME = "tpe"
PRUNER_NAME = "none"
N_TRIALS: Optional[int] = 30
TIMEOUT: Optional[float] = None
EXP_PREFIX = "deeponet_rdc_optuna"
JSON_LOG: Optional[Path] = None
STUDY_NAME: Optional[str] = None
STORAGE: Optional[str] = None


if __name__ == "__main__":
    sampler = make_sampler(SAMPLER_NAME, BASE_TUNE_SETTINGS.seed)
    pruner = make_pruner(PRUNER_NAME)
    run_optuna(
        base_settings=BASE_TUNE_SETTINGS,
        search_space=PARAM_SPACE,
        sampler=sampler,
        pruner=pruner,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        study_name=STUDY_NAME,
        storage=STORAGE,
        exp_prefix=EXP_PREFIX,
        json_log=str(JSON_LOG) if JSON_LOG else None,
    )
