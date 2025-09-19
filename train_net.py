"""Optuna-based hyperparameter search for MLP and DeepONet models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import optuna
from optuna import Trial
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler

from model_validation.train import TrainSettings, run_training

# Dataset configuration
DATA_DIR = "dataset/data/tapered_seal"
MAT_FILES = (
    # "20250908_T_182846",
    # "20250911_T_091324",
    # "20250908_T_183632",
    "20250908_T_203220",
)
TARGET = "rdc"

# Optuna configuration
OPTUNA_SEED = 42
OUTPUT_DIR = Path("net") / "optuna"
SUMMARY_PATH = OUTPUT_DIR / "best_trials.json"
N_TRIALS: Dict[str, int] = {"mlp": 20, "deeponet": 30}

# Search spaces
MLP_WIDTH_CHOICES = [32, 64, 96, 128, 192, 256, 320, 384, 512]
DEEPONET_BRANCH_CHOICES = [32, 64, 96, 128, 160, 192, 224, 256, 320]
DEEPONET_TRUNK_CHOICES = [32, 48, 64, 80, 96, 112, 128, 160]


def create_base_settings(model_type: str) -> TrainSettings:
    """Return base training settings for the requested model."""
    common_kwargs = dict(
        target=TARGET,
        data_dir=DATA_DIR,
        mat_files=MAT_FILES,
        leak_index=6,
        rdc_indices=(2, 3, 4, 5),
        seed=OPTUNA_SEED,
        device=None,
        head_names=None,
    )

    if model_type == "mlp":
        return TrainSettings(
            model="mlp",
            batch_size=256,
            epochs=600,
            lr=1e-4,
            weight_decay=1e-6,
            activation="relu",
            hidden_layers=[128, 128, 128],
            dropout=0.0,
            warmup=0,
            patience=150,
            grad_clip=1.0,
            layernorm=False,
            out_dir=str(OUTPUT_DIR / "mlp"),
            **common_kwargs,
        )
    if model_type == "deeponet":
        return TrainSettings(
            model="deeponet",
            batch_size=512,
            epochs=1500,
            lr=1e-4,
            weight_decay=1e-6,
            activation="gelu",
            hidden_layers=[64, 64, 64, 64],
            branch_layers=[128, 128, 128],
            trunk_layers=[64, 64],
            param_embedding_dim=64,
            dropout=0.0,
            n_basis=64,
            warmup=500,
            patience=200,
            grad_clip=0.5,
            layernorm=False,
            out_dir=str(OUTPUT_DIR / "deeponet"),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def suggest_layer_stack(trial: Trial, prefix: str, min_layers: int, max_layers: int, choices: List[int]) -> List[int]:
    """Sample a variable-depth stack of layer widths."""
    n_layers = trial.suggest_int(f"{prefix}_n_layers", min_layers, max_layers)
    return [trial.suggest_categorical(f"{prefix}_width_{idx}", choices) for idx in range(n_layers)]


def build_mlp_settings(trial: Trial) -> TrainSettings:
    settings = create_base_settings("mlp")
    settings.hidden_layers = suggest_layer_stack(trial, "mlp_hidden", 2, 6, MLP_WIDTH_CHOICES)
    # settings.dropout = trial.suggest_float("mlp_dropout", 0.0, 0.3, step=0.05)
    # settings.activation = trial.suggest_categorical("mlp_activation", ["relu", "gelu", "tanh"])
    settings.activation = trial.suggest_categorical("mlp_activation", ["relu", "gelu"])
    settings.layernorm = trial.suggest_categorical("mlp_layernorm", [False, True])
    settings.lr = trial.suggest_float("mlp_lr", 1e-5, 5e-3, log=True)
    settings.weight_decay = trial.suggest_float("mlp_weight_decay", 1e-6, 1e-3, log=True)
    settings.batch_size = trial.suggest_categorical("mlp_batch_size", [256, 512, 1024])
    # settings.grad_clip = trial.suggest_categorical("mlp_grad_clip", [0.0, 0.5, 1.0, 2.0])
    # settings.patience = trial.suggest_int("mlp_patience", 100, 400, step=50)
    settings.epochs = trial.suggest_int("mlp_epochs", 1000, 3000, step=500)
    settings.exp_name = f"mlp_optuna_trial{trial.number:03d}"
    settings.out_dir = str(OUTPUT_DIR / "mlp")
    return settings


def build_deeponet_settings(trial: Trial) -> TrainSettings:
    settings = create_base_settings("deeponet")
    settings.branch_layers = suggest_layer_stack(trial, "deeponet_branch", 2, 5, DEEPONET_BRANCH_CHOICES)
    settings.trunk_layers = suggest_layer_stack(trial, "deeponet_trunk", 2, 4, DEEPONET_TRUNK_CHOICES)
    settings.param_embedding_dim = trial.suggest_categorical("deeponet_param_dim", [16, 32, 48, 64, 96, 128, 160])
    settings.n_basis = trial.suggest_categorical("deeponet_latent_dim", [16, 32, 48, 64, 80, 96, 128])
    # settings.dropout = trial.suggest_float("deeponet_dropout", 0.0, 0.2, step=0.05)
    settings.activation = trial.suggest_categorical("deeponet_activation", ["relu", "gelu"])
    settings.lr = trial.suggest_float("deeponet_lr", 1e-6, 1e-4, log=True)
    settings.weight_decay = trial.suggest_float("deeponet_weight_decay", 1e-6, 1e-4, log=True)
    settings.batch_size = trial.suggest_categorical("deeponet_batch_size", [256, 512, 1024])
    # settings.grad_clip = trial.suggest_categorical("deeponet_grad_clip", [0.0, 0.5, 1.0])
    # settings.warmup = trial.suggest_categorical("deeponet_warmup", [0, 200, 400, 600, 800])
    # settings.patience = trial.suggest_int("deeponet_patience", 150, 400, step=50)
    settings.epochs = trial.suggest_int("deeponet_epochs", 1000, 5000, step=500)
    settings.exp_name = f"deeponet_optuna_trial{trial.number:03d}"
    settings.out_dir = str(OUTPUT_DIR / "deeponet")
    return settings


def evaluate_trial(trial: Trial, settings: TrainSettings) -> float:
    try:
        result = run_training(settings)
    except Exception as err:
        trial.set_user_attr("error", str(err))
        raise TrialPruned(f"Training failed: {err}") from err

    metrics = result["metrics"]
    val_rmse = float(metrics["val"]["rmse"])
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("checkpoint", result["checkpoint"])
    trial.set_user_attr("best_val_loss", float(result["best_val_loss"]))
    trial.report(val_rmse, step=settings.epochs)
    print(f"[{settings.model.upper()}][trial {trial.number}] val RMSE={val_rmse:.6f}")
    return val_rmse


def make_objective(model_type: str):
    def objective(trial: Trial) -> float:
        if model_type == "mlp":
            settings = build_mlp_settings(trial)
        else:
            settings = build_deeponet_settings(trial)
        return evaluate_trial(trial, settings)

    return objective


def create_study(model_type: str) -> optuna.Study:
    sampler = TPESampler(seed=OPTUNA_SEED)
    return optuna.create_study(direction="minimize", study_name=f"{model_type}_optuna", sampler=sampler)


def save_summary(studies: Dict[str, optuna.Study]) -> None:
    if not studies:
        return
    summary = {}
    for model_type, study in studies.items():
        if not study.trials:
            continue
        best = study.best_trial
        summary[model_type] = {
            "best_value": float(best.value),
            "best_params": best.params,
            "user_attrs": best.user_attrs,
        }
    if not summary:
        return
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    studies: Dict[str, optuna.Study] = {}
    for model_type in ("mlp", "deeponet"):
        n_trials = N_TRIALS.get(model_type, 0)
        if n_trials <= 0:
            continue
        print(f"Starting Optuna study for {model_type} ({n_trials} trials)")
        study = create_study(model_type)
        objective = make_objective(model_type)
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial
        print(f"Best {model_type} RMSE: {best.value:.6f}")
        print(json.dumps(best.params, indent=2))
        studies[model_type] = study

    save_summary(studies)


if __name__ == "__main__":
    main()
