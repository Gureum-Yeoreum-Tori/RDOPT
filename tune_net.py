#%%
"""Optuna-based hyperparameter search for MLP and DeepONet models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import optuna
from optuna import Trial
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from model_validation.train import TrainSettings, run_training

import shutil
from datetime import datetime

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
# N_TRIALS: Dict[str, int] = {"mlp": 20, "deeponet": 30}
N_TRIALS: Dict[str, int] = {"deeponet": 30}

# Search spaces
MLP_WIDTH_CHOICES = [32, 64, 96, 128, 192, 256]
DEEPONET_BRANCH_CHOICES = [64, 96, 128, 160, 192, 224, 256]
DEEPONET_TRUNK_CHOICES = [32, 48, 64, 80, 96, 112, 128]


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
            epochs=3000,
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
            epochs=5000,
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
    settings.param_embedding_dim = trial.suggest_categorical("deeponet_param_dim", [16, 32, 48, 64, 96, 128])
    settings.n_basis = trial.suggest_categorical("deeponet_latent_dim", [16, 32, 48, 64, 96, 128])
    # settings.dropout = trial.suggest_float("deeponet_dropout", 0.0, 0.2, step=0.05)
    settings.activation = trial.suggest_categorical("deeponet_activation", ["relu", "gelu"])
    settings.lr = trial.suggest_float("deeponet_lr", 1e-6, 1e-4, log=True)
    settings.weight_decay = trial.suggest_float("deeponet_weight_decay", 1e-6, 1e-4, log=True)
    # settings.batch_size = trial.suggest_categorical("deeponet_batch_size", [256, 512, 1024])
    # settings.grad_clip = trial.suggest_categorical("deeponet_grad_clip", [0.0, 0.5, 1.0])
    # settings.warmup = trial.suggest_categorical("deeponet_warmup", [0, 200, 400, 600, 800])
    # settings.patience = trial.suggest_int("deeponet_patience", 150, 400, step=50)
    # settings.epochs = trial.suggest_int("deeponet_epochs", 1000, 5000, step=500)
    settings.exp_name = f"deeponet_optuna_trial{trial.number:03d}"
    settings.out_dir = str(OUTPUT_DIR / "deeponet")
    return settings


def evaluate_trial(trial: Trial, settings: TrainSettings) -> float:
    try:
        result = run_training(settings)
    except Exception as err:  # noqa: BLE001
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


# def save_summary(studies: Dict[str, optuna.Study]) -> None:
#     if not studies:
#         return
#     summary = {}
#     for model_type, study in studies.items():
#         completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
#         if not completed:
#             continue
#         best = min(completed, key=lambda t: t.value)
#         summary[model_type] = {
#             "best_value": float(best.value),
#             "best_params": best.params,
#             "user_attrs": best.user_attrs,
#         }
#     if not summary:
#         return
#     SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
#     SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def save_best_artifact(study: optuna.Study, model_type: str, mat_file: str, out_root: Path) -> None:
    """study에서 best trial을 골라 체크포인트를 별도 폴더에 복사하고, 메타데이터를 JSON으로 저장."""
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        print(f"[{model_type}] No completed trials; skip saving best artifact.")
        return

    best = min(completed, key=lambda t: t.value)
    ckpt_src = best.user_attrs.get("checkpoint")
    if not ckpt_src:
        print(f"[{model_type}] Best trial has no 'checkpoint' in user_attrs; skip.")
        return

    best_dir = out_root / model_type / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    # 파일명에 시간/값/트라이얼 번호를 반영해 가독성 확보
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(ckpt_src).suffix or ".pt"
    ckpt_dst = best_dir / f"best_{model_type}_trial{best.number:03d}_{stamp}{ext}"

    try:
        shutil.copyfile(ckpt_src, ckpt_dst)
    except FileNotFoundError:
        print(f"[{model_type}] checkpoint not found: {ckpt_src}")
        return

    meta = {
        "model_type": model_type,
        "mat_file": mat_file,
        "trial_number": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "user_attrs": best.user_attrs,
        "checkpoint_copied_to": str(ckpt_dst),
        "timestamp": stamp,
    }
    (best_dir / "best_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{model_type}] Best RMSE={best.value:.6f} (trial {best.number}) -> {ckpt_dst}")

def save_summary(studies: Dict[str, optuna.Study]) -> None:
    if not studies:
        return
    summary = {}
    for model_type, study in studies.items():
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            continue
        best = min(completed, key=lambda t: t.value)
        summary[model_type] = {
            "best_value": float(best.value),
            "best_params": best.params,
            "user_attrs": best.user_attrs,
            "best_trial_number": best.number,
        }
    if not summary:
        return
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {SUMMARY_PATH}")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    studies: Dict[str, optuna.Study] = {}
    for model_type in ("mlp", "deeponet"):
        n_trials = N_TRIALS.get(model_type, 0)
        if n_trials <= 0:
            continue

        print(f"[{model_type}] Start Optuna ({n_trials} trials)")
        study = create_study(model_type)
        objective = make_objective(model_type)
        study.optimize(objective, n_trials=n_trials)

        # 완료 여부 및 best 출력
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed:
            print(f"[{model_type}] No completed trials; all failed or pruned.")
        else:
            best = min(completed, key=lambda t: t.value)
            print(f"[{model_type}] Best RMSE={best.value:.6f} (trial {best.number})")

            # 여기서 바로 best 체크포인트 별도 저장
            save_best_artifact(study, model_type, MAT_FILES[0], OUTPUT_DIR)

        studies[model_type] = study

    save_summary(studies)
    
    
# def main() -> None:
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     optuna.logging.set_verbosity(optuna.logging.WARNING)

#     studies: Dict[str, optuna.Study] = {}
#     for model_type in ("mlp", "deeponet"):
#         n_trials = N_TRIALS.get(model_type, 0)
#         if n_trials <= 0:
#             continue
#         print(f"Starting Optuna study for {model_type} ({n_trials} trials)")
#         study = create_study(model_type)
#         objective = make_objective(model_type)
#         study.optimize(objective, n_trials=n_trials)
#         completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
#         if not completed:
#             print(f"No completed trials for {model_type}; all trials failed or were pruned.")
#         else:
#             best = min(completed, key=lambda t: t.value)
#             print(f"Best {model_type} RMSE: {best.value:.6f}")
#             # print(json.dumps(best.params, indent=2))
#         studies[model_type] = study

#     save_summary(studies)


if __name__ == "__main__":
    main()

# %%
