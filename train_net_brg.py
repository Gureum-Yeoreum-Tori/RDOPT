from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from model_validation.train import TrainSettings, run_training

BEST_META_PATHS: Dict[str, Path] = {
    "mlp": Path("net/optuna/brg/mlp/best/best_meta.json"),
    "deeponet": Path("net/optuna/brg/deeponet/best/best_meta.json"),
}

DEFAULT_DATA_DIR = Path("dataset/data/fixed")
DEFAULT_OUT_ROOT = Path("net") / "fixed/tuned_runs"
DEFAULT_HEAD_NAMES = ("Kxx", "Kxy", "Kyx", "Kyy", "Cxx", "Cxy", "Cyx", "Cyy")
DEFAULT_LEAK_INDEX = 6
DEFAULT_RDC_INDICES = (2, 3, 4, 5)
DEFAULT_PATIENCE = 200
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Dataset folder names to train on (defaults to every subdirectory in data-dir, excluding the one used for tuning).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=("mlp", "deeponet"),
        default=("mlp", "deeponet"),
        help="Model types to train.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device spec passed to TrainSettings (e.g., cuda, mps).",
    )
    parser.add_argument(
        "--include-tuned-dataset",
        action="store_true",
        help="Also train on the dataset that was used during hyperparameter tuning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without running training.",
    )
    return parser.parse_args()


def available_datasets(data_dir: Path) -> List[str]:
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


def load_best_meta(model_type: str) -> dict:
    meta_path = BEST_META_PATHS.get(model_type)
    if meta_path is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Best meta file not found for {model_type}: {meta_path}")
    return json.loads(meta_path.read_text())


def best_dataset_from_meta(meta: dict) -> str:
    return str(meta.get("mat_file", ""))


def extract_mlp_hidden_layers(params: dict) -> List[int]:
    n_layers = int(params.get("mlp_hidden_n_layers", 0))
    widths: List[int] = []
    for idx in range(n_layers):
        widths.append(int(params[f"mlp_hidden_width_{idx}"]))
    return widths


def extract_deeponet_layers(params: dict, prefix: str) -> List[int]:
    n_layers = int(params.get(f"deeponet_{prefix}_n_layers", 0))
    widths: List[int] = []
    for idx in range(n_layers):
        widths.append(int(params[f"deeponet_{prefix}_width_{idx}"]))
    return widths


def build_mlp_settings(
    dataset: str,
    data_dir: Path,
    out_root: Path,
    device: str | None,
    meta: dict,
) -> TrainSettings:
    params = meta["best_params"]
    hidden_layers = extract_mlp_hidden_layers(params)
    settings = TrainSettings(
        model="mlp",
        target="leak",
        data_dir=str(data_dir),
        mat_files=(dataset,),
        leak_index=DEFAULT_LEAK_INDEX,
        rdc_indices=DEFAULT_RDC_INDICES,
        batch_size=256,
        epochs=int(params.get("mlp_epochs", 3000)),
        lr=float(params["mlp_lr"]),
        weight_decay=float(params["mlp_weight_decay"]),
        activation=str(params.get("mlp_activation", "relu")),
        hidden_layers=tuple(hidden_layers),
        branch_layers=(128, 128, 128),
        trunk_layers=(64, 64),
        param_embedding_dim=0,
        dropout=0.0,
        n_basis=0,
        warmup=0,
        patience=DEFAULT_PATIENCE,
        grad_clip=0.0,
        seed=DEFAULT_SEED,
        device=device,
        out_dir=str(out_root / "mlp"),
        exp_name=f"mlp_leak_{dataset}_tuned",
        baseline_alpha=1.0,
        head_names=None,
        layernorm=bool(params.get("mlp_layernorm", False)),
    )
    return settings


def build_deeponet_settings(
    dataset: str,
    data_dir: Path,
    out_root: Path,
    device: str | None,
    meta: dict,
) -> TrainSettings:
    params = meta["best_params"]
    branch_layers = extract_deeponet_layers(params, "branch")
    trunk_layers = extract_deeponet_layers(params, "trunk")
    settings = TrainSettings(
        model="deeponet",
        target="rdc",
        data_dir=str(data_dir),
        mat_files=(dataset,),
        leak_index=DEFAULT_LEAK_INDEX,
        rdc_indices=DEFAULT_RDC_INDICES,
        batch_size=256,
        epochs=5000,
        lr=float(params["deeponet_lr"]),
        weight_decay=float(params["deeponet_weight_decay"]),
        activation=str(params.get("deeponet_activation", "gelu")),
        hidden_layers=tuple(branch_layers),
        branch_layers=tuple(branch_layers),
        trunk_layers=tuple(trunk_layers),
        param_embedding_dim=int(params.get("deeponet_param_dim", 64)),
        dropout=0.0,
        n_basis=int(params.get("deeponet_latent_dim", 32)),
        warmup=500,
        patience=DEFAULT_PATIENCE,
        grad_clip=0.0,
        seed=DEFAULT_SEED,
        device=device,
        out_dir=str(out_root / "deeponet"),
        exp_name=f"deeponet_rdc_{dataset}_tuned",
        baseline_alpha=1.0,
        head_names=DEFAULT_HEAD_NAMES,
        layernorm=False,
    )
    return settings


BUILDERS = {
    "mlp": build_mlp_settings,
    "deeponet": build_deeponet_settings,
}


def plan_jobs(
    datasets: Sequence[str],
    models: Sequence[str],
    data_dir: Path,
    out_root: Path,
    device: str | None,
) -> List[tuple[str, str, TrainSettings]]:
    jobs: List[tuple[str, str, TrainSettings]] = []
    for model_type in models:
        meta = load_best_meta(model_type)
        builder = BUILDERS[model_type]
        for dataset in datasets:
            settings = builder(dataset, data_dir, out_root, device, meta)
            jobs.append((model_type, dataset, settings))
    return jobs


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidate_datasets = available_datasets(data_dir)
    if not candidate_datasets:
        raise RuntimeError(f"No datasets detected under {data_dir}")

    selected_datasets: Iterable[str]
    if args.datasets:
        invalid = [name for name in args.datasets if name not in candidate_datasets]
        if invalid:
            raise ValueError(f"Unknown datasets requested: {invalid}")
        selected_datasets = args.datasets
    else:
        tuned_sources = {
            model: best_dataset_from_meta(load_best_meta(model))
            for model in ("mlp", "deeponet")
        }
        tuned_exclusions = {name for name in tuned_sources.values() if name}
        if args.include_tuned_dataset:
            selected_datasets = candidate_datasets
        else:
            selected_datasets = [
                name for name in candidate_datasets
                if name not in tuned_exclusions
            ]
        if not selected_datasets:
            selected_datasets = candidate_datasets

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    models = tuple(dict.fromkeys(args.models))  # preserve order, drop duplicates
    jobs = plan_jobs(selected_datasets, models, data_dir, out_root, args.device)

    print("Planned training jobs:")
    for model_type, dataset, settings in jobs:
        print(f"  - {model_type:<8} on {dataset} -> exp {settings.exp_name}")

    if args.dry_run:
        return

    summary: List[dict] = []
    for model_type, dataset, settings in jobs:
        print(f"\n[{model_type}] Training on {dataset}")
        result = run_training(settings)
        summary.append(
            {
                "model": model_type,
                "dataset": dataset,
                "checkpoint": result.get("checkpoint"),
                "metrics": result.get("metrics"),
                "best_val_loss": result.get("best_val_loss"),
            }
        )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
