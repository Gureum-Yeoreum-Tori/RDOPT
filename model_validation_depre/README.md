# Seal Model Validation Toolkit

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place source data (CSV/HDF5/MAT) under the paths referenced in `configs/example.yaml`. Feature/target names must match the config.

## Training
```bash
cd model_validation
python train.py --config configs/example.yaml --exp_name mlp_leak
```
Optional overrides:
```bash
python train.py --config configs/example.yaml --exp_name mlp_leak \
  --override training.lr=1e-3 model.hidden_layers=[512,512,256] training.batch_size=1024
```
Key flags: `--mixed_precision/--no-mixed-precision`, `--seed`, `--resume path.ckpt`, `--fold N`, `--kfold K`.

## Evaluation
```bash
cd model_validation
python eval.py --config configs/example.yaml --exp_name mlp_leak --checkpoint runs/mlp_leak/checkpoints/best.ckpt
```
Outputs: metrics JSON, `predictions.parquet`, parity/error/curve plots, optional baseline results.

## Hyperparameter Tuning
```bash
cd model_validation
python tune.py --config configs/example.yaml --exp_name search_mlp --n_trials 50
```
The study database is stored at `runs/<exp>/tuning.db`. The best trial’s checkpoint/config are copied to `runs/<exp>/checkpoints/tune_best.ckpt` and `runs/<exp>/best_trial_config.json`.

## Extending Models & Targets
- Add new heads/targets by updating `data.targets` and `model.heads`; multi-head models auto-align names.
- Implement custom architectures in `models.py` and register selection logic within `train.py`.
- Extend metrics via `utils.METRIC_FUNCS`.

## Reproducibility & Artifacts
- Global seeding, deterministic flags, and per-trial logging are enabled by default.
- Scalers, checkpoints, predictions, and plots are saved under `runs/<exp_name>/` for auditability.
