#%%
import json
from typing import Optional
from model_validation.train import DEFAULT_MAT_FILES, TrainSettings, run_training


mat_files = (
    "20250908_T_182846",
    "20250911_T_091324",
    "20250908_T_183632",
    "20250908_T_203220",
)

for mat_file in mat_files:
    BASE_TRAIN_SETTINGS = TrainSettings(
        model="deeponet",
        target="rdc",
        data_dir="dataset/data/tapered_seal",
        mat_files=(mat_file,),
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


# %%
