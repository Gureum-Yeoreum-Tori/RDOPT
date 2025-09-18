#%%
import json
from typing import Optional
from model_validation.train import DEFAULT_MAT_FILES, TrainSettings, run_training
from time import time as tt

mat_files = (
    "20250908_T_182846",
    "20250911_T_091324",
    "20250908_T_183632",
    "20250908_T_203220",
)

# MODEL_REGISTRY: Dict[str, nn.Module] = {
#     "mlp": SimpleMLP,
#     "multihead_mlp": MultiHeadMLP,
#     "deeponet": MultiHeadDeepONet,
#     "deeponet_single": DeepONet,
# }

for i, mat_file in enumerate(mat_files):
    settings = TrainSettings(
        model="deeponet",
        target="rdc",
        data_dir="dataset/data/tapered_seal",
        mat_files=(mat_file,),
        leak_index=6,
        rdc_indices=(4, 5, 2, 3),
        batch_size=512,
        epochs=1000,
        lr=1e-4,
        weight_decay=1e-6,
        hidden_layers= [64, 64, 64, 64],
        branch_layers= [128, 128, 128, 128],
        trunk_layers= [64, 64],
        param_embedding_dim=64,
        dropout=0.0,
        n_basis=64,
        warmup=500,
        patience=1000,
        grad_clip=0.0,
        seed=42,
        device=None,
        out_dir="net",
        exp_name=f"deeponet_{i}",
        baseline_alpha=1.0,
        head_names=None,
    )


    t0 = tt()
    result = run_training(settings)
    t1 = tt()
    print(json.dumps(result, indent=2))
    print(f'DeepONet_multi training time= {t1-t0}\n')
    
    settings2 = settings
    settings2.head_names=('K','k','C','c',)
    t0 = tt()
    result = run_training(settings2)
    t1 = tt()
    print(json.dumps(result, indent=2))
    print(f'training time= {t1-t0}\n')


# %%
