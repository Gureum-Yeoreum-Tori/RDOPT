#%%
import torch
import os
import matplotlib.pyplot as plt
from make_paper_figures import (
    _default_rcparams,
    figsize_F,
    figsize_SC,
    figsize_DC,
    figsize_DC_b,
)

model_dir='net/'
model_dir = model_dir
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_files = {
    1: 'deeponet_multihead_20250908_T_182846.pth',
    2: 'deeponet_multihead_20250911_T_091324.pth',
    3: 'deeponet_multihead_20250908_T_183632.pth',
    4: 'deeponet_multihead_20250908_T_203220.pth',
}

for model_id, filename in model_files.items():
    ckpt_path = os.path.join(model_dir, filename)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt['hparams']
    add = ckpt['additional']
    spl = ckpt['splits']
    hist = ckpt['train_history']
    loss_train = hist['loss_train'][:]
    loss_val = hist['loss_val'][:]
    epochs = len(loss_val)
    
    _default_rcparams()
    plt.figure(figsize=figsize_DC)
    plt.plot(loss_train, label="Train loss")
    plt.plot(loss_val, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.yscale('log', base=10)
    plt.savefig(f'training_{model_id}.png',dpi=600,bbox_inches="tight")

# %%
