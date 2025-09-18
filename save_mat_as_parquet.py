# save_mat_as_parquet.py
import os
import numpy as np
import pandas as pd
import h5py

def convert_mat(mat_path: str, save_path: str) -> None:
    with h5py.File(mat_path, "r") as mat:
        input_ = np.array(mat["input"]).T.astype(np.float32)            # (n_data, n_para)
        Leak = np.array(mat["Leak"])[6,:].reshape(-1,1)
        rdc = np.array(mat["RDC"])[2:6].transpose(2, 1, 0).astype(np.float32)
        # shape: (n_data, n_vel, 4)  -> heads = [K, k, C, c]
        n_data = input_.shape[0]  # number of samples
        assert rdc.shape[0] == n_data, f"RDC first axis ({rdc.shape[0]}) must equal n_data ({n_data})"
        w_vec = np.array(mat["params/wVec"]).squeeze().astype(np.float32)

    df = pd.DataFrame(input_, columns=[f"param_{i}" for i in range(input_.shape[1])])
    df["Leak"] = Leak.astype(np.float32).reshape(-1)
    df["rdc"] = [sample.astype(np.float32).tolist() for sample in rdc]
    df["w_vec"] = [w_vec.astype(np.float32).tolist()] * n_data
    df["id"] = np.arange(n_data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_parquet(save_path, index=False)

if __name__ == "__main__":
    data_dir = 'dataset/data/tapered_seal'
    mat_files = ('20250908_T_182846','20250911_T_091324','20250908_T_183632','20250908_T_203220',)
    for mat_file in mat_files:
        mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')
        parq_path = os.path.join('dataset/data', f'{mat_file}.parquet')
        convert_mat(
            mat_path,
            parq_path,
        )
