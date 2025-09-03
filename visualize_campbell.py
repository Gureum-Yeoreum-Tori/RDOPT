#%%
import os
import numpy as np
import matplotlib.pyplot as plt

from import_data import rotor_import
from loader_brg_seal import BearingNondModel, SealFNOModel, SealDONModel, SealLeakModel
from solver_rotordyn import assemble_system_matrix
from solver_rotordyn import eig_batch
from solver_rotordyn import eig_batch_with_vectors


def main():
    # Output dir
    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    # Speed grid (same as RDOPT defaults)
    w_range = np.array([1500, 6000]) * np.pi / 30
    n_w = 12
    w_vec = np.linspace(w_range[0], w_range[1], n_w)

    # Import rotor and matrices
    data_dir = 'dataset/data'
    rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
    rotor_sheet = "RDOPT"

    bs_params = {
        'mu_brg': 0.01,
        'mu_seal': 1.4e-3,
        'rho_seal': 850,
    }

    (n_ele, n_node, n_dof, n_add, n_brg, n_seal,
    rotor_elements, rotor_nodal_props, added_elements, added_props,
    mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb,
    brgs, seals, support_dofs) = rotor_import(file_path=rotor_file, sheet_name=rotor_sheet, bs_params=bs_params)

    rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
    C_struct = np.zeros_like(mat_K_r)

    # Models
    model_brg = BearingNondModel()
    model_seal = SealDONModel()
    model_seal_leak = SealLeakModel()

    # Parameter scalings
    f_brg_dim = np.array([[1, 1e-4], [1, 1e-4]])
    f_seal_dim = [1e-6, 1e-6, 1e-1]
    rdc_signs = np.array([1, 1, -1, 1])

    # Group seals by type for batched inference
    n_type_seal = 3
    groups = {}
    for i, s in enumerate(seals):
        groups.setdefault(s.SealNet, []).append(i)
    idx_seal = [np.array(groups.get(t+1, []), dtype=int) for t in range(n_type_seal)]

    # One sample X (similar to RDOPT quick test)
    X_brg = np.tile([21, 15], (n_brg, 1))[None, :, :]  # (1, n_brg, 2)
    X_seal = np.tile([200, 200, 0], (n_seal, 1))[None, :, :]  # (1, n_seal, 3)

    # Bearing rdc
    x_brg = X_brg * f_brg_dim
    K_brg, C_brg = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

    # Seal rdc
    pop = 1
    seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
    for t in range(n_type_seal):
        idx = idx_seal[t]
        if idx.size == 0:
            continue
        params_t = X_seal[:, idx]
        x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
        rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)
        seal_rdc[:, idx] = rdc_flat

    K_seal = seal_rdc[:, :, [2, 3, 3, 2], :] * rdc_signs[None, None, :, None]
    C_seal = seal_rdc[:, :, [0, 1, 1, 0], :] * rdc_signs[None, None, :, None]

    # Assemble overall K/C for all speeds
    K_vals = np.concatenate([K_brg, K_seal], axis=1)
    C_vals = np.concatenate([C_brg, C_seal], axis=1)
    K_all, Ceff_all = assemble_system_matrix(
        mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals
    )

    # Eigen without/with tracking
    eig_pre  = eig_batch_cpu_parallel(mat_M, K_all, Ceff_all, track=False, w_max=w_vec[-1])
    eig_post = eig_batch_cpu_parallel(mat_M, K_all, Ceff_all, track=True, w_max=w_vec[-1])

    # Campbell comparison: pick n modes with smallest |Imag(λ)| from forward branch
    n_eff = eig_pre.shape[2]
    n_pick = min(8, n_eff)
    beta_pre  = np.abs(eig_pre[0].imag)   # (W, n)
    beta_post = np.abs(eig_post[0].imag)  # (W, n)
    # Choose modes to plot based on average frequency magnitude over speeds
    score_pre = beta_pre.mean(axis=0)
    idx_plot = np.argsort(score_pre)[:n_pick]

    rpm = w_vec * 30.0 / np.pi
    plt.figure(figsize=(8, 5))
    for k in idx_plot:
        plt.plot(rpm, beta_pre[:, k] * 30.0 / np.pi, 'k--', alpha=0.6)
        plt.plot(rpm, beta_post[:, k] * 30.0 / np.pi, 'b-')
    plt.xlabel('Speed [RPM]')
    plt.ylabel('Modal Frequency [RPM]')
    plt.title('Campbell Diagram: pre (dashed) vs post (solid) tracking')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 6000])
    plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'campbell_compare.png'), dpi=150)
    # plt.close()

    # Mode shapes: use tracked vectors at a chosen speed (e.g., mid speed)
    eigvals_trk, phi_fwd = eig_batch_with_vectors(mat_M, K_all, Ceff_all, track=True, w_max=w_vec[-1])
    iw = n_w // 2
    Phi = phi_fwd[0, iw]  # (n, n)

    # Node positions along axis
    vec_L = np.array([e.L for e in rotor_elements])
    z_nodes = np.concatenate(([0.0], np.cumsum(vec_L)))  # (n_node,)

    # Build node indices for x,y DOFs
    idx_x = np.arange(n_node) * 4
    idx_y = idx_x + 2

    # Plot first few picked modes
    os.makedirs(out_dir, exist_ok=True)
    for i, k in enumerate(idx_plot[:min(4, len(idx_plot))]):
        v = Phi[:, k]  # (n_dof,)
        vx = v[idx_x]
        vy = v[idx_y]
        amp = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2)  # magnitude along axis
        # Normalize for visualization
        amp = amp / (np.max(amp) + 1e-12)
        plt.figure(figsize=(8, 3))
        plt.plot(z_nodes, amp, 'r-')
        plt.xlabel('Axial Position [m]')
        plt.ylabel('Normalized |disp|')
        fr = np.abs(eigvals_trk[0, iw, k].imag) * 30.0 / np.pi
        sp = float(rpm[iw])
        plt.title(f'Mode shape (tracked) k={k}, speed={sp:.0f} RPM, freq~{fr:.0f} RPM')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.savefig(os.path.join(out_dir, f'mode_shape_k{k}_iw{iw}.png'), dpi=150)
        # plt.close()

    # print("Saved:")
    # print(" -", os.path.join(out_dir, 'campbell_compare.png'))
    # for k in idx_plot[:min(4, len(idx_plot))]:
    #     print(" -", os.path.join(out_dir, f'mode_shape_k{k}_iw{n_w//2}.png'))


# if __name__ == '__main__':
#     main()

# Import rotor and matrices
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

# Models
model_brg = BearingNondModel()
model_seal = SealDONModel()
model_seal_leak = SealLeakModel()

#%%

# Speed grid (same as RDOPT defaults)
w_range = np.array([500, 7000]) * np.pi / 30
n_w = 14
w_vec = np.linspace(w_range[0], w_range[1], n_w)

bs_params = {
    'mu_brg': 0.001,
    'mu_seal': 1.4e-3,
    'rho_seal': 850,
}

(n_ele, n_node, n_dof, n_add, n_brg, n_seal,
rotor_elements, rotor_nodal_props, added_elements, added_props,
mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb,
brgs, seals, support_dofs) = rotor_import(file_path=rotor_file, sheet_name=rotor_sheet, bs_params=bs_params)

rows_sup, cols_sup = support_dofs.rows, support_dofs.cols
C_struct = np.zeros_like(mat_K_r)

# Parameter scalings
f_brg_dim = np.array([[1, 1e-4], [1, 1e-4]])
f_seal_dim = [1e-6, 1e-6, 1e-1]
rdc_signs = np.array([1, 1, -1, 1])

# Group seals by type for batched inference
n_type_seal = 3
groups = {}
for i, s in enumerate(seals):
    groups.setdefault(s.SealNet, []).append(i)
idx_seal = [np.array(groups.get(t+1, []), dtype=int) for t in range(n_type_seal)]

# One sample X (similar to RDOPT quick test)
X_brg = np.tile([21, 15], (n_brg, 1))[None, :, :]  # (1, n_brg, 2)
X_seal = np.tile([200, 200, 0], (n_seal, 1))[None, :, :]  # (1, n_seal, 3)

# Bearing rdc
x_brg = X_brg * f_brg_dim
K_brg, C_brg, Pl = model_brg.calculate_brg_rdc_batch(brgs=brgs, params_batch=x_brg, w_vec=w_vec)

# Seal rdc
pop = 1
seal_rdc = np.zeros((pop, n_seal, 4, n_w), dtype=float)
for t in range(n_type_seal):
    idx = idx_seal[t]
    if idx.size == 0:
        continue
    params_t = X_seal[:, idx]
    x_seal = (params_t.reshape(-1, 3) * f_seal_dim)
    rdc_flat  = model_seal.predict(t+1, x_seal, w_vec).reshape(pop, len(idx), 4, n_w)
    seal_rdc[:, idx] = rdc_flat

K_seal = seal_rdc[:, :, [2, 3, 3, 2], :] * rdc_signs[None, None, :, None]
C_seal = seal_rdc[:, :, [0, 1, 1, 0], :] * rdc_signs[None, None, :, None]

# Assemble overall K/C for all speeds
K_vals = np.concatenate([K_brg, K_seal], axis=1)
C_vals = np.concatenate([C_brg, C_seal], axis=1)
K_all, Ceff_all = assemble_system_matrix(
    mat_K_r, C_struct, mat_C_g, w_vec, rows_sup, cols_sup, K_vals, C_vals
)

# Eigen without/with tracking
eig_pre, _  = eig_batch(mat_M, K_all, Ceff_all, track=False)
eig_post, _ = eig_batch(mat_M, K_all, Ceff_all, track=True)

# Campbell comparison: pick n modes with smallest |Imag(λ)| from forward branch
n_eff = eig_pre.shape[2]
n_pick = min(8, n_eff)
beta_pre  = np.abs(eig_pre[0].imag)   # (W, n)
beta_post = np.abs(eig_post[0].imag)  # (W, n)
# Choose modes to plot based on average frequency magnitude over speeds
score_pre = beta_pre.mean(axis=0)
idx_plot = np.argsort(score_pre)[:n_pick]

rpm = w_vec * 30.0 / np.pi
plt.figure(figsize=(8, 5))
for k in idx_plot:
    plt.plot(rpm, beta_pre[:, k] * 30.0 / np.pi, '--')
    plt.plot(rpm, beta_post[:, k] * 30.0 / np.pi, '-', linewidth = 2, alpha=0.6)
plt.plot(rpm, rpm, 'k-')
plt.xlabel('Speed [RPM]')
plt.ylabel('Modal Frequency [RPM]')
plt.title('Campbell Diagram: pre (dashed) vs post (solid) tracking')
plt.grid(True, alpha=0.3)
plt.ylim([0, 7000])
plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'campbell_compare.png'), dpi=150)
# plt.close()

#%%
# # Mode shapes: use tracked vectors at a chosen speed (e.g., mid speed)
# eigvals_trk, phi_fwd = eig_batch_with_vectors(mat_M, K_all, Ceff_all, track=True, w_max=w_vec[-1]*2)
# iw = n_w // 2
# Phi = phi_fwd[0, iw]  # (n, n)

# # Node positions along axis
# vec_L = np.array([e.L for e in rotor_elements])
# z_nodes = np.concatenate(([0.0], np.cumsum(vec_L)))  # (n_node,)

# # Build node indices for x,y DOFs
# idx_x = np.arange(n_node) * 4
# idx_y = idx_x + 2

# # Plot first few picked modes
# # os.makedirs(out_dir, exist_ok=True)
# for i, k in enumerate(idx_plot[:min(4, len(idx_plot))]):
#     v = Phi[:, k]  # (n_dof,)
#     vx = v[idx_x]
#     vy = v[idx_y]
#     amp = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2)  # magnitude along axis
#     # Normalize for visualization
#     amp = amp / (np.max(amp) + 1e-12)
#     plt.figure(figsize=(8, 3))
#     plt.plot(z_nodes, amp, 'ro-')
#     plt.xlabel('Axial Position [m]')
#     plt.ylabel('Normalized |disp|')
#     fr = np.abs(eigvals_trk[0, iw, k].imag) * 30.0 / np.pi
#     sp = float(rpm[iw])
#     plt.title(f'Mode shape (tracked) k={k}, speed={sp:.0f} RPM, freq~{fr:.0f} RPM')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     # plt.savefig(os.path.join(out_dir, f'mode_shape_k{k}_iw{iw}.png'), dpi=150)
#     # plt.close()

# # print("Saved:")
# # print(" -", os.path.join(out_dir, 'campbell_compare.png'))
# # for k in idx_plot[:min(4, len(idx_plot))]:
# #     print(" -", os.path.join(out_dir, f'mode_shape_k{k}_iw{n_w//2}.png'))

#%%
## end







# %%
