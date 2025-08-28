import os
import numpy as np
from import_data import rotor_import, calculate_bearing_loads
from loader_brg_seal import BearingH5Loader, SealFNOModel, SealDONModel
from collections import defaultdict
from solver_seal import main_seal_solver as seal_solver
# from solver_rotordyn import BrgModel, SealModel
# from torch.linalg import inv
import torch
import matplotlib.pyplot as plt



## Import data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

w_range = np.array([500, 6000]) * np.pi / 30
w_oper = 3500 * np.pi / 30
oper = {
        'w_min': w_range[0],
        'w_max': w_range[1],
        'range': w_oper,
    }
bs_params = {
        'mu_brg': 0.04, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }

n_w = 12
n_pop = 1
w_vec = np.linspace(w_range[0], w_range[1], n_w)

n_ele, n_node, n_dof, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb, brgs, seals = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)


matrix_params = {
    'mat_M': mat_M,
    'mat_K_r': mat_K_r,
    'mat_C_g': mat_C_g,
    'n_ele': n_ele,
    'n_node': n_node,
    'n_dof': n_dof,
}

## Load brg, seal model
brg = BearingH5Loader()
seal = SealFNOModel()




## predict bearing and seal rdc. single case

# # brg
# brg = BearingH5Loader()
# brg1_id = np.random.randint(1,55,size=n_pop)[:,None] # bearing_id: 1 ~ 55
# brg2_id = np.random.randint(1,55,size=n_pop)[:,None] 
# brg1_cr = np.random.randint(10,30,size=n_pop)[:,None] # Cr/D = 10/10000 ~ 30/10000
# brg2_cr = np.random.randint(10,30,size=n_pop)[:,None] 
# brg_params = np.concatenate([brg1_id, brg1_cr, brg2_id, brg2_cr], axis=1)
# brg_params = np.array([[21, 10, 21, 10]])


# K_brg = np.zeros([n_pop,n_w,4,2])
# C_brg = np.zeros([n_pop,n_w,4,2])
# for i in range(len(brgs)):
#     for j in range(n_pop):
#         K_, C_, _ = brg.calculate_bearing_coefficients(
#             brg_params[j,2*(i-1)].astype(int), 
#             brgs[i].Db, 
#             brgs[i].Db*brg_params[j,2*(i-1)+1]*1e-4, 
#             brgs[i].mu, 
#             brgs[i].load, 
#             w_vec
#         )
#         K_brg[j,:, :, i] = np.stack([
#                             K_[:,0,0],  # Kxx
#                             K_[:,0,1],  # Kxy
#                             K_[:,1,0],  # Kyx
#                             K_[:,1,1],  # Kyy
#                         ], axis=1)
#         C_brg[j,:, :, i] = np.stack([
#                             C_[:,0,0],  # Kxx
#                             C_[:,0,1],  # Kxy
#                             C_[:,1,0],  # Kyx
#                             C_[:,1,1],  # Kyy
#                         ], axis=1)

# # seal
# seal = SealFNOModel()

# groups = defaultdict(list)          # {SealNet: [원본 인덱스,...]}
# for i, s in enumerate(seals):
#     groups[s.SealNet].append(i)
# outputs = [None] * len(seals)  # 원래 순서로 채울 버퍼

# n_types = 3

# h_in  = np.random.randint(100, 500, size=(n_pop, n_seal, 1)) # radial clearance, 100 um ~ 200 um
# h_out = np.random.randint(100, 500, size=(n_pop, n_seal, 1))
# psr   = np.random.randint(1, 10,   size=(n_pop, n_seal, 1))

# h_in = 100*np.ones((n_pop, n_seal, 1))
# h_out = 100*np.ones((n_pop, n_seal, 1))
# psr = 0*np.ones((n_pop, n_seal, 1))

# params_per_type = []
# for t in range(n_types):
#     idx = np.array(groups[t+1], dtype=int)
#     n_k = len(idx)
#     # (n_pop, n_k, 3)
#     params_t = np.concatenate([h_in[:, idx, :]*1e-6, h_out[:, idx, :]*1e-6, psr[:, idx, :]*1e-1], axis=-1)
#     params_per_type.append(params_t)
    
# x_type1 = params_per_type[0].reshape(-1, 3)
# x_type2 = params_per_type[1].reshape(-1, 3)
# x_type3 = params_per_type[2].reshape(-1, 3)
    
# y_type1 = seal.predict(1,x_type1,w_vec)
# y_type2 = seal.predict(2,x_type2,w_vec)
# y_type3 = seal.predict(3,x_type3,w_vec)


# geometry = {
#     'hIn': x_type1[0,0],
#     'hOut': x_type1[0,1],
#     'Ds': seals[0].Ds,
#     'Ls': seals[0].Ls,
#     'NxSeal': 25,
# }

# fluid = {
#     'mu': bs_params['mu_seal'],
#     'rho': bs_params['rho_seal'],   
# }

# op_conditions = {
#     'dp': seals[0].dp,
#     'w_vec': w_vec,
#     'psr': x_type1[0,2],
# }
# _, RDC, _, _, _, _, _ = seal_solver(geometry=geometry,fluid=fluid,op_conditions=op_conditions)

# idx_w = 0
# r1 = RDC[idx_w,2:6]
# r2 = y_type1[0,:,idx_w]
# r1-r2


# def xy_dofs(node_idx: int) -> tuple[int,int]:
#     """노드 인덱스(0-based) -> 전역 행렬에서 x,y DOF 인덱스."""
#     base = 4*node_idx
#     return base + 0, base + 2   # [x, thx, y, thy] 가정

# def submat_from_bearing(vec4: np.ndarray) -> np.ndarray:
#     """vec4 = [Kxx, Kxy, Kyx, Kyy] 또는 [Cxx, Cxy, Cyx, Cyy] -> (2,2)"""
#     return np.array([[vec4[0], vec4[1]],
#                      [vec4[2], vec4[3]]], dtype=float)

# def submat_from_seal(K: float, k: float, C: float, c: float) -> tuple[np.ndarray,np.ndarray]:
#     """씰 구조 Kxx=Kyy=K, Kxy=-Kyx=k; C도 동일 패턴."""
#     K2 = np.array([[K,  k],
#                    [-k, K]], dtype=float)
#     C2 = np.array([[C,  c],
#                    [-c, C]], dtype=float)
#     return K2, C2

# def assemble_support_mats_per_speed(n_dof: int,
#                                     w_vec: np.ndarray,
#                                     # 베어링 계수: K_brg, C_brg shape = (n_pop, n_w, 4, n_brg)
#                                     K_brg: np.ndarray,
#                                     C_brg: np.ndarray,
#                                     brg_nodes: list[int],   # 각 베어링의 노드(0-based)
#                                     # 씰 계수: 각 타입/개체 정리 후 speed별로 (K,k,C,c)를 제공
#                                     seal_KkCc_per_speed: list[dict],
#                                     # 예: [{'node': i_node, 'K': Kw, 'k': kw, 'C': Cw, 'c': cw}, ...]
#                                    ):
#     """
#     반환: K_sup, C_sup shape = (n_pop, n_w, n_dof, n_dof)
#     """
#     n_pop, n_w, _, n_brg = K_brg.shape
#     K_sup = np.zeros((n_pop, n_w, n_dof, n_dof))
#     C_sup = np.zeros((n_pop, n_w, n_dof, n_dof))

#     for j in range(n_pop):
#         for iw, w in enumerate(w_vec):
#             # --- 베어링 추가 ---
#             for b in range(n_brg):
#                 ix, iy = xy_dofs(brg_nodes[b])
#                 K2 = submat_from_bearing(K_brg[j, iw, :, b])   # (2,2)
#                 C2 = submat_from_bearing(C_brg[j, iw, :, b])   # (2,2)

#                 # 전역 조립
#                 dofs = [ix, iy]
#                 for a, I in enumerate(dofs):
#                     for b_, J in enumerate(dofs):
#                         K_sup[j, iw, I, J] += K2[a, b_]
#                         C_sup[j, iw, I, J] += C2[a, b_]

#             # --- 씰 추가 ---
#             # seal_KkCc_per_speed 는 같은 iw에 대응하는 리스트(여러 씰)라고 가정
#             for item in seal_KkCc_per_speed[iw]:
#                 ix, iy = xy_dofs(item['node'])
#                 K2, C2 = submat_from_seal(item['K'], item['k'], item['C'], item['c'])
#                 dofs = [ix, iy]
#                 for a, I in enumerate(dofs):
#                     for b_, J in enumerate(dofs):
#                         K_sup[j, iw, I, J] += K2[a, b_]
#                         C_sup[j, iw, I, J] += C2[a, b_]
#     return K_sup, C_sup

# def build_system_mats(mat_K_r: np.ndarray,
#                       mat_C_g: np.ndarray,
#                       K_sup: np.ndarray,
#                       C_sup: np.ndarray,
#                       w_vec: np.ndarray):
#     """
#     K(ω) = K_r + K_sup(ω)
#     C(ω) = ω*C_g + C_sup(ω)
#     입력 K_sup, C_sup shape = (n_pop, n_w, n_dof, n_dof)
#     반환 K_sys, C_sys 동일 shape
#     """
#     n_pop, n_w, n_dof, _ = K_sup.shape
#     K_sys = np.zeros_like(K_sup)
#     C_sys = np.zeros_like(C_sup)
#     for j in range(n_pop):
#         for iw, w in enumerate(w_vec):
#             K_sys[j, iw] = mat_K_r + K_sup[j, iw]
#             C_sys[j, iw] = w*mat_C_g + C_sup[j, iw]
#     return K_sys, C_sys

# def _reshape_type(y_type, idx_list):
#     n_k = len(idx_list)
#     return y_type.reshape(n_pop, n_k, 4, n_w)  # (n_pop, n_k, 4, n_w)

# y1 = _reshape_type(y_type1, groups[1])
# y2 = _reshape_type(y_type2, groups[2])
# y3 = _reshape_type(y_type3, groups[3])

# outputs = np.zeros((n_pop, n_seal, 4, n_w), dtype=float)

# for k, i in enumerate(groups[1]):
#     outputs[:, i, :, :] = y1[:, k, :, :]
# for k, i in enumerate(groups[2]):
#     outputs[:, i, :, :] = y2[:, k, :, :]
# for k, i in enumerate(groups[3]):
#     outputs[:, i, :, :] = y3[:, k, :, :]
    
# seal_KkCc_per_speed = []
# for iw in range(n_w):
#     lst = []
#     for i, s in enumerate(seals):
#         # 채널: [C, c, K, k]
#         C_ = rdc_seal*float(outputs[0, i, 0, iw])
#         c_ = rdc_seal*float(outputs[0, i, 1, iw])
#         K_ = rdc_seal*float(outputs[0, i, 2, iw])
#         k_ = rdc_seal*float(outputs[0, i, 3, iw])
#         lst.append({
#             'node': int(s.node - 1),  # xy_dofs가 0-based이므로 -1
#             'K': K_, 'k': k_, 'C': C_, 'c': c_
#         })
#     seal_KkCc_per_speed.append(lst)
    

# brg_nodes = [b.node for b in brgs]

# K_sup, C_sup = assemble_support_mats_per_speed(
#     n_dof=n_dof,
#     w_vec=w_vec,
#     K_brg=rdc_brg*K_brg,          # (n_pop, n_w, 4, n_brg)
#     C_brg=rdc_brg*C_brg,
#     brg_nodes=brg_nodes,  # 각 베어링이 연결된 노드
#     seal_KkCc_per_speed=seal_KkCc_per_speed
# )

# K_sys, C_sys = build_system_mats(
#     mat_K_r=mat_K_r,
#     mat_C_g=mat_C_g,
#     K_sup=K_sup,
#     C_sup=C_sup,
#     w_vec=w_vec
# )

# # K_sys, C_sys = build_system_mats(
# #     mat_K_r=mat_K_r,
# #     mat_C_g=mat_C_g,
# #     K_sup=0*K_sup,
# #     C_sup=0*C_sup,
# #     w_vec=w_vec
# # )

# import numpy as np
# from scipy.linalg import eig

# def modal_scan(mat_M: np.ndarray,
#                K_sys: np.ndarray,
#                C_sys: np.ndarray):

#     n_pop, n_w, n_dof, _ = K_sys.shape
#     I = np.eye(n_dof)
#     Z = np.zeros((n_dof, n_dof))

#     raw_eig = np.empty((n_pop, n_w, n_dof))
#     raw_V   = np.empty((n_pop, n_w, n_dof, n_dof), dtype=np.complex128)

#     for j in range(n_pop):
#         for i in range(n_w):
#             A = np.block([[-C_sys[j, i], -K_sys[j, i]],
#                           [ I          ,  Z           ]])
#             B = np.block([[ mat_M,  Z ],
#                           [ Z    ,  I ]])
#             w, V = eig(A, B)

#             idx_all = np.argsort(np.imag(w))
#             idx_sel = idx_all[-n_dof:]

#             raw_eig[j, i] = np.imag(w[idx_sel])
#             raw_V[j, i]   = V[n_dof:, :][:, idx_sel]
    
#     sigma = np.real(raw_eig)
#     imag_part = np.imag(raw_eig)
#     zeta = -sigma / np.sqrt(sigma**2 + imag_part**2)

#     return raw_eig, raw_V, zeta


# raw_eig, raw_vec, zeta = modal_scan(mat_M, K_sys, C_sys)

# # --- Campbell diagram plotting ---

# def plot_campbell(w_vec: np.ndarray,
#                   raw_eig: np.ndarray,
#                   modes: list[int] | None = None,
#                   orders: tuple[int, ...] = (1, 2, 3),
#                   j: int = 0,
#                   title: str = "Campbell Diagram") -> None:
#     """
#     w_vec: (n_w,) shaft speed [rad/s]
#     raw_eig: (n_pop, n_w, n_dof)  # imag(λ) already, [rad/s]
#     modes: indices of modes to plot along speed; default first min(8, n_dof)
#     orders: excitation line orders (1x, 2x, ...)
#     j: population index to visualize
#     """
#     n_w = w_vec.shape[0]
#     n_dof = raw_eig.shape[2]
#     if modes is None:
#         modes = list(range(min(8, n_dof)))

#     # axes: x = shaft speed [RPM], y = frequency [Hz]
#     rpm = w_vec * 60.0 / (2*np.pi)
#     f_modes = raw_eig[j] / (2*np.pi)  # (n_w, n_dof) in Hz
#     f_shaft = w_vec / (2*np.pi)               # (n_w,) Hz

#     plt.figure()
#     # plot modes
#     for m in modes:
#         plt.plot(rpm, f_modes[:, m])

#     # excitation lines r×
#     # for r in orders:
#     #     plt.plot(rpm, r * f_shaft, linestyle='--', linewidth=1)

#     plt.xlabel('Shaft speed (RPM)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title(title)
#     plt.grid(True, which='both', linestyle=':')

# # draw Campbell
# plot_campbell(w_vec, raw_eig, modes=None, orders=(1,2,3), j=0, title='Campbell Diagram (pop 0)')


# def build_unbalance_force(unb, n_dof, w_vec: np.ndarray):
#     unb_force = np.zeros((n_w, n_dof),dtype=np.complex128)
#     nodes = np.array(unb.cases[0].node, dtype=int)
#     dofs = np.ravel(np.column_stack([4*nodes, 4*nodes+2]))

#     for iw, w in enumerate(w_vec):
#         fvec = np.zeros(n_dof, dtype=np.complex128)
#         f = np.asarray(unb.cases[0].force, dtype=np.complex128)
#         fvec[dofs] = f * (w**2)
#         unb_force[iw] = fvec
#     return unb_force

# unb_force = build_unbalance_force(unb,n_dof,w_vec)


# def forced_response(mat_M: np.ndarray,
#                K_sys: np.ndarray,
#                C_sys: np.ndarray,
#                unb_force: np.ndarray,
#                w_vec: np.ndarray):
#     n_w, n_dof = unb_force.shape
#     resp = np.empty((n_w, n_dof), dtype=np.complex128)
    
#     for i, w in enumerate(w_vec):
#         resp[i] = np.linalg.solve(K_sys[0,i] - w**2*mat_M + 1j*w*C_sys[0,i], unb_force[i])
        
#     return resp

# resp = forced_response(mat_M, K_sys, C_sys, unb_force, w_vec)

# # Plot amplitude at each DOF (e.g., norm of displacement)
# rpm = w_vec * 60.0 / (2*np.pi)
# amplitude = np.abs(resp)


# dof_x = np.arange(0,n_dof,4)
# plt.figure()
# plt.plot(rpm, amplitude[:, dof_x]*1e6, linewidth=1.8)

# # --- Helper: build x/y DOF indices from node indices ---

# def dof_xy_from_nodes(nodes_list):
#     nodes_arr = np.array(list(nodes_list), dtype=int)
#     dof_x = 4*nodes_arr
#     dof_y = 4*nodes_arr + 2
#     return dof_x, dof_y

# # Collect node sets
# brg_nodes_ = [b.node for b in brgs]                      # assumed 0-based (consistent with assembly)
# seal_nodes_ = [s.node - 1 for s in seals]                # seals used 1-based earlier → convert to 0-based
# add_nodes_ = []
# try:
#     add_nodes_ = [e.node for e in added_elements if hasattr(e, 'node')]
# except Exception:
#     pass

# # Convert to DOF indices and clip to valid range
# brg_x, brg_y = dof_xy_from_nodes(brg_nodes_)
# seal_x, seal_y = dof_xy_from_nodes(seal_nodes_)
# add_x,  add_y = dof_xy_from_nodes(add_nodes_)

# valid = np.arange(amplitude.shape[1])
# brg_x = [d for d in brg_x if d in valid]
# brg_y = [d for d in brg_y if d in valid]
# seal_x = [d for d in seal_x if d in valid]
# seal_y = [d for d in seal_y if d in valid]
# add_x  = [d for d in add_x  if d in valid]
# add_y  = [d for d in add_y  if d in valid]

# # --- Plot categorized forced response (μm) ---

# plt.figure()

# # Bearings
# for dof in brg_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'BRG x@{dof}', linewidth=1.8)
# for dof in brg_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'BRG y@{dof}', linestyle='--', linewidth=1.8)
# # Seals
# for dof in seal_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'SEAL x@{dof}', linewidth=1.2)
# for dof in seal_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'SEAL y@{dof}', linestyle='--', linewidth=1.2)
# # Added mass
# for dof in add_x:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'ADD x@{dof}', linewidth=1.2)
# for dof in add_y:
#     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'ADD y@{dof}', linestyle='--', linewidth=1.2)


# plt.xlabel('Shaft speed (RPM)')
# plt.ylabel('Amplitude [um]')
# plt.title('Forced Response')
# # plt.legend()
# plt.grid(True)
# plt.show()
    
# # for dof in range(amplitude.shape[1]):
# #     plt.plot(rpm, amplitude[:, dof]*1e6, label=f'DOF {dof}')
