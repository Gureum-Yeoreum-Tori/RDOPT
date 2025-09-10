#%%
import os
import numpy as np
from import_data import rotor_import, plot_rotor_2d

import matplotlib.pyplot as plt


bs_params = {
        'mu_brg': 0.01, # Pa s, bearing fluid 
        'mu_seal': 1.4e-3, # Pa s, seal fluid 
        'rho_seal': 850, # kg/m^3, seal fluid 
    }

## Import and generate rotor data
data_dir = 'dataset/data'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

n_ele, n_node, n_dof, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_K_r, mat_C_g, mat_M_r, mat_M_a, F_mass, F_ex, unb, brgs, seals, support_dofs = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

#%%


# def plot_rotor_2d(rotor_elements, added_elements=None, brgs=None, seals=None,
#                 ax=None, show=True, draw_inner=True,
#                 colors=None, lw=1.5):
#     """
#     2D profile plot of the rotor along the axial direction.

#     - rotor_elements: list of Element with fields (node, L, Od, Id)
#     - brgs: list of Bearing (optional). Drawn as vertical markers with labels "B".
#     - seals: list of Seal (optional). Drawn as vertical markers with labels "S".
#     - draw_inner: if True, also draw inner diameter (dashed lines)
#     - annotate_nodes: if True, annotate node indices at the top of the axis

#     The plot shows stepwise outer/inner radii mirrored about the centerline (y=0).
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Rectangle, Circle

#     if colors is None:
#         colors = {
#             'outer': 'tab:red',
#             'inner': '0.35',
#             'added': 'tab:purple',
#             'spring_brg': 'tab:red',
#             'damper_brg': 'tab:green',
#             'spring_seal': 'tab:orange',
#             'damper_seal': 'tab:blue',
#             'center': 'k',
#         }

#     # Axial node coordinates
#     vec_L = np.array([e.L for e in rotor_elements], dtype=float)
#     x_nodes = np.concatenate([[0.0], np.cumsum(vec_L)])
#     # Create axes if needed
#     created_ax = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(14, 4))
#         created_ax = True
        
#     # Draw centerline
#     ax.axhline(0.0, color=colors['center'], lw=1.0, zorder=1, linestyle='--')

#     # Draw each element as a rectangle (outer) and optional inner (dashed)
#     rmax = 0.0
#     for i, e in enumerate(rotor_elements):
#         x0, L = x_nodes[i], float(e.L)
#         ro = 0.5 * float(e.Od)
#         ri = 0.5 * float(e.Id)
#         rmax = max(rmax, ro)

#         if ro > 0 and L > 0:
#             rect = Rectangle((x0, -ro), width=L, height=2*ro,
#                             fill=False, edgecolor=colors['outer'], lw=lw)
#             ax.add_patch(rect)

#         if draw_inner and ri > 0 and L > 0:
#             rect_i = Rectangle((x0, -ri), width=L, height=2*ri,
#                             fill=False, edgecolor=colors['inner'],
#                             lw=lw*0.9, linestyle='--')
#             ax.add_patch(rect_i)
    
#     def draw_disk(ax, x0, h, r, label=None):
#         arm_x = [x0, x0]
#         arm_y = [-h/2, h/2]
#         ax.plot(arm_x, arm_y, color="#0D7441", lw=1.5*lw, label=label)
        
#         circ = Circle((x0,h/2+r),r , color="#0D7441", fill=False, lw=1.5*lw)
#         ax.add_patch(circ)
#         circ = Circle((x0,-h/2-r),r , color="#0D7441", fill=False, lw=1.5*lw)
#         ax.add_patch(circ)
#         # rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
#         #                 fill=True, facecolor='#B8BEBB', edgecolor="#737976", lw=lw)
#         # ax.add_patch(rect)
#         # xx = [x0-w/2, x0+w/2]
#         # yx = [-h/2, h/2]
        
#         # xx = [x0-w/2, x0+w/2]
#         # yx = [h/2, -h/2]
#         # ax.plot(xx, yx, color='#737976', lw=0.9*lw)
#     # Added elements overlay (if any)
#     r_disk = 0.025
#     if added_elements:
#         added_D_max = 0.0
#         for a in added_elements: 
#             added_D_max = max(added_D_max,a.Od)
#         added_D_max *= 1.2
#         for idx, a in enumerate(added_elements): 
#             x0 = x_nodes[int(a.node)]
#             added_D_max = max(added_D_max,a.Od)
#             label = None
#             if idx == 0:
#                 label = "added"
#             draw_disk(ax, x0=x0, h=added_D_max, r=r_disk, label=label)
#     rmax = max(rmax, added_D_max/2.0)
#     # if added_elements:
#     #     for a in added_elements:
#     #         x0 = x_nodes[int(a.node)]
#     #         L = float(a.L)
#     #         ro = 0.5 * float(a.Od)
#     #         ri = 0.5 * float(a.Id)
#     #         rmax = max(rmax, ro)
#     #         draw_disk(ax, x0=x0, w=L, h=added_D_max)
#     #         # if ro > 0 and L > 0:
#     #         #     rect = Rectangle((x0-L/2, -ro), width=L, height=2*ro,
#     #         #                     fill=False, edgecolor=colors['added'], lw=lw)
#     #         #     ax.add_patch(rect)
#     #         # if draw_inner and ri > 0 and L > 0:
#     #         #     rect_i = Rectangle((x0-L/2, -ri), width=L, height=2*ri,
#     #         #                     fill=False, edgecolor=colors['added'],
#     #         #                     lw=lw*0.9, linestyle=':')
#     #         #     ax.add_patch(rect_i)

#     # Compute node-wise attach radius for supports
#     n_ele = len(rotor_elements)
#     ro_nodes = np.zeros_like(x_nodes)
#     for i in range(n_ele + 1):
#         left_ro = 0.5 * rotor_elements[i-1].Od if i > 0 else 0.5 * rotor_elements[0].Od
#         right_ro = 0.5 * rotor_elements[i].Od if i < n_ele else 0.5 * rotor_elements[-1].Od
#         ro_nodes[i] = max(left_ro, right_ro)
#     if added_elements:
#         for a in added_elements:
#             ro_nodes[int(a.node)] = max(ro_nodes[int(a.node)], 0.5*float(a.Od))

            
#     def draw_xbox(ax, x0, w, h, label=None):
#         rect = Rectangle((x0-w/2, -h/2), width=w, height=h,
#                         fill=True, facecolor='#B8BEBB', edgecolor="#737976", lw=lw, label=label)
#         ax.add_patch(rect)
#         xx = [x0-w/2, x0+w/2]
#         yx = [-h/2, h/2]
#         ax.plot(xx, yx, color="#737976", lw=0.9*lw)
#         xx = [x0-w/2, x0+w/2]
#         yx = [h/2, -h/2]
#         ax.plot(xx, yx, color='#737976', lw=0.9*lw)
        
#     def draw_seal(ax, x0, w, ro, h, c, label=None):
        
#         # rect = Rectangle((x0-w/2, -ro-c), width=w, height=h,
#         #                 fill=True, facecolor="#4F60BE", edgecolor="#4F60BE", lw=lw, label=label)
#         rect = Rectangle((x0-w/2, -ro-c), width=w, height=h,
#                         fill=True, color="#4F60BE", lw=lw, label=label)
#         ax.add_patch(rect)
#         rect = Rectangle((x0-w/2, ro-h+c), width=w, height=h,
#                         fill=True, facecolor="#4F60BE", edgecolor="#4F60BE", lw=lw)
#         ax.add_patch(rect)
        
        
#         # xx = [x0-w/2, x0+w/2]
#         # yx = [-h/2, h/2]
#         # ax.plot(xx, yx, color="#737976", lw=0.9*lw)
#         # xx = [x0-w/2, x0+w/2]
#         # yx = [h/2, -h/2]
#         # ax.plot(xx, yx, color='#737976', lw=0.9*lw)
        
#     # Helpers for support drawing
#     # def draw_spring(ax, x, y_top, y_bottom, width, color):
#     #     turns = 6
#     #     ys = np.linspace(y_top, y_bottom, 2*turns + 1)
#     #     xs = np.full_like(ys, x)
#     #     for k in range(1, len(ys)-1):
#     #         xs[k] = x + (width/2 if k % 2 else -width/2)
#     #     ax.plot(xs, ys, color=color, lw=lw)

#     # def draw_damper(ax, x, y_top, y_bottom, width, color):
#     #     h = y_top - y_bottom
#     #     pad = 0.18 * h
#     #     body_top = y_top - pad
#     #     body_bot = y_bottom + pad
#     #     ax.plot([x, x], [y_top, body_top], color=color, lw=lw)
#     #     ax.add_patch(Rectangle((x - width/2, body_bot), width, body_top - body_bot,
#     #                         fill=False, edgecolor=color, lw=lw))
#     #     ax.plot([x, x], [body_bot, body_bot - 0.15*h], color=color, lw=lw)

#     # def draw_ground(ax, x, y, width, color='0.2'):
#     #     ax.plot([x - width, x + width], [y, y], color=color, lw=lw)

#     # span_x = x_nodes.max() - x_nodes.min() if len(x_nodes) > 1 else 1.0
#     # x_off = 0.012 * span_x
#     # w_sym = 0.02 * span_x
#     # h_sup = max(0.7*rmax, 0.05)
#     # y_ground = -(rmax + 0.06*rmax + h_sup)

#     # def _draw_support(x, y_attach, spring_color, damper_color, label=None):
#     #     ax.plot([x, x], [y_attach, y_attach - 0.04*h_sup], color='0.4', lw=lw)
#     #     y0 = y_attach - 0.04*h_sup
#     #     draw_spring(ax, x - x_off, y0, y_ground + 0.02*h_sup, w_sym, spring_color)
#     #     draw_damper(ax, x + x_off, y0, y_ground + 0.02*h_sup, w_sym*0.9, damper_color)
#     #     draw_ground(ax, x, y_ground, 1.5*w_sym)
#     #     if label:
#     #         ax.text(x, y_ground - 0.06*h_sup, label, ha='center', va='top', fontsize=8)

#     if brgs:
#         for idx, b in enumerate(brgs):
#             j = int(b.node)
#             xb = x_nodes[j]
#             label = None
#             if idx == 0:
#                 label = "bearing"
#             draw_xbox(ax, x0=xb, w=0.1, h=ro_nodes[j]*2.4, label=label)
#             # y_attach = -ro_nodes[j]
#             # _draw_support(xb, y_attach, colors['spring_brg'], colors['damper_brg'], label='B')

#     if seals:
#         for idx, s in enumerate(seals):
#             j = int(s.node)
#             xs = x_nodes[j]
#             c = 0.015
#             label = None
#             if idx == 0:
#                 label = "seal"
#             draw_seal(ax, x0=xs, w=max(s.Ls,0.045), ro=s.Ds/2, h=0.04, c=c, label=label)
#             # draw_xbox(ax, x0=xs, w=max(s.Ls,0.025), h=s.Ds)
#             # y_attach = -ro_nodes[j]
#             # _draw_support(xs, y_attach, colors['spring_seal'], colors['damper_seal'], label='S')
            
    
#     for i in np.arange(0,x_nodes.shape[0],4):
#         x = x_nodes[i]
#         # ax.plot(x, 0, 'ko')
#         # ax.text(x, -1.5 * rmax, f"#{i}", ha='center', va='bottom', fontsize=12, color='0.3')
#         # circ = Circle((x,0), 0.012 , edgecolor="#000000", fill=True)
#         circ = Circle((x,0), 0.012 , color="#000000", fill=True)
#         ax.add_patch(circ)
#         # circ = Circle((x,0), 0.015 , edgecolor=None, facecolor="#000000", alpha=0.2, fill=True, lw=1.5*lw)
#         # ax.add_patch(circ)
#         ax.text(x, -rmax -6*r_disk, f"#{i}", ha='center', va='bottom', fontsize=10)

#     # # Node annotations (optional)
#     # if annotate_nodes:
#     #     for i, x in enumerate(x_nodes):
#     #         ax.text(x, 0.98 * rmax, f"#{i}", ha='center', va='bottom', fontsize=8, color='0.3')

#     # Formatting
#     xmin, xmax = x_nodes.min(), x_nodes.max()
#     ax.set_xlim(xmin - 0.02*(xmax-xmin), xmax + 0.02*(xmax-xmin))
#     # y_min = min(-1.15*rmax, y_ground - 0.15*h_sup)
#     # ax.set_ylim(-rmax -6*r_disk, np.inf)
#     y_lim = ax.get_ylim()

#     ax.set_ylim(-rmax -8*r_disk, y_lim[1]+r_disk)
#     ax.set_xlabel('Axial location (m)')
#     ax.set_ylabel('Shaft radius (m)')
#     # ax.set_title('Rotor model')
#     ax.set_aspect('equal')
    
#     ax.legend()
#     ax.xaxis.grid(True)

#     if created_ax and show:
#         plt.tight_layout()
#         plt.savefig("shaft.png", dpi=300, bbox_inches="tight")
#         plt.show()
#     return ax



# # plot_rotor_3d(rotor_elements=rotor_elements)
# # plot_rotor_2d(rotor_elements=rotor_elements, added_elements=added_elements, brgs=brgs, seals=seals, annotate_nodes=True)
plot_rotor_2d(rotor_elements=rotor_elements, added_elements=added_elements, brgs=brgs, seals=seals, fontsize=13, save_img=True)


# %%
