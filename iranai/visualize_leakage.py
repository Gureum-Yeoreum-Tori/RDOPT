import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py

data_dir = 'dataset/data/tapered_seal'
mat_files = ('20250826_T_091719','20250826_T_093534','20250826_T_095326',)

for mat_file in mat_files:
    mat_path = os.path.join(data_dir, mat_file, 'dataset.mat')

    print('current file: '+mat_file)
    
    base_dir = 'net'
    os.makedirs(base_dir, exist_ok=True)
    model_save_path = os.path.join(base_dir, 'mlp_leak_best_'+mat_file+'.pth')
    network_path = os.path.join(base_dir, 'mlp_leak_'+mat_file+'.pth')
    network_path_ts = os.path.join(base_dir, 'mlp_leak_'+mat_file+'.pt')

    # 데이터 로딩 및 전처리
    with h5py.File(mat_path, 'r') as mat:
        # input_: [nPara, nData] 형상 파라미터
        input_ = np.array(mat.get('input'))
        idx = input_[2,:] == 1.0
        leak = np.array(mat.get('Leak'))
        Leak = leak[6,:].reshape(-1,1)
        # Leak = np.average(leak,0).reshape(-1,1)
        n_data = Leak.shape[0]
        print(np.min(Leak))
        
    # 예시 데이터 (x, y: 1D 벡터, f: 2D)
    x = input_[0,idx]
    y = input_[1,idx]
    
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    
    X, Y = np.meshgrid(x_unique, y_unique)
    
    F = np.full_like(X, np.nan, dtype=float)

    # 각 (x,y)에 맞는 Leak 채우기
    for xi, yi, li in zip(x, y, Leak):
        ix = np.where(x_unique == xi)[0][0]
        iy = np.where(y_unique == yi)[0][0]
        F[iy, ix] = li

    # surf 그리기
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, F, cmap='viridis')

    fig.colorbar(surf)
    plt.show()