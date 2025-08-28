import h5py
import numpy as np
from scipy.interpolate import PPoly

h5_file = "dataset/data/nondBrg.h5"

with h5py.File(h5_file, "r") as f:
    # 상위 그룹들 (brg_1, brg_2, ...)
    print("Top groups:", list(f.keys()))

    # 특정 베어링 (예: brg_1)
    brg1 = f["brg_1"]
    print("brg_1 fields:", list(brg1.keys()))

    # 스칼라 데이터 (예: LD, Mp)
    LD = np.array(brg1["LD"])
    Mp = np.array(brg1["Mp"])
    print("LD =", LD, "Mp =", Mp)

    # 다항 보간 함수 데이터 (예: fCoF)
    fCoF = brg1["fCoF"]
    breaks = np.array(fCoF["breaks"]).squeeze()
    coefs  = np.array(fCoF["coefs"])

    # SciPy PPoly로 복원
    pp = PPoly(c=coefs.T, x=breaks)  # MATLAB에서 저장할 때 shape 맞춰놨으면 그대로
    # (만약 차원이 반대면 coefs.T 로 바꾸면 됨)

    # 테스트 평가
    x_test = np.linspace(breaks[0], breaks[-1], 5)
    y_test = pp(x_test)
    print("fCoF(x) at sample points:", y_test)