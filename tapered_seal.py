#%%
import numpy as np
import itertools
import time
import datetime
from tqdm import tqdm
from script.seal_solver import main_seal_solver

wRange = np.array([1500, 5000]) * np.pi / 30
wOper = np.array([3800, 4200]) * np.pi / 30
w_oper = 3500 * np.pi / 30 # I renamed Oper.w to w_oper to be more "Pythonic"
w_vec = np.linspace(wRange[0],wRange[1]*1.25,14)
# w_vec = w_oper

Ds = 0.22
Ls = 0.02
NxSeal = 15
mu = 1.4e-3
rho = 850
dp = 850000 # Seal 1

# Ds = 0.23
# Ls = 0.15
# NxSeal = 15
# mu = 1.4e-3
# rho = 850
# dp = 16000000 # 20250812_T_113003, Seal 3

lb = [20, 20, 0]
ub = [50, 50, 10]
vhIn = np.linspace(lb[0], ub[0], 15)
vhOut = np.linspace(lb[1], ub[1], 15)
vPsr = np.linspace(lb[2], ub[2], 6)
inputVec = np.array(list(itertools.product(vhIn, vhOut, vPsr)))
# print(f"Data size (shape): {inputVec.shape}")

nV = inputVec.shape[0] # Get the number of combinations
print(f"Data size: {nV}")

# main loop

fluid = {
    'mu': mu,
    'rho': rho
}

tStart = time.perf_counter()

for cV in tqdm(range(nV), desc="Processing Seal Data"):
# for cV in tqdm(range(5), desc="Processing Seal Data"):
    # Get the current row of input parameters
    inSeal_cur = inputVec[cV, :]

    # Perform calculations as in your MATLAB loop
    hIn = Ds * inSeal_cur[0] / 10000
    hOut = Ds * inSeal_cur[1] / 10000
    psr = inSeal_cur[2] / 10
    
    # 형상(Geometry) 관련 파라미터
    geometry = {
        'hIn': hIn,
        'hOut': hOut,
        'Ds': Ds,
        'Ls': Ls,
        'NxSeal': NxSeal
    }
    
    op_conditions = {
        'dp': dp,
        'psr': psr,
        'w_vec': w_vec
    }

    Leak, RDC, P, U, V, x, h = main_seal_solver(geometry, fluid, op_conditions)
    print(f"Case {cV}, Leak = {Leak[0]:.2f}")
    # print(RDC)


elapsed = time.perf_counter() - tStart
elapsed_str = str(datetime.timedelta(seconds=elapsed))

print(f"\nTotal elapsed time: {elapsed:.4f} seconds")
print(f"Formatted time: {elapsed_str}")


# %%
