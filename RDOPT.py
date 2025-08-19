import os
import numpy as np
from ng_import_data import rotor_import, calculate_bearing_loads

## Import data
data_dir = 'dataset'
rotor_file = os.path.join(data_dir, "input_Optim_Rotor.xlsx")
rotor_sheet = "RDOPT"

w_range = np.array([1500, 5000]) * np.pi / 30
w_oper = 3500 * np.pi / 30
oper = {
        'wMin': w_range[0],
        'wMax': w_range[1],
        'range': w_oper,
    }
bs_params = {
        'muBrg': 0.04, # Pa s, bearing fluid 
        'muSeal': 1.4e-3, # Pa s, seal fluid 
        'rhoSeal': 850, # kg/m^3, seal fluid 
    }

n_ele, n_node, n_add, n_brg, n_seal, rotor_elements, rotor_nodal_props, added_elements, added_props, mat_M, mat_Ks, mat_Cg, mat_MA, F_mass, F_ex, unb, brgs, seals = rotor_import(file_path=rotor_file,sheet_name=rotor_sheet,bs_params=bs_params)

normalLoad = calculate_bearing_loads(
    rotor_elements=rotor_elements,
    brgs=brgs,
    F_mass=F_mass,
    F_ex=F_ex,
    n_brg=n_brg,
)

## optimization condition

paretoRatio = 0.4;
plottt = {'gaplotpareto','gaplotstopping','gaplotrankhist','gaplotspread'};

nVarBrg = 2*nBrg; % [Brg#1 Cr1/Db*10000 Brg#2 Cr2/Db*10000]
% intconBrg = 1:2*nBrg;
lbBrg = [1 1 5 5]; 
ubBrg = [55 55 20 20];

nVarSeal = 3*nSeal; % [hIn* hOut*1000 psr*10]
% intconSeal = 1:3*nSeal;
lbSeal = [20 20 0]; 
ubSeal = [50 50 10];