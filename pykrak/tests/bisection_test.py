"""
Description:
    Test solve2
    Multiple acoustic layers

Date:
    2024/11/21

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import krak_routines as kr
from pykrak import attn_pert as ap
import sys


N_list_base = [1000]
Nv = np.array([1])#, 2, 4, 8, 16])
Nset = len(Nv)
M_max = 5000
freq = 10.0
omega2 = (2*np.pi*freq)**2
print('omega2', omega2)
ev_mat = np.zeros((Nset, M_max))
extrap = np.zeros((Nset, M_max))


c_low = 1400.0
c_high = 2000.0

attn_units = 'dbplam'
rho_top = 0.0
cp_top = 300.0 #doesn't matter because rho=0 is vaccuum
cs_top = 0.0
cp_top_imag = 0.0
cs_top_imag = 0.0
cp_top = cp_top + 1j*cp_top_imag
cs_top = cs_top + 1j*cs_top_imag


cp_bott = 2000.0 # 
cs_bott = 0.0
rho_bott = 1.0
attn_bott = 0.0
cp_bott_imag = ap.get_c_imag(cp_bott, attn_bott, attn_units, 2*np.pi*freq)
cs_bot_imag = ap.get_c_imag(cs_bott, attn_bott, attn_units, 2*np.pi*freq)
cp_bott = cp_bott + 1j*cp_bott_imag
cs_bott = cs_bott + 1j*cs_bot_imag

for iset, fact in enumerate(Nv):
    N_list = [x*fact for x in N_list_base]
    z_arr = np.linspace(0.0, 5000.0, N_list[0])
    N_arr = np.array(N_list)
    ind_arr = np.array([0])
    h_arr = np.array([z_arr[ind_arr[x] + 1] - z_arr[ind_arr[x]] for x in range(len(N_list))])
    cp_arr = 1500.0 * np.ones(N_list[0])
    cs_arr = 0.0 * np.ones(N_list[0])
    rho_arr = 1.0 * np.ones(N_list[0])
    alpha_arr = 0.0* np.ones(N_list[0])
    alphas_arr = 0.0* np.ones(N_list[0])
    cp_imag_arr = ap.get_c_imag(cp_arr, alpha_arr, 'dbplam', 2*np.pi*freq)
    cs_imag_arr = ap.get_c_imag(cs_arr, alphas_arr, 'dbplam', 2*np.pi*freq)

    cp_arr = cp_arr + 1j*cp_imag_arr
    cs_arr = cs_arr + 1j*cs_imag_arr

    b1, b1c, b2, b3, b4, rho_arr, c_low, c_high, elastic_flag, first_acoustic, last_acoustic =kr.initialize(h_arr, ind_arr, z_arr, omega2, cp_arr, cs_arr, rho_arr, cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, c_low, c_high)
    #for i in range(b1.size):
    #    print('i, z, b1', i , z_arr[i], b1[i])


    args = (omega2, ev_mat, iset,
             h_arr, ind_arr, z_arr, N_arr, 
             cp_top, cs_top, rho_top, 
             cp_bott, cs_bott, rho_bott, 
             b1, b1c, b2, b3, b4, rho_arr, 
             c_low, c_high, elastic_flag, first_acoustic, last_acoustic)

    if iset == 0:
        h_v = np.array([h_arr[0]])
    else:
        h_v = np.append(h_v, h_arr[0])

    ev_mat, M = kr.solve1(args, h_v)
    print('iset, M', iset, M)

    """
    Do Richardson extrapolation on ev_mat
    """
    extrap[iset,:M] = ev_mat[iset,:].copy()
    KEY   = int(2 * M / 3)   # index of element used to check convergence
    if iset > 0:
        T1 = extrap[0, KEY]
        for j in range(iset-1, -1, -1):
            for m in range(M):
                x1 = Nv[j]**2
                x2 = Nv[iset]**2
                F1 = extrap[j,m]
                F2 = extrap[j+1,m]
                extrap[j, m] = F2 - (F1 - F2) / (x2 / x1 - 1.0)

        T2 = extrap[0, KEY]
        error = np.abs(T2 - T1)
        print('Error', error)
for i in range(M):
    print('i: {} kr: {}'.format(i+1, np.sqrt(extrap[0,i])))


plt.show()
