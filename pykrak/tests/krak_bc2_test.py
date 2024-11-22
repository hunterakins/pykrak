"""
Description:
    Test shooting up

Date:
    2024/11/21

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import mesh_routines as mr
from pykrak import attn_pert as ap
import sys

z_arr = np.concatenate((np.linspace(0.0, 10.0, 101), np.linspace(10.0, 15.0, 501)))
ind_arr = np.array([0, 101])
h_arr = np.array([z_arr[ind_arr[x] + 1] - z_arr[ind_arr[x]] for x in range(2)])
print(h_arr)
freq = 1500.0
omega2 = (2*np.pi*freq)**2
cp_arr = np.concatenate((1500.0 * np.ones(101), 1600.0 * np.ones(501)))
cs_arr = np.concatenate((0.0 * np.ones(101), 300.0 * np.ones(501)))
rho_arr = np.concatenate((1.0 * np.ones(101), 1.5 * np.ones(501)))
alpha_arr = np.concatenate((0.0* np.ones(101), 0.2 * np.ones(501)))
alphas_arr = np.concatenate((0.0* np.ones(101), 1.0 * np.ones(501)))
cp_imag_arr = ap.get_c_imag(cp_arr, alpha_arr, 'dbplam', 2*np.pi*freq)
cs_imag_arr = ap.get_c_imag(cs_arr, alphas_arr, 'dbplam', 2*np.pi*freq)

cp_arr = cp_arr + 1j*cp_imag_arr
cs_arr = cs_arr + 1j*cs_imag_arr

c_low = 1500.0
c_high = 1800.0

rho_top = 0.0
cp_top = 300.0
cs_top = 0.0
cp_top_imag = 0.0
cs_bot_imag = 0.0
cp_top = cp_top + 1j*cp_top_imag
cs_top = cs_top + 1j*cs_bot_imag


cp_bott = -1 # doesnt end up mattering for rigid bottom
cs_bott = -1
rho_bott = 1e10 # rigid bottom
attn_bott = -1
attn_units = 'dbplam'
#cp_bott_imag = ap.get_c_imag(cp_bott, attn_bott, attn_units, 2*np.pi*freq)
#cs_bot_imag = ap.get_c_imag(cs_bott, attn_bott, attn_units, 2*np.pi*freq)
#cp_bott = cp_bott + 1j*cp_bott_imag
#cs_bott = cs_bott + 1j*cs_bot_imag

b1, b1c, b2, b3, b4, rho_arr, c_low, c_high, elastic_flag, first_acoustic, last_acoustic =mr.initialize(h_arr, ind_arr, z_arr, omega2, cp_arr, cs_arr, rho_arr, cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, c_low, c_high)

x = 45.320065860122355

fbott, gbott, iPower = mr.get_bc_impedance(x, omega2, False, cp_bott, cs_bott, rho_bott, 
                                h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                first_acoustic, last_acoustic)

CountModes = True
modeCount = 0
f,g, iPower = mr.acoustic_layers(x, fbott, gbott, iPower, ind_arr, h_arr, z_arr, b1, rho_arr, CountModes, modeCount, first_acoustic, last_acoustic)
print('f, g', f, g)
sys.exit()
ftop, gtop = mr.get_bc_impedance(x, omega2, True, cp_top, cs_top, rho_top, 
                                h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                first_acoustic, last_acoustic)
print('ftop, gtop')
print(ftop, gtop)
print('fbott, gbott')
print(fbott, gbott)

print('c_low = ', c_low)
print('c_high = ', c_high)
for i in range(b1.size):
    print('i b1 b1c', i, b1[i], b1c[i])

