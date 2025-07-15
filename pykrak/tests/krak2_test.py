"""
Description:
    Testing the improved match to Porter's code

Date:
    2024/11/21

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import mesh_routines as mr
from pykrak import krak_routines as kr
from pykrak import attn_pert as ap
import sys

h_arr = np.array([0.1, 0.1, 0.1]) # one elastic and two acoustic layers
z_arr = np.concatenate((np.linspace(-1.0, 0.0, 11), np.linspace(0.0, 10.0, 101), np.linspace(10.0, 11.0, 11)))
ind_arr = np.array([0, 11, 112])
freq = 1500.0
omega2 = (2*np.pi*freq)**2
cp_arr = np.concatenate((3500.0 * np.ones(11), 1500.0 * np.ones(101), 1600.0 * np.ones(11))) 
cs_arr = 0.0 * np.ones_like(z_arr)
cs_arr[:11] = 1800.0
rho_arr = np.concatenate((0.9*np.ones(11),1.0 * np.ones(101), 1.5 * np.ones(11)))
alpha_arr = np.concatenate((0.3* np.ones(11), 0.0 * np.ones(101), 0.2 * np.ones(11)))
alphas_arr = np.zeros_like(z_arr)
alphas_arr[:11] = 1.0
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


cp_bott = 1600.0
cs_bott = 0.0
rho_bott = 2.0
attn_bott = 0.2
attn_units = 'dbplam'
cp_bott_imag = ap.get_c_imag(cp_bott, attn_bott, attn_units, 2*np.pi*freq)
cs_bot_imag = ap.get_c_imag(cs_bott, attn_bott, attn_units, 2*np.pi*freq)
cp_bott = cp_bott + 1j*cp_bott_imag
cs_bott = cs_bott + 1j*cs_bot_imag

b1, b1c, b2, b3, b4, rho_arr, c_low, c_high, elastic_flag, first_acoustic, last_acoustic =kr.initialize(h_arr, ind_arr, z_arr, omega2, cp_arr, cs_arr, rho_arr, cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, c_low, c_high)
print('first_acoustic = ', first_acoustic)
print('last_acoustic = ', last_acoustic)

x = 45.320065857951519
mode_count = 0
complex_flag=True
ftop, gtop, _,_ = kr.get_bc_impedance(x, omega2, True, cp_top, cs_top, rho_top, 
                                h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                first_acoustic, last_acoustic,
                                mode_count, complex_flag)

mode_count = 0
complex_flag=True
fbott, gbott, _,_ = kr.get_bc_impedance(x, omega2, False, cp_bott, cs_bott, rho_bott, 
                                h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                first_acoustic, last_acoustic,
                                mode_count, complex_flag)
print('ftop, gtop')
print(ftop, gtop)
print('fbott, gbott')
print(fbott, gbott)

print('c_low = ', c_low)
print('c_high = ', c_high)
for i in range(b1.size):
    print('i b1 b1c', i, b1[i], b1c[i])

