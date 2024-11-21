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
from pykrak import attn_pert as ap

h_arr = np.array([0.1]) # one acoustic layer
ind_arr = np.array([0], dtype=int)
z_arr = np.linspace(0.0, 10.0, 101)
freq = 1500.0
omega2 = (2*np.pi*freq)**2
cp_arr = 1500.0 * np.ones_like(z_arr)
cs_arr = 0.0 * np.ones_like(z_arr)
rho_arr = 1.0 * np.ones_like(z_arr)
alpha_arr = np.zeros_like(z_arr)
alphas_arr = np.zeros_like(z_arr)

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
attn_bott = 0.0
attn_units = 'dbplam'
cp_bott_imag = ap.get_c_imag(cp_bott, attn_bott, attn_units, 2*np.pi*freq)
cs_bot_imag = ap.get_c_imag(cs_bott, attn_bott, attn_units, 2*np.pi*freq)
cp_bott = cp_bott + 1j*cp_bott_imag
cs_bott = cs_bott + 1j*cs_bot_imag

b1, b1c, b2, b3, b4, rho_arr, c_low, c_high, elastic_flag=mr.initialize(h_arr, ind_arr, z_arr, omega2, cp_arr, cs_arr, rho_arr, cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, c_low, c_high)

print('c_low = ', c_low)
print('c_high = ', c_high)
print(b1, b1c)

