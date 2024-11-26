"""
Description:
    Test the updated implementation 

Date:
    11/25/2024

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import krak_routines as kr


def test1():
    z_list = [np.array([0.0, 5000.0])]
    cp_list = [np.array([1500.0, 1500.0])]
    cs_list = [np.array([0.0, 0.0])]
    attnp_list = [np.array([0.0, 0.0])]
    attns_list = [np.array([0.0, 0.0])]
    rho_list = [np.array([1.0, 1.0])]
    cp_top = None
    cs_top = None
    rho_top = 0.0
    attnp_top = 0.0
    attns_top = 0.0

    cp_bott = 2000.0
    cs_bott = 0.0
    rho_bott = 2.0
    attnp_bott = 0.0
    attns_bott = 0.0

    attn_units = 'dbplam'
    Ng_list = [1001]
    rmax = 0.0
    freq = 10.0
    c_low = 1400.0
    c_high = 2000.0

    krs, z, phi = kr.list_input_solve(freq, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_top, cs_top, rho_top, attnp_top, attns_top, cp_bott, cs_bott, rho_bott, attnp_bott, attns_bott, attn_units, Ng_list, rmax, c_low, c_high)

    for i in range(krs.size):
        print('i, krs[i]', i+1, krs[i])
        plt.figure()
        plt.plot(phi[:,i], z)
        plt.gca().invert_yaxis()
        plt.show()

def test2():
    z_list = [np.array([-10.0, 0.0]), np.array([0.0, 100.0])]
    cp_list = [np.array([3500.0, 3500.0]), np.array([1482.0, 1482.0])]
    cs_list = [np.array([1800.0, 1800.0]), np.array([0.0, 0.0])]
    attnp_list = [np.array([0.3, 0.3]), np.array([0.0, 0.0])]
    attns_list = [np.array([1.0, 1.0]), np.array([0.0, 0.0])]
    rho_list = [np.array([0.9, 0.9]), np.array([1.0, 1.0])]
    cp_top = None
    cs_top = None
    rho_top = 0.0
    attnp_top = 0.0
    attns_top = 0.0

    cp_bott = 2000.0
    cs_bott = 1000.0
    rho_bott = 2.2
    attnp_bott = 0.76
    attns_bott = 1.05

    attn_units = 'dbplam'
    Ng_list = [1000]
    rmax = 0.0
    freq = 10.0
    c_low = 1400.0
    c_high = 2000.0

    krs = kr.list_input_solve(freq, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_top, cs_top, rho_top, attnp_top, attns_top, cp_bott, cs_bott, rho_bott, attnp_bott, attns_bott, attn_units, Ng_list, rmax, c_low, c_high)

    for i in range(krs.size):
        print('i, krs[i]', i+1, krs[i])


test1()
