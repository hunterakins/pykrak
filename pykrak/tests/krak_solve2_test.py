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


N_list_base = [801, 801, 4001]
Nset = 5
M_max = 5000
freq = 200.0
omega2 = (2 * np.pi * freq) ** 2
print("omega2", omega2)
ev_mat = np.zeros((Nset, M_max))
print(ev_mat.flags)
extrap = np.zeros((Nset, M_max))
Nv = np.array([1, 2, 4, 8, 16])
for iset, fact in enumerate(Nv):
    N_list = [x * fact for x in N_list_base]
    N_arr = np.array(N_list)
    z_arr = np.concatenate(
        (
            np.linspace(0.0, 10.0, N_list[0]),
            np.linspace(10.0, 20.0, N_list[1]),
            np.linspace(20.0, 25.0, N_list[2]),
        )
    )
    ind_arr = np.array([0, N_list[0], N_list[0] + N_list[1]])
    h_arr = np.array(
        [z_arr[ind_arr[x] + 1] - z_arr[ind_arr[x]] for x in range(len(N_list))]
    )
    print(h_arr, ind_arr)
    cp_arr = np.concatenate(
        (
            1500.0 * np.ones(N_list[0]),
            1550.0 * np.ones(N_list[1]),
            1600.0 * np.ones(N_list[2]),
        )
    )
    cs_arr = np.concatenate(
        (0.0 * np.ones(N_list[0]), 0.0 * np.ones(N_list[1]), 300.0 * np.ones(N_list[2]))
    )
    rho_arr = np.concatenate(
        (1.0 * np.ones(N_list[0]), 1.2 * np.ones(N_list[1]), 1.5 * np.ones(N_list[2]))
    )
    alpha_arr = np.concatenate(
        (0.0 * np.ones(N_list[0]), 0.0 * np.ones(N_list[1]), 0.0 * np.ones(N_list[2]))
    )
    alphas_arr = np.concatenate(
        (0.0 * np.ones(N_list[0]), 0.0 * np.ones(N_list[1]), 0.0 * np.ones(N_list[2]))
    )
    cp_imag_arr = ap.get_c_imag(cp_arr, alpha_arr, "dbplam", 2 * np.pi * freq)
    cs_imag_arr = ap.get_c_imag(cs_arr, alphas_arr, "dbplam", 2 * np.pi * freq)

    cp_arr = cp_arr + 1j * cp_imag_arr
    cs_arr = cs_arr + 1j * cs_imag_arr

    c_low = 1500.0
    c_high = 1800.0

    rho_top = 0.0
    cp_top = 300.0
    cs_top = 0.0
    cp_top_imag = 0.0
    cs_bot_imag = 0.0
    cp_top = cp_top + 1j * cp_top_imag
    cs_top = cs_top + 1j * cs_bot_imag

    cp_bott = 2600.0  # doesnt end up mattering for rigid bottom
    cs_bott = 2000.0
    rho_bott = 1e10  # rigid bottom
    attn_bott = 0.0
    attn_units = "dbplam"
    # cp_bott_imag = ap.get_c_imag(cp_bott, attn_bott, attn_units, 2*np.pi*freq)
    # cs_bot_imag = ap.get_c_imag(cs_bott, attn_bott, attn_units, 2*np.pi*freq)
    # cp_bott = cp_bott + 1j*cp_bott_imag
    # cs_bott = cs_bott + 1j*cs_bot_imag

    (
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
    ) = kr.initialize(
        h_arr,
        ind_arr,
        z_arr,
        omega2,
        cp_arr,
        cs_arr,
        rho_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        c_low,
        c_high,
    )
    print("b1 .size", b1.size)
    print("c_low", c_low, "c_high", c_high)
    # for i in range(b1.size):
    #    print('i, z, b1', i , z_arr[i], b1[i])

    print("iset", iset)
    args = (
        omega2,
        ev_mat,
        iset,
        h_arr,
        ind_arr,
        z_arr,
        N_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        c_high,
        elastic_flag,
        first_acoustic,
        last_acoustic,
    )

    if iset == 0:
        h_v = np.array([h_arr[0]])
        M = M_max
    else:
        h_v = np.append(h_v, h_arr[0])
        print("h_v", h_v)

    print("ev mat shap", ev_mat.shape)
    ev_mat, M = kr.solve2(args, h_v, M)
    print("ev mat shap", ev_mat.shape)
    print("M", M)
    print("iset", iset)
    ev_i = ev_mat[iset, :]
    print("ev_i", ev_i)
    Mi = np.sum(ev_i > 0)
    inds = np.arange(0, Mi)
    print("iter i", np.sqrt(ev_i[:Mi]))
    plt.plot(inds, ev_i[:Mi], "k.")

    """
    Do Richardson extrapolation on ev_mat
    """
    extrap[iset, :Mi] = ev_i[:Mi].copy()
    KEY = int(2 * Mi / 3)  # index of element used to check convergence
    if iset > 0:
        T1 = extrap[0, KEY]
        for j in range(iset - 1, -1, -1):
            for m in range(Mi):
                x1 = Nv[j] ** 2
                x2 = Nv[iset] ** 2
                F1 = extrap[j, m]
                F2 = extrap[j + 1, m]
                extrap[j, m] = F2 - (F1 - F2) / (x2 / x1 - 1.0)

        T2 = extrap[0, KEY]
        error = np.abs(T2 - T1)
        print("Error", error)


for i in range(Mi):
    print("i: {} kr: {}".format(i, np.sqrt(extrap[0, i])))


plt.show()
