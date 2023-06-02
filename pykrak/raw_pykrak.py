"""
Description:
    Raw forward model for PyNM. Useful for plugging into optimization codes
    that you are planning on compiling with numba.

Date:
    5/17/2024


Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from pykrak.attn_pert import add_arr_attn
from pykrak.inverse_iteration import get_phi
from pykrak.sturm_seq import get_comp_krs
from numba import njit

@njit
def get_modes(freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax):
    omega = 2*np.pi*freq
    kr_min = omega / cmax
    kr_max = omega / cmin

    M = 1e10 # number of modes

    h0 = h_arr[0]

    # get wavenumbers for this mesh
    lam_min = np.square(h0*kr_min) 
    lam_max = np.square(h0*kr_max) 
    krs = get_comp_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr,\
                 c_hs, rho_hs, lam_min, lam_max)

    M = min(M,krs.size) # keep track of number of modes

    phi = get_phi(krs, omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs)

    """ Factor in attenuation with perturbation theory"""
    if np.sum(attn_arr) > 0 or attn_hs: # there is attenuation in the model
        comp_krs = add_arr_attn(omega, krs, phi, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs,rho_hs, attn_hs)
    else:
        comp_krs = krs + 1j*np.zeros(krs.size)
    
    phi_z = get_phi_z(ind_arr, z_arr)
        
    return comp_krs, phi, phi_z

@njit
def get_phi_z(ind_arr, z_arr):
    """
    Remove doubled layer values
    """
    N = len(ind_arr)
    for i in range(N):
        if i < N-1:
            phi_z_i = z_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            phi_z_i = z_arr[ind_arr[i]:]
        if i == 0:
            phi_z = phi_z_i
        else:
            phi_z = np.concatenate((phi_z, phi_z_i[1:])) # elimiinae doubled layer values
    return phi_z

