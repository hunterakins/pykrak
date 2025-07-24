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
import numba as nb

#float_arr_type = nb.types.Array(nb.f8, 1, 'A', readonly=True)
#complex_arr_type = nb.types.Array(nb.c16, 1, 'A', readonly=True)
#int_arr_type = nb.types.Array(nb.i8, 1, 'A', readonly=True)
#tuple_type = nb.types.Tuple((complex_arr_type, float_arr_type, float_arr_type))

#@njit(float_arr_type(int_arr_type, float_arr_type), cache=True)
@njit(cache=True)
def get_phi_z(ind_arr, z_arr):
    """
    Remove doubled layer values
    """
    N = ind_arr.size
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


#@njit(tuple_type(nb.f8, float_arr_type, int_arr_type, float_arr_type, float_arr_type, float_arr_type, float_arr_type,nb.f8, nb.f8, nb.f8, nb.f8, nb.f8), cache=True)
@njit(cache=True)
def get_modes(freq, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, cmin, cmax):
    """
    Get modes for a given frequency associated with the mesh info passed in
    h_arr - np 1d array
        array of mesh size within each layer
    ind_arr - np 1d array of ints
        array of indices of where each layer starts
    z_arr - np 1d array
        depths of each layer mesh stacked together
    k_sq_arr - np 1d array
        wavenumber squared of each layer mesh stacked together
    rho_arr - np 1d array
        density of each layer mesh stacked together
    k_hs_sq - float
        wavenumber squared of halfspace
    rho_hs - float
        density of halfspace
    cmin - float
        minimum modal phase speed  (kr_max = omega / cmin)
    cmax - float
        maximum modal phase speed  (kr_min = omega / cmax)
    """

    omega = 2*np.pi*freq
    kr_min = omega / cmax
    kr_max = omega / cmin

    M = 1e10 # number of modes

    h0 = h_arr[0]

    # get wavenumbers for this mesh
    lam_min = np.square(h0*kr_min) 
    lam_max = np.square(h0*kr_max) 
    krs = get_comp_krs(h_arr, ind_arr, z_arr, k_sq_arr.real, rho_arr,\
                 k_hs_sq, rho_hs, lam_min, lam_max)


    M = min(M,krs.size) # keep track of number of modes
    phi = get_phi(omega, krs, h_arr, ind_arr, z_arr, k_sq_arr.real, rho_arr, k_hs_sq, rho_hs)

    """ Factor in attenuation with perturbation theory"""
    #if np.sum(attn_arr) > 0 or attn_hs: # there is attenuation in the model
    if np.any(k_sq_arr.imag != 0) or (k_hs_sq.imag != 0): # there is attenuation in the model
        comp_krs = add_arr_attn(omega, krs, phi, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs)
    else:
        comp_krs = krs + 1j*np.zeros(krs.size)

    
    phi_z = get_phi_z(ind_arr, z_arr) #this gets rid of double entries in z_arr for layer interfaces
        
    return comp_krs, phi, phi_z

