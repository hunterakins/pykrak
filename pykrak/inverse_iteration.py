"""
Description:
    Inverse iteration for solution of eigenvectors given eigenvalues

Date:
    4/18/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from numba import njit
from pykrak.mesh_routines import get_A_size_numba, get_A_numba


@njit(cache=True)
def tri_diag_solve(a, e1, d1, w):
    """
    Solve matrix A x = w
    where A is triadiagonal
    a is diagonal
    e1 is upper diagoanl
    d1 is lower 
    This is only stable when a-lambda is larger than d1 for all mesh points
    """
    N = w.size
    x = np.zeros(N)
    e1_new = np.zeros(N-1) 
    w_new = np.zeros(N)

    # 
    e1_new[0] = e1[0]/a[0]
    w_new[0] = w[0]/a[0]
    for i in range(1, N-1):
        #if np.abs(d1[i-1]) > np.abs(a[i]):
        #    print('should swap rows')
        scale = (a[i] - d1[i-1]*e1_new[i-1])
        e1_new[i] = e1[i] / scale
        w_new[i] = (w[i] - d1[i-1]*w_new[i-1]) / scale
    scale = (a[N-1] - d1[N-2] * e1_new[N-2])
    if scale == 0:
        scale = 1e-16
    w_new[N-1]  = (w[N-1] - d1[N-2]*w_new[N-2]) / scale

    x[N-1] = w_new[-1] # solution
    for i in range(1, N):
        ind = N - i - 1
        x[ind] = -x[ind+1]*e1_new[ind] + w_new[ind]
    return x

@njit(cache=True)
def inverse_iter(a, e1, d1, lam):
    """
    Use inverse iteration to find the eigenvector 
    (A - lambda I ) w = 0
    where lam is an estimate of lambda
    """
    wprev = np.ones(a.size)
    wprev /= np.sqrt(a.size) # normalize
    diff=10
    max_num_iter = 5
    count = 0
    while diff > 1e-3 and count < max_num_iter:
        wnext = tri_diag_solve(a-lam, e1, d1, wprev)
        #A_mat = np.zeros((a.size, a.size))
        #A_mat += np.diag(a-lam)
        #A_mat += np.diag(e1, 1)
        #A_mat += np.diag(d1, -1)
        #print(np.abs(A_mat@wnext- wprev).max())
        #wnext = np.linalg.solve(A_mat, wprev)
        #print(np.abs(wnext-wnext_alt).max())
        if np.any(np.isnan(wnext)):
            print('yikes!')
            lam += 1e-10*abs(lam)
            wnext = tri_diag_solve(a-lam, e1, d1, wprev)
            if np.any(np.isnan(wnext)):
                raise ValueError('The sparse matrix mystery strikes again...')
        wnext /= np.linalg.norm(wnext)
        diff = np.linalg.norm(wnext-wprev)/np.linalg.norm(wnext)
        if abs(diff - 2.0) < 1e-10: # sometimes the sign just flips...
            diff = np.linalg.norm(wnext+wprev)/np.linalg.norm(wnext)
            
        wprev = wnext
        count += 1
    if count == max_num_iter:
        print(diff)
        print('Warning: max num iterations reached. Eigenvector may be incorrect.')
    return wnext

@njit(cache=True)
def single_layer_sq_norm(om_sq, phi, h, depth_ind, rho):
    """
    Do the integral over a single layer using trapezoid rule
    phi - np 2d ndarray
        first axis is depth, second is mode index
    om_sq - float
        omega squared
    depth_ind - integer
        input value is the depth index for the first value
        in the layer
    rho - np nd array
        1 dimension, density as a function of depth
    """
    N_layer = rho.size
    N_modes = phi.shape[-1]
    layer_norm_sq = np.zeros(N_modes)
    for k in range(N_layer-1): #end pt handled separately
        depth_val = h*.5*(np.square(phi[depth_ind,:]) + np.square(phi[depth_ind+1]))/rho[k]/om_sq
        layer_norm_sq += depth_val
        depth_ind += 1
   
    # last value is cut in half (since its an interface pt ?
    depth_val = h*np.square(phi[depth_ind,:]) / rho[-1] / om_sq 
    layer_norm_sq += depth_val #
    return layer_norm_sq, depth_ind

@njit(cache=True)
def normalize_phi(omega, phi, krs, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs):
    num_layers = ind_arr.size
    num_modes = phi.shape[-1]
    om_sq = np.square(omega)
    norm_sq = np.zeros(num_modes)
    depth_ind = 0
    """ Use trapezoid rule to integrate each layer"""
    for j in range(num_layers):
        h = h_arr[j]
        if j < num_layers-1:
            z = z_arr[ind_arr[j]:ind_arr[j+1]]
            rho = rho_arr[ind_arr[j]:ind_arr[j+1]]
        else:
            z = z_arr[ind_arr[j]:]
            rho = rho_arr[ind_arr[j]:]

        rho_exp = np.zeros((z.size, 1))
        rho_exp[:,0] = rho.copy()
        integrand = phi[depth_ind:depth_ind+z.size,:]**2 / rho_exp / om_sq
        layer_norm_sq = h*(np.sum(integrand, axis=0) - 0.5*(integrand[0,:]+integrand[-1,:]))
        #layer_norm_sq, depth_ind = single_layer_sq_norm(om_sq, phi, h, depth_ind, rho)
        depth_ind += z.size-1 # the -1 is because the last point is shared with the next layer
        norm_sq += layer_norm_sq
    """
    Now get the halfspace term
    """
    gamma_m = np.sqrt(np.square(krs) - k_hs_sq).real
    norm_sq += np.square(phi[-1,:]) / 2 / gamma_m / rho_hs / om_sq
    rn = om_sq * norm_sq

    phi *= 1.0/np.sqrt(rn)

    """
    Find index of turning point nearest the top for consistent polarization
    """
    for i in range(num_modes):
        itp = np.argmax(np.abs(phi[:,i]))
        j = 1
        while abs(phi[j,i]) > abs(phi[j-1, i]): # while it increases in depth
            j += 1
            if j == phi.shape[0]:
                break
        itp = min(j-1, itp)
        if phi[itp, i] < 0:
            phi[:,i] *= -1
    return phi

@njit(cache=True)
def get_phi(omega, krs, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs):
    A_size = get_A_size_numba(z_arr, ind_arr)
    phi = np.zeros((A_size+1, krs.size)) # +1 point for the surface point
    phi[0,:]=0 # first row is zero
    h0 = h_arr[0]
    for i in range(len(krs)):
        kr = krs[i]
        lam = np.square(h0*kr)
        a, e1, d1 = get_A_numba(h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, lam)
        eig = inverse_iter(a, e1, d1, lam)
        phi[1:,i] = eig

    phi = normalize_phi(omega, phi, krs, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs) 
    return phi
        
