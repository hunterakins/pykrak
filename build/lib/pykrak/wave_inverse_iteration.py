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
from matplotlib import rc
import matplotlib
from numba import njit
from pykrak.wave_mesh_routines import get_A_numba


@njit
def tri_diag_solve(b, c, a, r):
    """
    Solve matrix A x = r
    where A is triadiagonal
    b is diagonal
    c is upper diagoanl
    a is lower 

    Use partial pivoting

    """
    b = b.copy()
    r = r.copy()
    d = np.zeros(c.size-1) # upper upper diagonal
    N = b.size
    u = np.zeros(N) #
    for i in range(N-1):
        gamma = a[i] / b[i]
        b[i+1] = b[i+1] - gamma * c[i]
        r[i+1] = r[i+1] - gamma * r[i]

    u[N-1] = r[-1]/b[-1] 
    for i in range(1, N):
        ind = N - i - 1
        if i == 1:
            u[ind] =  1/b[ind]*(r[ind] - c[ind]*u[ind+1])
        else:
            u[ind] =  1/b[ind]*(r[ind] - c[ind]*u[ind+1] - d[ind]*u[ind+2])
    return u

@njit
def tri_diag_solve_pp(b, c, a, r):
    """
    Solve matrix A x = r
    where A is triadiagonal
    b is diagonal
    c is upper diagoanl
    a is lower 

    Use partial pivoting
    """
    bb = b.copy()
    rr = r.copy()
    cc = c.copy()
    aa = a.copy()
    d = np.zeros(c.size-1) # upper upper diagonal
    N = b.size
    u = np.zeros(N) #
    swap = np.zeros(N-1)


    for i in range(N-1):
        if abs(aa[i]) > abs(bb[i]): # swap rows
            swap[i] = 1
            temp = aa[i]
            aa[i] = bb[i]
            bb[i] = temp
            temp = rr[i]
            rr[i] = rr[i+1]
            rr[i+1] = temp

            temp = cc[i]
            cc[i] = bb[i+1]
            bb[i+1] = temp

            if i < N-2: # set upper upper diagonal
                temp = cc[i+1]
                cc[i+1] = 0.
                d[i] = temp

        gamma = aa[i] / bb[i]
        bb[i+1] = bb[i+1] - gamma * cc[i]
        rr[i+1] = rr[i+1] - gamma * rr[i]
        if i < N-2:
            cc[i+1] = cc[i+1] - gamma * d[i]
        aa[i] = 0


    u[N-1] = rr[-1]/bb[-1] 
    for i in range(1, N):
        ind = N - i - 1
        if i == 1:
            u[ind] =  1/bb[ind]*(rr[ind] - cc[ind]*u[ind+1])
        else:
            u[ind] =  1/bb[ind]*(rr[ind] - cc[ind]*u[ind+1] - d[ind]*u[ind+2])
    return u

@njit
def inverse_iter(a, b, lam):
    """
    Use inverse iteration to find the eigenvector 
    (A - lambda I ) w = 0
    where lam is an estimate of lambda
    """
    norm = np.sum(np.abs(a)) + np.sum(np.abs(b))
    eps3 = 1e-10 *norm
    scale = a.size
    eps4 = scale * eps3
    scale = eps4 / np.sqrt(scale)

    wprev = np.ones(a.size)*scale
    #wprev /= np.sqrt(a.size) # normalize
    diff=10
    max_num_iter = 200
    count = 0
    while diff > 1e-3 and count < max_num_iter:
        wnext = tri_diag_solve_pp(a-lam, b, b, wprev)
        if np.any(np.isnan(wnext)):
            lam += 1e-8*abs(lam)
            wnext = tri_diag_solve_pp(a-lam, b, b, wprev)
            if np.any(np.isnan(wnext)):
                raise ValueError('nan in inverse_iter')
        tmp = np.linalg.norm(wnext)
        wnext /= tmp
        diff = np.linalg.norm(wnext-wprev)/np.linalg.norm(wnext)
        if abs(diff - 2.0) < 1e-10: # sometimes the sign just flips...
            diff = np.linalg.norm(wnext+wprev)/np.linalg.norm(wnext)
        wprev = wnext
        count += 1
    if count == max_num_iter:
        print('Warning: max num iterations reached. Eigenvector may be incorrect.')
    return wnext

@njit
def single_layer_sq_norm(b_arr_sq, phi, h):
    """
    Do the integral over a single layer using trapezoid rule
    b_arr_sq - np 1d array
        N(z)**2 - omega_I**2  units (rad/s)**2
    phi - np 2d ndarray
        first axis is depth, second is mode index
    h - float 
        mesh width

    Implicitly uses fact that phi is 0 at surface and end point so their
    contribution to integral is 0
    """
    N_layer = b_arr_sq.size
    N_modes = phi.shape[-1]
    layer_norm_sq = np.zeros(N_modes)
    for k in range(N_layer): 
        depth_val = b_arr_sq[k] * np.square(phi[k,:])
        layer_norm_sq += depth_val
    layer_norm_sq *= h # times dz
    return layer_norm_sq

@njit
def normalize_phi(b_arr_sq, phi, h):
    layer_norm_sq = single_layer_sq_norm(b_arr_sq, phi[1:-1,:], h)
    phi *= 1.0/np.sqrt(layer_norm_sq)
    num_modes = phi.shape[1]

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

@njit
def get_phi(gammas, k, h, b_arr_sq):
    phi = np.zeros((b_arr_sq.size+2, gammas.size)) #+2 for surface and bottom points...
    a, b = get_A_numba(k, h, b_arr_sq)
    for i in range(len(gammas)):
        gamma = gammas[i]
        lam = np.square(gamma)
        eig = inverse_iter(a, b, lam)
        # first and last row are zero!
        phi[1:-1,i] = eig
    phi = normalize_phi(b_arr_sq, phi, h) 
    return phi
        
