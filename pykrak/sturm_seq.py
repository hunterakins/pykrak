import numpy as np
from matplotlib import pyplot as plt
from numba import njit, vectorize, jit
import numba as nb
import time
from pykrak.mesh_routines import *

"""
Description:
Compute sturm sequence for mode problem.
Find eigenvalues of normal problem using sturm sequences with bisection plus 
scipy brentq root finder. 
Find eigenvectors by inverse iteration.
Normalize eigenvectors as described in COA Ch. 5.

Main routines are
get_krs
get_phi

References:
Computational Ocean Acoustics:
Jensen, F. B., Kuperman, W. A., Porter, M. B., Schmidt, H., & Tolstoy, A. (2011). Computational ocean acoustics (Vol. 794). New York: Springer.

Porter, Michael B., and Edward L. Reiss. "A note on the relationship between finite-difference and hooting methods for ODE eigenvalue problems." SIAM journal on numerical analysis 23.5 (1986): 1034-1039.

Porter, Michael, and Edward L. Reiss. "A numerical method for ocean‐acoustic normal modes." The Journal of the Acoustical Society of America 76.1 (1984): 244-252.

Porter, Michael B. The KRAKEN normal mode program. Naval Research Lab Washington DC, 1992.


Date:
2/9/2022

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego



Copyright (C) 2023  F. Hunter Akins

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


float_arr_type = nb.types.Array(nb.f8, 1, 'A', readonly=True)
int_arr_type = nb.types.Array(nb.i8, 1, 'A', readonly=True)

@njit
def get_scale(seq, j, Phi=1e8, Gamma=1e-8):
    """
    For rescaling sturm sequence
    """
    seq_val = abs(seq[j])
    seq_prev_val = abs(seq[j-1])
    w = max(seq_val, seq_prev_val)
    if w > Phi:
        s = Gamma
    elif w < Gamma and w > 0.:
        s = Phi
    else:
        s = 1.
    return s

@njit
def sturm_subroutine(a_diag, e1, d1, lam):
    """
    Compute sturm sequence (recursive calculation of
    determinant of characteristic function for candidate
    eigenvalue lam
    """
    N = a_diag.size-1 # 
    sturm_seq = np.zeros(N+3)
    sturm_seq[0] = 0.0 #p_{-1}
    sturm_seq[1] = 1.0 #p_{0}
    sturm_seq[2] = -(a_diag[0] - lam) #p_{1} 
    count = 0
    if sturm_seq[2] < 0: #since this doesn't get checked in the loop...
        count += 1
    for k in range(1,N+1):
        sturm_seq[2+k] = -(a_diag[k]-lam)*sturm_seq[2+k-1]  - (e1[k-1]*d1[k-1])*sturm_seq[2+k-2]
        s = get_scale(sturm_seq, k)
        sturm_seq[2+k] = s*sturm_seq[2+k]
        sturm_seq[2+k-1] = s*sturm_seq[2+k-1]
        sturm_seq[2+k-2] = s*sturm_seq[2+k-2]
        if (sturm_seq[2+k])*(sturm_seq[2+k-1]) <= 0.0:
            count += 1
    return sturm_seq[1:], count

def cat_list_to_arr(list_of_arrs):
    """
    Get the environmental description from a list of arrays
    to a single array
    """
    num_arrs = len(list_of_arrs)
    final_arr = list_of_arrs[0]
    ind_arr = np.zeros(num_arrs, dtype=int)
    for i in range(1, num_arrs):
        ind_arr[i] = final_arr.size # index to access the first entry in ith layer
        final_arr = np.concatenate((final_arr, list_of_arrs[i]))
    return final_arr, ind_arr

def get_arrs(h_list, z_list, c_list, rho_list):
    """
    Convert env_lists to arrays for numba-ing
    """
    h_arr = np.array(h_list)
    z_arr, ind_arr = cat_list_to_arr(z_list)
    c_arr, ind_arr = cat_list_to_arr(c_list)
    rho_arr, ind_arr = cat_list_to_arr(rho_list)
    return h_arr, ind_arr, z_arr, c_arr, rho_arr
    
@njit(nb.f8(nb.f8, float_arr_type, int_arr_type, float_arr_type, float_arr_type, float_arr_type, nb.f8, nb.f8, nb.f8))
def get_sturm_seq(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam):
    """
    Compute sturm sequence  for given parameters
    Return final element and number of eigenvalues greater than lam
    """

    a_diag, e1, d1 = get_A_numba(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam)
    
    sturm_seq, count = sturm_subroutine(a_diag, e1, d1, lam)
    return sturm_seq[-1]

@njit(float_arr_type(nb.f8, float_arr_type, int_arr_type, float_arr_type, float_arr_type, float_arr_type, nb.f8, nb.f8, nb.f8))
def get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam):
    """
    Compute sturm sequence  for given parameters
    Return final element and number of eigenvalues greater than lam
    """

    a_diag, e1, d1 = get_A_numba(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam)
    
    sturm_seq, count = sturm_subroutine(a_diag, e1, d1, lam)
    return np.array([sturm_seq[-1], count])

@njit(nb.f8(nb.f8, float_arr_type, int_arr_type, float_arr_type, float_arr_type, float_arr_type, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8))
def layer_brent(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, a, b, t):
    """
      Licensing:
    
        This code is distributed under the GNU LGPL license.
    
      Modified:
    
        08 April 2023
    
      Author:
    
        Original FORTRAN77 version by Richard Brent
        Python version by John Burkardt
        Numba-ized version specific for the layered S-L problem by Hunter Akins
    """
    machep=1e-16

    sa = a
    sb = b
    fa = get_sturm_seq(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs,  sa )
    fb = get_sturm_seq(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs,  sb )

    c = sa
    fc = fa
    e = sb - sa
    d = e

    while ( True ):

        if ( abs ( fc ) < abs ( fb ) ):

            sa = sb
            sb = c
            c = sa
            fa = fb
            fb = fc
            fc = fa

        tol = 2.0 * machep * abs ( sb ) + t
        m = 0.5 * ( c - sb )

        if ( abs ( m ) <= tol or fb == 0.0 ):
            break

        if ( abs ( e ) < tol or abs ( fa ) <= abs ( fb ) ):

            e = m
            d = e

        else:

            s = fb / fa

            if ( sa == c ):

                p = 2.0 * m * s
                q = 1.0 - s

            else:

                q = fa / fc
                r = fb / fc
                p = s * ( 2.0 * m * q * ( q - r ) - ( sb - sa ) * ( r - 1.0 ) )
                q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 )

            if ( 0.0 < p ):
                q = - q

            else:
                p = - p

            s = e
            e = d

            if ( 2.0 * p < 3.0 * m * q - abs ( tol * q ) and p < abs ( 0.5 * s * q ) ):
                d = p / q
            else:
                e = m
                d = e

        sa = sb
        fa = fb

        if ( tol < abs ( d ) ):
            sb = sb + d
        elif ( 0.0 < m ):
            sb = sb + tol
        else:
            sb = sb - tol

        fb = get_sturm_seq(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs,  sb )

        if ( ( 0.0 < fb and 0.0 < fc ) or ( fb <= 0.0 and fc <= 0.0 ) ):
            c = sa
            fc = fa
            e = sb - sa
            d = e

    value = sb
    return value

@njit(cache=True)
def find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam_max):
    tol = 1e-16 # close to machine precision
    root = layer_brent(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam_max, tol)
    return root
    
def get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam_max, N=-1, Nr_max=-1):
    """
    Recursively bisect the domain, counting the modes in each 
    subdomain and employing Brentq root finder once the number 
    of modes has been isolated
    """
    if z_arr[0] != 0:
        raise ValueError("Must include z=0")
    h0 = h_arr[0]
    dz0 = z_arr[1] - z_arr[0]
    if h0 != dz0:
        raise ValueError('Mesh grid differs from spacing in h_arr')
    if N == -1: # need to compute total number of modes
        det, N = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min)
        if N == 0:
            return np.array([])
    if Nr_max == -1: # there may exist modes beyond what the user wants to compute
        det, Nr_max = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_max)

    lam = .5*(lam_min + lam_max)
    if (lam == lam_max) or (lam == lam_min): # this seems to be an error?
        return []

    det, Nr_mid = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam)

    sub_Nr = Nr_mid - Nr_max # the number greater than lam and less than lam_max
    sub_Nl = N - Nr_mid #  number to the left of the midpoint

    if sub_Nl == 0: #none in less than lam
        if sub_Nr == 0: # none less than lam_max and greater than lam
            return [] 
        elif sub_Nr == 1: # one between lam and lam_max
            kr_right = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam, lam_max)
            kr_right = np.sqrt(kr_right)/h0
            return [kr_right]
        else: # more than one between lam and lam_max
            kr_right = get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam, lam_max, N, Nr_max=Nr_max)
            return kr_right
    elif sub_Nl == 1:
        kr_left = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam)
        kr_left = np.sqrt(kr_left)/h0
        if sub_Nr == 0:
            return [kr_left]
        elif sub_Nr == 1:
            kr_right = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam, lam_max)
            kr_right = np.sqrt(kr_right)/h0
            return [kr_left, kr_right]
        else:
            kr_right = get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs,lam, lam_max, N-sub_Nl, Nr_max=Nr_max)
            return [kr_left]+ kr_right
    else: # more then one to the left
        #kr_left = get_krs(omega, h, z, c, rhow, cb, rhob, avec, lam_min, lam, N, Nr_max=Nr_max+sub_Nr)
        kr_left = get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam, N, Nr_max=Nr_max + sub_Nr)
        if sub_Nr == 0:
            return kr_left
        elif sub_Nr == 1:
            kr_right = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam, lam_max)
            kr_right = np.sqrt(kr_right)/h0
            return kr_left+ [kr_right]
        else:
            #kr_right = get_krs(omega, h, z, c, rhow, cb, rhob, avec, lam, lam_max, N-sub_Nl, Nr_max=Nr_max)
            kr_right = get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam, lam_max, N-sub_Nl, Nr_max=Nr_max)
            return kr_left+ kr_right

@njit(cache=True)
def get_comp_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam_max):
    """
    Recursively bisect the domain, counting the modes in each 
    subdomain and employing Brentq root finder once the number 
    of modes has been isolated
    """
    h0 = h_arr[0]
    dz0 = z_arr[1] - z_arr[0]
    det, Nr = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_max)

    det, Nl = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min)
    M = int(Nl - Nr)

    if M == 0:
        return np.zeros((0))


    
    max_num_bisections = 50
    
    lam_l = lam_min*np.ones((M))
    lam_r = lam_max*np.ones((M))

    krs = np.zeros((M))
    for i in range(M-1):
        lam1 = lam_l[i] 
        lam2 = lam_r[i]

        for k in range(max_num_bisections):
            lam_prop = .5*(lam1 + lam2)
            det, N = get_sturm_seq_count(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_prop)

            Nz = N - Nr # number of zeros in interval from lam_prop to lam_max
            if Nz < i + 1: # move the right boundary
                lam2 = lam_prop
                lam_r[i] = lam_prop
            else: # move the left boundary
                lam1 = lam_prop
                # also give the clue to all coming modes
                for mode_i in range(i, M):
                    if Nz >= mode_i+1:
                        lam_l[mode_i] = lam_prop
                    if Nz < mode_i + 1:
                        lam_r[mode_i] = lam_prop
            if Nz == i +1: # found the right interval  
                # lam_prop is the left boundary
                lam_l[i] = lam_prop
                lam_r[i+1] = lam_prop
                break
        root = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_l[i], lam_r[i])
        krs[i] = np.sqrt(root)/h0
    root = find_root(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_l[M-1], lam_r[M-1])
    krs[M-1] = np.sqrt(root)/h0
    return krs

