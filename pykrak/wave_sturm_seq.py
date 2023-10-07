import numpy as np
from matplotlib import pyplot as plt
from numba import njit, vectorize, jit
import numba as nb
import time
from pykrak.wave_mesh_routines import *

"""
Description:
    Modification of sturm sequence codes for computing dispersion
    relation for internal waves. 


Main routines are
get_gammas


Date:
    5/19/2023

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
def get_scale(seq, j, Phi=1e5, Gamma=1e-5):
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
def sturm_subroutine(a, b, lam):
    """
    Compute sturm sequence (recursive calculation of
    determinant of characteristic function for candidate
    eigenvalue lam
    """
    roof = 1e5
    floor = 1e-5
    N = a.size # 
    sturm_seq = np.zeros(N+1)
    sturm_seq[0] = 1.0 #p_{0}
    sturm_seq[1] = (a[0] - lam) #p_{1} 
    p1 = 1.0
    p2 = a[0] - lam
    count = 0
    for k in range(N-1):
        p0 = p1
        p1 = p2
        p2 = (a[k+1] - lam)*p1 - (b[k]**2)*p0

        while (abs(p2) < floor) and (abs(p2) > 0.0):
            p0 = roof*p0
            p1 = roof*p1
            p2 = roof*p2

        while (abs(p2) > roof) and (abs(p2) > 0.0):
            p0 = floor*p0
            p1 = floor*p1
            p2 = floor*p2

        if p2*p1 <= 0:
            count += 1

    return p2, count

@njit(nb.f8(nb.f8, nb.f8, float_arr_type, nb.f8))
def get_sturm_seq(k, h, b_arr_sq, lam):
    """
    Compute sturm sequence  for given parameters
    Return final element and number of eigenvalues greater than lam
    """
    a, b = get_A_numba(k, h, b_arr_sq)
    det, count = sturm_subroutine(a, b, lam)
    return det

@njit(float_arr_type(nb.f8, nb.f8, float_arr_type, nb.f8))
def get_sturm_seq_count(k, h, b_arr_sq, lam):
    """
    Compute sturm sequence  for given parameters
    Return final element and number of eigenvalues greater than lam
    """
    a, b = get_A_numba(k, h, b_arr_sq)
    det, count = sturm_subroutine(a, b, lam)
    return np.array([det, count])

@njit(nb.f8(nb.f8, nb.f8, float_arr_type, nb.f8, nb.f8, nb.f8))
def layer_brent(k, h, bn, a, b, t):
    """
      Licensing:
    
        This code is distributed under the GNU LGPL license.
    
      Modified:
    
        08 April 2023
    
      Author:
    
        Original FORTRAN77 version by Richard Brent
        Python version by John Burkardt
        Numba-ized version specific for the layered S-L problem by Hunter Akins

    k is the horizontal wavenumber 
    h is the mesh grid
    bn is the array of buoyancy frequency squared minus intertial frequency
    a is left bracket wavenumber
    b is right
    t is tolerance
    """
    machep=1e-16

    sa = a
    sb = b
    fa = get_sturm_seq(k, h, bn,  sa )
    fb = get_sturm_seq(k, h, bn,  sb )

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

        fb = get_sturm_seq(k, h, bn,  sb )

        if ( ( 0.0 < fb and 0.0 < fc ) or ( fb <= 0.0 and fc <= 0.0 ) ):
            c = sa
            fc = fa
            e = sb - sa
            d = e

    value = sb
    return value

@njit
def find_root(k, h, b_arr_sq, lam_min, lam_max):
    tol = 1e-16 # close to machine precision
    root = layer_brent(k, h, b_arr_sq, lam_min, lam_max, tol)
    return root
   
@njit
def get_comp_gammas(k_rad, h, b_arr_sq, J):
    """
    Find J modes
    Start at lam_min
    Move in step size of size dlam until a new mode is discovered
    Then find it with find_root
    Quit after finding J modes
    """
    max_num_iter = 1e7
    lam_step = 0.1
    lam_min = k_rad*k_rad / (np.max(b_arr_sq))
    num_found = 0
    gammas = np.zeros((J))
    j = 0 # index of one I'm looking for
    lam_l = lam_min + lam_step
    #lam = 0.14864785075187684
    detl, Nl = get_sturm_seq_count(k_rad, h, b_arr_sq, lam_l)
    curr_lam = lam_l + lam_step
    count = 0
    while j < J and (count < max_num_iter):
        det, N = get_sturm_seq_count(k_rad, h, b_arr_sq, curr_lam)
        if det*detl < 1: # found a new one
            root = find_root(k_rad, h, b_arr_sq, lam_l, curr_lam)
            gammas[j] = np.sqrt(root)
            j += 1
            lam_l = curr_lam
            Nl = N
            detl = det
        else:
            Nl = N
            detl = det
            lam_l = curr_lam
        curr_lam += lam_step
        count += 1
    return gammas
