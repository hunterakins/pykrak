"""
Description:
    The discretized grids for the simple single layer
    pressure release boundary condition problem for solving
    the internal wave modal equation

Date:
    5/19/2023 (coied from mesh_routines.py

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego

Copyright (C) 2023 F. Hunter Akins

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

import numpy as np
import numba as nb
from numba import njit


float_arr_type = nb.types.Array(nb.f8, 1, 'A', readonly=True)
int_arr_type = nb.types.Array(nb.i8, 1, 'A', readonly=True)

@njit(float_arr_type(nb.f8, nb.f8, float_arr_type))
def get_a(k, h, b_sq):
    """
    Input
    k - float 
        (sqrt(kx^2 + ky^2))
    h - float
        grid spacing
    b_sq - np 1d array
        depth values for grid squared
        b^2(z) = N^2(z) - \omega_I^2 
        assumes that grid values for the ocean surface and ocean bottom
        are NOT included 
    Output - 
    avec - np 1d array
        diagonal values of matrix 
    """
    avec = (2.0+ h**2 * k**2) / (h**2 * b_sq)
    return avec

@njit
def get_b(h, b_sq):
    """
    Return off diagonal components
    """
    b = np.sqrt(b_sq)
    bvec = -1.0 / (h**2 * b[1:] * b[:-1])
    return bvec

@njit
def get_A_numba(k, h, b_sq):
    """
    the arrays are the concatenated list elements in the equivalent function above
    ind_arr gives the index of the ith layer
    so the first element is always zero
    z_arr[ind_arr[i]] is the first value in the 
    """
    a = get_a(k, h, b_sq)
    b = get_b(h, b_sq)
    return a, b

