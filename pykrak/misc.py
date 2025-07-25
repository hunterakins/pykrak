"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt


from numba import njit
import numba as nb


@njit
def get_simpsons_integrator(N, dz):
    """
    Given a number of evenly-spaced points N at which a
    function has been evaluated, return a row vector that
    acts as an operator on the evaluated function as a column
    vector
    """
    integrator = np.ones(N)
    inds = np.linspace(0, N - 1, N)
    if N % 2 == 1:  #
        integrator[inds % 2 == 1] = 4 * dz / 3
        integrator[inds % 2 == 0] = 2 * dz / 3
        integrator[0] -= dz / 3
        integrator[-1] -= dz / 3

    else:  # do trapezoid on first interval and simpson's on rest of them
        """ Simpson's rule """
        integrator[inds % 2 == 0] = 4 * dz / 3
        integrator[inds % 2 == 1] = 2 * dz / 3
        """ Trapezoid rule """
        integrator[0] = dz / 2
        integrator[1] = dz / 2
        if N > 2:
            integrator[1] += dz / 3
            integrator[-1] -= dz / 3
    integrator = integrator.reshape(1, N)
    return integrator


def test_simpsons_integrator():
    for N in [2, 3, 4, 5, 6, 7]:
        print(get_simpsons_integrator(N, 1))

    f = lambda x: np.sin(x)
    h_list = [0.1, 0.01, 0.001]
    for h in h_list:
        N = int(2 * np.pi / h)
        x_grid = np.linspace(0, 2 * np.pi, N)
        dx = x_grid[1] - x_grid[0]
        f_vals = f(x_grid)
        f_vals = f_vals.reshape(N, 1)
        integrator = get_simpsons_integrator(N, dx)
        integral = integrator @ f_vals
        print("dz, integral", dx, integral)


@njit
def get_layer_N(ind_arr, z_arr):
    """
    An index array is used to allow one to concatenate the meshes of each layer
    in a layered model into a single array
    ind_arr[i] accesses the first element in the ith layer mesh
    This routine gets the number of points in each layer
    """
    num_layers = len(ind_arr)
    layer_arr = np.zeros((num_layers), dtype=nb.i4)
    if num_layers == 1:
        layer_arr[0] = z_arr.size
        return layer_arr
    layer_arr = np.zeros((num_layers), dtype=nb.i4)
    for i in range(num_layers - 1):
        N = ind_arr[i + 1] - ind_arr[i]
        layer_arr[i] = N
    layer_arr[-1] = z_arr[ind_arr[-1] :].size
    return layer_arr
