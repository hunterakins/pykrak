"""
Description:

Date:

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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from numba import njit

@njit
def integrate_down_layer(x0, h, kr_sq, omega, c_layer):
    """
    Integrate through the layer
    """
    num_pts = len(c_layer)-1 # value at bottom is from prediction
    x = np.zeros((2, num_pts+1))
    x[:,0] = x0
    for i in range(num_pts):
        c = c_layer[i]
        k1 = downshoot_dxdz(x[:,i], kr_sq, omega, c)
        k2 = downshoot_dxdz(x[:,i] + h*k1/2, kr_sq, omega, c)
        k3 = downshoot_dxdz(x[:,i] + h*k2/2, kr_sq, omega, c)
        k4 = downshoot_dxdz(x[:,i] + h*k3, kr_sq, omega, c)
        x0 = x0 + 1/6*h*(k1 + 2*k2 + 2*k3 + k4) # move x_{n} to x_{n+1}
        x[:, i+1] = x0
    return x

@njit
def shoot_first_layer(omega, h, z, c, rho, lam):
    """ Shoot a mode down from the surface to the bottom of first layer 
    Input
    d """
    #z = z[2:]
    #c = c[2:]
    x0 = np.zeros(2)
    x0[1] = 1/(z[1]-z[0]) # so that mode[z[1]] = 1[:,0]
    kr_sq = lam/(h*h) 
    int_x = integrate_down_layer(x0, h, kr_sq, omega, c)
    mode  = int_x[0,:]
    return mode

@njit
def downshoot_dxdz(x, kr_sq, omega, c):
    """
    At a given depth z, get the update equation
    for runge kutta
    """
    dxdz = np.zeros(2)
    dxdz[0] = x[1]
    dxdz[1] = x[0]*(kr_sq - np.square(omega / c))
    return dxdz

@njit
def upshoot_dxdz(x, kr_sq, omega, c):
    """
    At a given depth z, get the update equation
    for runge kutta
    """
    dxdz = np.zeros(2)
    dxdz[0] = x[1]
    dxdz[1] = x[0]*(kr_sq - np.square(omega / c))
    return dxdz

@njit
def integrate_layer(x0, h, kr_sq, omega, c_layer):
    """
    Integrate through the layer
    """
    num_pts = len(c_layer)-1 # don't need c at top of layer (since it's a forward integration)
    x = np.zeros((2, num_pts+1))
    x[:,0] = x0
    for i in range(num_pts):
        c = c_layer[i]
        k1 = upshoot_dxdz(x[:,i], kr_sq, omega, c)
        k2 = upshoot_dxdz(x[:,i] + h*k1/2, kr_sq, omega, c)
        k3 = upshoot_dxdz(x[:,i] + h*k2/2, kr_sq, omega, c)
        k4 = upshoot_dxdz(x[:,i] + h*k3, kr_sq, omega, c)
        x0 = x0 + 1/6*h*(k1 + 2*k2 + 2*k3 + k4) # move x_{n} to x_{n+1}
        x[:, i+1] = x0
    return x[:,1:]

@njit
def shoot_from_bottom(omega, h_list, z_list, c_list, rho_list, c_hs, rho_hs, lam):
    """ Shoot a mode from the bottom to the surface  
    Use fourth order Runge-Kutta to integrate up
    Input
    omega - 2 pi f
    h_list - list of grid meshes...
     """
    h0 = h_list[0]
    kr_sq = lam / h0/h0
    gamma = np.sqrt(kr_sq - omega*omega / c_hs / c_hs)
    rho_above_bottom = rho_list[-1][-1]
    dphi_dz =  - gamma / rho_hs * rho_above_bottom
    num_layers = len(h_list)
    #num_depths = sum([len(x) for x in c_list]) - len(c_list) + 1
    num_depths = 0
    for i in range(len(c_list)):
        num_depths += len(c_list[i])
    num_depths = num_depths - len(c_list) + 1
    x = np.zeros((2, num_depths))
    z = np.zeros(num_depths)
    #x0 = np.array([1e-20, dphi_dz], dtype=np.float64)
    x[0, 0] = 1e-10
    x[1,0] = dphi_dz*1e-10
    x0 = x[:,0]
    depth_ind = 1
    z[0] = z_list[-1][-1]
    for ind in range(len(h_list)):
        layer_ind = num_layers - 1 - ind
        h_layer = h_list[layer_ind]
        c_layer = c_list[layer_ind][::-1]
        z_layer = z_list[layer_ind][::-1]
        num_layer_pts = len(c_layer) - 1
        x_layer = integrate_layer(x0, h_layer, kr_sq, omega, c_layer)
        x[:, depth_ind:depth_ind + num_layer_pts] = x_layer
        z[depth_ind:depth_ind + num_layer_pts] = z_layer[1:]

        depth_ind += num_layer_pts
        x0 = x_layer[:, -1]
        if ind < (len(h_list) -1): # enforce continuity of velocity
            dphi_dz_below = x0[-1]
            rho_below = rho_list[layer_ind][0] # top density of this layer
            rho_above = rho_list[layer_ind-1][-1] # next layer's density above layer
            dphi_dz_above =  dphi_dz_below / rho_below * rho_above
            x0[-1] = dphi_dz_above
    x = x[:,::-1] # sort from shallowest to deepest
    z = z[::-1]
    x = x[0,1:] #throw out first point because that one is at the top boundary
    z = z[1:]
    return x,z 
