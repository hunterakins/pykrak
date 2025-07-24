"""
Description:
    Functions to get the discretized wave equation on a grid

Date:
    4/18/2023 (copied from sturm_seq)

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

@njit(float_arr_type(nb.f8, float_arr_type, float_arr_type, nb.f8))
def get_a(h, k_sq, rho, h0):
    """
    Compute the diagonal elements a used for depths within a layer
    For the pseudo-Sturm Liouville problem
    I use the layer mesh scaling so that multiple layers can be used with different
    meshes and operate on the same eigenvalue kr^2 h0^2
    which results in 
    d_{i} = -2 h_{0}^{2}/h_{i}^{2} + h_{0}^{2}(\omega^{2} / c^{2}(z_{i})) - \lambda
    \implies a_{i} = -2 h_{0}^{2}/h_{i}^{2} + h_{0}^{2}(\omega^{2}

    The first and last entries of c and rho are the values
    at the layers. The interface value is handled separately, 
    so the first depth returned here is z[1] and the last is z[-2]
    To accomplish that I compute for every detph and discard first and last pt
    I compute the value for the interface depth using a_last or a_bdry
    Input
    k_sq - np 1d array
        (omega / ssp vals)**2 at depths z0, z1, ..., zN-1  (local wavenumber omega / c(z) squared)
        as mentioned, depths z0 and zN-1 are interface depths
    rho - np 1d array
        density at depths z0, z1, ..., zN-1 
    h0 - float
        spacing of first mesh grid
    Output - 
    avec - np 1d array
        diagonal values of matrix
    """
    avec = (-2.0*h0*h0/h/h + h0*h0*k_sq)
    return avec[1:-1]
   
@njit(nb.f8(nb.f8, nb.f8, nb.f8, nb.c16, nb.f8, nb.f8, nb.f8))
def get_a_last(h, ku_sq, rhou, kb_sq, rhob, h0, lam):
    """
    Enforce halfspace boundary condition
    Eqn. 5.110 but multiplied by 2*h*rho * h0^2 / h^2 
    fb(kr^2) = 1
    gb(kr^2) = rhob / sqrt(kr^2- (w/cb)^2) 5.62

    a_{N} = $ -h_{0}^{2} / h^{2} + \frac{1}{2}h_{0}^{2} \omega^{2} / c_{u}^{2} 
            -\frac{h_{0}^{2} \rho_{u} h}{h^{2} \rho_{b}} \gamma_{b}^{2} $
    """
    kr_sq = lam /(h0*h0)
    if kr_sq < kb_sq.real:
        raise ValueError('cmax not set properly (mode is not exponentially decaying in halfspace). kr_sq < (omega / cb)^2 ({0} < {1}')
    gamma_b = np.sqrt(kr_sq - kb_sq)
    gamma_b = gamma_b.real

    term1 = -2*h0*h0 / (h*h)
    term2 = h0*h0*ku_sq
    term3 = -gamma_b * 2* h0*h0 * rhou / ( h * rhob )
    alast = term1 + term2 + term3
    return alast

@njit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8))
def get_a_bdry(hu, hb, ku_sq, rhou, kb_sq, rhob, h0):
    """
    Get center term for finite difference approximation to layer condition  
    Equation 5.131 multiplied through by alpha = h0^2 (h_u / (2 \rho_u) + h_b / 2 \rho_b)^{-1}
    hu - float
        mesh grid stepsize in above layer
    hb - float
        mesh grid stepsize in below layer
    ku_sq - float
        local wavenumber squared above
    rhou
        density above
    kb_sq - float
        local wavenumber k(z) squared evaluated below
    rhob - float
        density below
    h0 - float 
        reference meshgrid step size for A
    """
    #om_sq = np.square(omega)
    #cu_sq = np.square(cu)
    #cb_sq = np.square(cb)
    alpha = h0*h0 / ((hu/2/rhou + hb/2/rhob))
    term1 = -1/(hu*rhou)
    term2 = -1/(hb*rhob)
    term3 = .5*hu*ku_sq / rhou
    term4 = .5*hb*kb_sq / rhob
    a = alpha*(term1 + term2 + term3 + term4)
    return a

def get_A_size(z_list):
    """ Each layer in z_list shares an interface point
    Since I don't use the first depth (pressure release) in z_list[0],
    just counting z_list[i][1:] will include all depths needed (including
    last boundary condition)
    """
    a_size = 0
    for x in z_list:
        a_size += x.size -1 
    return a_size

@njit
def get_A_size_numba(z_arr, ind_arr):
    a_size = z_arr.size - ind_arr.size # subtract 
    return a_size

def get_A(h_list, z_list, k_sq_list, rho_list, k_hs_sq, rho_hs, lam):
    """
    Compute diagonal and off diagonal terms for the matrix
    required in the Sturm sequence solution method

    The way the indexing works:
    a[i] is the ith diagonal element
    d[i] is the element of the matrix directly beneath a[i]
    e[i] is the element of the matrix directly to the right of a[i]

    h_list - list of floats
        mesh width for each layer
    z_list - list of 1d numpy ndarrays
        each element is the depths of the c and rho vals
    k_sq_list - list of 1d numpy ndarrays
        each element is the local wavenumber squared k^2(z) = (omega / c(z))^2
    rho_list - list of 1d numpy ndarrays
        elements are discretized density for each layer
    c_hs - float
        halfspace speed
    rho_hs - float
        halfspace density
    """
    num_layers = len(h_list)
    h0 = h_list[0]
    a_size = get_A_size(z_list)
    a_diag = np.zeros((a_size))
    e1 = np.zeros((a_size)-1) # upper off-diagonal 
    d1 = np.zeros((a_size)-1) # lower off-diagonal 
    upper_layer_ind = 0 # index of upper layer (entry in diag a)
    for i in range(num_layers):
        """
        First compute the diagonal terms 
        """
        z = z_list[i]
        k_sq = k_sq_list[i]
        h = h_list[i]
        rho = rho_list[i]

        """ 
        Fill z.size - 2 entries corresponding to depths starting below the upper layer
        interface and going down to the grid point above the bottom of the layer interface
        Then add in line for boundary value
        """
        a_layer = get_a(h, k_sq, rho, h0) # remember this contains no interface points
        a_inds = (upper_layer_ind, upper_layer_ind + z.size-2) # z includes the interface points, so exclude those...
        a_diag[a_inds[0]:a_inds[1]] = a_layer[:]

        """ Now add the final entry for the interface beneath the layer """
        hu = h
        rhou = rho[-1]
        ku_sq = k_sq[-1]

        """ If it's not the bottom halfspace """
        if i < num_layers - 1: 
            hb = h_list[i+1]
            rhob = rho_list[i+1][0]
            kb_sq = k_sq_list[i+1][0]
            a_bdry = get_a_bdry(hu, hb, ku_sq, rhou, kb_sq, rhob, h0)

        else: # last layer
            rhob =rho_hs
            kb_sq = k_hs_sq
            a_bdry = get_a_last(hu, ku_sq, rhou, kb_sq, rhob, h0, lam)
        a_diag[a_inds[1]] = a_bdry

        """ Now compute off diagonal terms """
        e1[a_inds[0]:a_inds[1]] = h0*h0 / (h*h)  # above diag
        d1[a_inds[0]:a_inds[1]-1] = h0*h0 / (h*h) # below diag

        if i < num_layers - 1: # if not on the last layer, e1
            alpha = h0*h0 / (.5 * (hu / rhou + hb / rhob))
            f = alpha / (hu*rhou)
            d1[a_inds[1]-1] = f
            d1[a_inds[1]] = h0*h0 / (hb*hb) # this point extends into the next layer

            g = alpha / (hb*rhob)
            e1[a_inds[1]] = g

        else: # for bottom boundary condition
            d1[a_inds[1]-1] = 2*h0*h0 / (h*h)

        upper_layer_ind += z.size - 1
    return a_diag, e1, d1

@njit
def get_A_numba(h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, lam):
    """
    the arrays are the concatenated list elements in the equivalent function above
    ind_arr gives the index of the ith layer
    so the first element is always zero
    z_arr[ind_arr[i]] is the first value in the ith layer
    """
    num_layers = h_arr.size
    h0 = h_arr[0]
    #a_size = get_A_size(z_list)
    a_size = z_arr.size - ind_arr.size # don't double count layer interfaces
    a_diag = np.zeros((a_size))
    e1 = np.zeros((a_size)-1) # upper off-diagonal 
    d1 = np.zeros((a_size)-1) # lower off-diagonal 
    upper_layer_ind = 0 # index of upper layer (entry in diag a)
    for i in range(num_layers):
        """
        First compute the diagonal terms 
        """
        if i < num_layers-1:
            z = z_arr[ind_arr[i]:ind_arr[i+1]]
            k_sq = k_sq_arr[ind_arr[i]:ind_arr[i+1]]
            h = h_arr[i]
            rho = rho_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            z = z_arr[ind_arr[i]:]
            k_sq = k_sq_arr[ind_arr[i]:]
            h = h_arr[i]
            rho = rho_arr[ind_arr[i]:]
        """ 
        Fill z.size - 2 entries corresponding to depths starting below the upper layer
        interface and going down to the grid point above the bottom of the layer interface
        Then add in line for boundary value
        """
        a_layer = get_a(h, k_sq, rho, h0) # remember this contains no interface points
        a_inds = (upper_layer_ind, upper_layer_ind + z.size-2) # z includes the interface points, so exclude those...
        a_diag[a_inds[0]:a_inds[1]] = a_layer[:]

        """ Now add the final entry for the interface beneath the layer """
        hu = h
        rhou = rho[-1]
        ku_sq = k_sq[-1]

        """ If it's not the bottom halfspace """
        if i < num_layers - 1: 
            hb = h_arr[i+1]
            rhob = rho_arr[ind_arr[i+1]]
            kb_sq = k_sq_arr[ind_arr[i+1]]
            a_bdry = get_a_bdry(hu, hb, ku_sq, rhou, kb_sq, rhob, h0)

        else: # last layer
            rhob =rho_hs
            kb_sq = k_hs_sq
            a_bdry = get_a_last(hu, ku_sq, rhou, kb_sq, rhob, h0, lam)
        a_diag[a_inds[1]] = a_bdry

        """ Now compute off diagonal terms """
        e1[a_inds[0]:a_inds[1]] = h0*h0 / (h*h)  # above diag
        d1[a_inds[0]:a_inds[1]-1] = h0*h0 / (h*h) # below diag

        if i < num_layers - 1: # if not on the last layer, e1
            alpha = h0*h0 / (.5 * (hu / rhou + hb / rhob))
            f = alpha / (hu*rhou)
            d1[a_inds[1]-1] = f
            d1[a_inds[1]] = h0*h0 / (hb*hb) # this point extends into the next layer

            g = alpha / (hb*rhob)
            e1[a_inds[1]] = g

        else: # for bottom boundary condition
            d1[a_inds[1]-1] = 2*h0*h0 / (h*h)

        upper_layer_ind += z.size - 1
    return a_diag, e1, d1

