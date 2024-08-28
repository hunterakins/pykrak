import numpy as np
from matplotlib import pyplot as plt
from numba import njit

"""
Description:
Use perturbation theory to estimate group speed of modes

Date:
8/3/2022

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


@njit
def ug_layer_integral(omega, krm, phim_layer, c, rho, dz):
    layer_integrand = np.square(phim_layer) / np.square(c) / rho 
    integral = dz*(np.sum(layer_integrand) - .5*layer_integrand[0] - .5*layer_integrand[-1])
    ug_layer = integral 
    return ug_layer

def get_ugs(omega, krs, phi, h_list, z_list, c_list, rho_list,\
                c_hs, rho_hs):
    """
    Use perturbation theory to get group speed
    attenuation
    Relevant equations from JKPS is Eq. 5.189, where D is the depth of the halfspace (not ht elayer)
    The imaginary part of the wavenumber is negative, so that a field exp(- i k_{r} r) radiates outward,
    consistent with a forward fourier transform of the form P(f) = \int_{-\infty}^{\infty} p(t) e^{-i \omega t} \dd t

    Input - 
    omega - float
        source frequency
    krs - np ndarray of floats
        wavenumbers (real)
    phi - np ndarray of mode shapes
        should be evaluated on the supplied grid of z_list
    h_list - list of floats
        step size of each layer mesh
    z_list - list of np ndarray of floats   
        depths for mesh of each layer
    c_list - list of np ndarray of floats
        values of sound speed at each depth (real)
    rho_list - list of np ndarray of floats 
        densities at each depth
    c_hs - float
        halfspace speed (m/s)
    rho_hs - float
        halfpace density (g / m^3) 
    """
    ugs = np.zeros((krs.size))
    num_modes = krs.size
    num_layers = len(h_list)
    for i in range(num_modes):
        layer_ind = 0
        phim = phi[:,i]
        krm = krs[i]
        ugm = 0.0
        for j in range(num_layers):
            z = z_list[j]
            num_pts = z.size
            phim_layer = phim[layer_ind:layer_ind+num_pts]
            c = c_list[j]
            rho = rho_list[j]
            dz = h_list[j]
            ug_layer = ug_layer_integral(omega, krm, phim_layer, c, rho, dz)
            ugm += ug_layer
            layer_ind += num_pts - 1
        gammam = np.sqrt(np.square(krm) - np.square(omega / c_hs))
        delta_ugm = np.square(phim[-1])/(2*gammam*rho_hs* c_hs**2)
        ugm += delta_ugm
        ugm *= omega/ krm
        ugs[i] = 1/ugm
    return ugs
        

from pykrak.misc import get_simpsons_integrator, get_layer_N
@njit(cache=True)
def get_layered_ug_integrator(h_arr, ind_arr, z_arr, k_sq_arr, rho_arr):
    """
    Input
    h_arr - array of mesh spacings for each layer
    ind_arr - array of index of first value for each layer
    z_arr - depths of the layer meshes concatenated
    k_sq_arr - real part  wavenumber square omega^2 / c^2 for the layer meshes concatenated
    rho_arr - density of the layer meshes concatenated

    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for i in range(len(layer_N)):
        if i < num_layers -1 :
            k_sq_i = k_sq_arr[ind_arr[i]:ind_arr[i+1]]
            rho_i = rho_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            k_sq_i = k_sq_arr[ind_arr[i]:]
            rho_i = rho_arr[ind_arr[i]:]
        integrator_i = get_simpsons_integrator(layer_N[i], h_arr[i])[0,:]
        integrator_i *= k_sq_i / (rho_i)
        if i == 0:
            integrator = integrator_i
        else:
            integrator[-1] += integrator_i[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_i[1:]))
    return integrator

@njit(cache=True)
def get_arr_ug(omega, krs, phi, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs):
    k_sq_arr = k_sq_arr.real
    integrator = get_layered_ug_integrator(h_arr, ind_arr, z_arr, k_sq_arr.real, rho_arr)
    ugs = np.zeros(krs.size)
    for m in range(krs.size):
        krm = krs[m]
        phim = phi[:,m]
        # integrate through layers
        integ = np.sum(np.square(phim) * integrator)

        # add contribution from tail
        gamma = np.sqrt(np.square(krm) - k_hs_sq.real)
        if gamma != 0:
            delta_int = k_hs_sq.real*np.square(phim[-1])/(2*gamma*rho_hs)
            integ += delta_int
        ugs[m] = (omega * krm) / integ
    return ugs
