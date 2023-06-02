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
    ug_layer = omega / krm *integral 
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
        delta_ugm = np.square(phim[-1])*omega/(2*krm*gammam*np.square(c_hs)*rho_hs)
        ugm += delta_ugm
        ugs[i] = 1/ugm
    return ugs
        
