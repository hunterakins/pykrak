import numpy as np
from matplotlib import pyplot as plt
from numba import njit

"""
Description:
Use perturbation theory to estimate modes for attenuating media

Date:
2/20/2022

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



def get_attn_conv_factor(units='npm', *args):
    """
    Get conversion factor to get attnuation into correct units
    Input - 
    units - string
        options are npm, dbpm, dbplam, dbpkmhz, q
    optional_args - 
        can be wavelength (in meters) for dbplam or for Q
        can be frequency (in Hz) for dbpkmhz
    """
    if units == 'npm':
        return 1.0
    elif units == 'dbpm':
        return 0.115
    elif units == 'dbplam':
        if len(args) == 0:
            raise ValueError('Wavelength must be passed in if using dbplam')
        lam = args[0]
        return 0.115 / lam
    elif units == 'dbpkmhz':
        f = args[0]
        if len(args) == 0:
            raise ValueError('Frequency must be passed in if using dbplam')
        return f / 8686
    elif units == 'q':
        if len(args) == 0:
            raise ValueError('Wavelength must be passed in if using dbplam')
        lam = args[0]
        return np.pi / lam / Q
    else:
        raise ValueError('Invalid units passed in. options are npm, dbpm, dbplam, \
                            dbpkmhz, q')

@njit
def alpha_layer_integral(omega, krm, phim_layer, c, rho, attn, dz):
    layer_integrand = attn*omega*np.square(phim_layer) / c / rho
    integral = dz*(np.sum(layer_integrand) - .5*layer_integrand[0] - .5*layer_integrand[-1])
    alpha_layer = integral / krm
    return alpha_layer

def add_attn(omega, krs, phi, h_list, z_list, c_list, rho_list,\
                attn_list, c_hs, rho_hs, attn_hs):
    """
    Use perturbation theory to update the waveumbers krs estimated from the media without
    attenuation
    Relevant equations from JKPS are eqn. 5.177, where D is the depth of the halfspace
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
    attn_list - list of np ndarray of floats
        attenuation at each mesh depth (nepers/meter)
    c_hs - float
        halfspace speed (m/s)
    rho_hs - float
        halfpace density (g / m^3) 
    attn_hs - float
        halfspace attenuation (nepers/meter)
    """
    pert_krs = np.zeros((krs.size), dtype=np.complex128)
    num_modes = krs.size
    num_layers = len(h_list)
    for i in range(num_modes):
        layer_ind = 0
        phim = phi[:,i]
        krm = krs[i]
        alpham = 0.0
        for j in range(num_layers):
            z = z_list[j]
            num_pts = z.size
            phim_layer = phim[layer_ind:layer_ind+num_pts]
            c = c_list[j]
            rho = rho_list[j]
            attn = attn_list[j]
            dz = h_list[j]
            alpha_layer = alpha_layer_integral(omega, krm, phim_layer, c, rho, attn, dz)
            alpham += alpha_layer
            layer_ind += num_pts - 1
        gammam = np.sqrt(np.square(krm) - np.square(omega / c_hs))
        delta_alpham = np.square(phim[-1])*attn_hs*omega/(2*krm*gammam*c_hs*rho_hs)
        alpham += delta_alpham
        pert_krs[i] = krm + complex(0,1)*alpham
    pert_krs = pert_krs.conj()
    return pert_krs


from pykrak.misc import get_simpsons_integrator, get_layer_N
@njit
def get_layered_attn_integrator(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr):
    """
    Input
    h_arr - array of mesh spacings for each layer
    ind_arr - array of index of first value for each layer
    z_arr - depths of the layer meshes concatenated
    c_arr - sound speed of the layer meshes concatenated
    rho_arr - density of the layer meshes concatenated
    attn_arr - attenuation of the layer meshes concatenated

    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for i in range(len(layer_N)):
        if i < num_layers -1 :
            c_i = c_arr[ind_arr[i]:ind_arr[i+1]]
            rho_i = rho_arr[ind_arr[i]:ind_arr[i+1]]
            attn_i = attn_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            c_i = c_arr[ind_arr[i]:]
            rho_i = rho_arr[ind_arr[i]:]
            attn_i = attn_arr[ind_arr[i]:]
        integrator_i = get_simpsons_integrator(layer_N[i], h_arr[i])[0,:]
        integrator_i *= omega*attn_i / (c_i * rho_i)
        if i == 0:
            integrator = integrator_i
        else:
            integrator[-1] += integrator_i[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_i[1:]))
    return integrator


@njit
def add_arr_attn(omega, krs, phi, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs):
    integrator = get_layered_attn_integrator(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr)
    krs_i = np.zeros(krs.size)
    for m in range(krs.size):
        krm = krs[m]
        phim = phi[:,m]
        # integate through layers
        alpha = np.sum(np.square(phim) * integrator) / krm

        # add contribuition from tail
        gamma = np.sqrt(np.square(krm) - np.square(omega / c_hs))
        delta_alpha = np.square(phim[-1])*attn_hs*omega/(2*krm*gamma*c_hs*rho_hs)
        alpha += delta_alpha
        krs_i[m] = alpha

    pert_krs = krs - 1j*krs_i
    return pert_krs
