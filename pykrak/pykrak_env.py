import numpy as np
from matplotlib import pyplot as plt
from pykrak.attn_pert import add_attn, get_attn_conv_factor, get_c_imag
from pykrak import krak_routines as kr

import numba as nb

""" Description:
    This module contains the class Env, which is used to store the model parameters
    and manage the normal mode calculation
    It also contains a Modes object to store the output of a single frequency run

Date:
    4/18/2023
    Revised 7/24/2025 to use the updated KRAKEN translation

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


class Modes:
    def __init__(self, freq, krs, phi, M, z, ugs):
        """Mode output from a single frequency run"""
        self.freq = freq
        self.krs = krs
        self.phi = phi
        self.M = M
        self.z = z
        self.ugs = ugs

    def get_phi_zr(self, zr, M=None):
        """
        Interpolate modes over array depths zr
        """
        phi = self.phi
        if np.all(phi == 0):
            phi = self.get_phi()
        if M is None:
            M = self.M
        phi_zr = np.zeros((zr.size, M))
        phi_z = self.z
        for i in range(M):
            phi_zr[:, i] = np.interp(zr, phi_z, phi[:, i])
        return phi_zr


class Env:
    """
    Store the environment parameters for a layered fluid medium
    There is the option to include an elastic layer on the surface or on the bottom, with a bounding halfspace
    For example, a pressure release halfspace (vacuum), followed by elastic layer ice, followed by fluid layers
    Input:
    z_list : list of numpy arrays
        the depth of each layer in the environment (positive down)
    cp_list : list of numpy arrays
        the compressional speed of sound in each layer
    cs_list : list of numpy arrays
        the shear speed of sound in each layer (0 for fluid layers)
    rho_list : list of numpy arrays
        the density of each layer
    attnp_list : list of numpy arrays
        the compressional attenuation in each layer
    attns_list : list of numpy arrays
        the shear attenuation in each layer (0 for fluid layers)
    cp_top : float
        the compressional speed of sound in the halfspace above the layers
    cs_top : float
        the shear speed of sound in the halfspace above the layers
    rho_top : float
        the density of the halfspace above the layers (0 for pressure release, 1e10 for rigid)
    attnp_top : float
        the compressional attenuation in the halfspace above the layers (0 for pressure release)
    attns_top : float
        the shear attenuation in the halfspace above the layers (0 for pressure release)
    cp_bott : float
        the compressional speed of sound in the halfspace below the layers
    cs_bott : float
        the shear speed of sound in the halfspace below the layers
    rho_bott : float
        the density of the halfspace below the layers
    attnp_bott : float
        the compressional attenuation in the halfspace below the layers
    attn_units : str
        options are npm, dbpm, dbplam, dbpkmhz, q
        which is Nepers/meter, decibels per meter, decibels per wavelength, decibels per kilometer per hertz, or q
    sigma_arr : numpy array of floats
        rms roughness for each interface (same units as grid)
        The number of interfaces is one more than the number of layers
        0 is no roughness
    """

    def __init__(
        self,
        z_list,
        cp_list,
        cs_list,
        rho_list,
        attnp_list,
        attns_list,
        cp_top,
        cs_top,
        rho_top,
        attnp_top,
        attns_top,
        cp_bott,
        cs_bott,
        rho_bott,
        attnp_bott,
        attns_bott,
        attn_units,
        sigma_arr,
    ):
        self.z_list = z_list
        self.cp_list = cp_list
        self.cs_list = cs_list
        self.rho_list = rho_list
        self.attnp_list = attnp_list
        self.attns_list = attns_list

        self.cp_top = cp_top
        self.cs_top = cs_top
        self.rho_top = rho_top
        self.attnp_top = attnp_top
        self.attns_top = attns_top

        self.cp_bott = cp_bott
        self.cs_bott = cs_bott
        self.rho_bott = rho_bott
        self.attnp_bott = attnp_bott
        self.attns_bott = attns_bott

        self.attn_units = attn_units
        self.sigma_arr = sigma_arr

    def get_modes(self, freq, Ng_list=[], rmax=0.0, c_low=0.0, c_high=1e10):
        """
        Compute wavenumbers , mode shapes, and group speeds for the environment at
        the specified frequency
        Ng_list : list of ints
            an optional manual specification of the number of mesh points to use in each layer
            in the first calculation. The mesh is refined and used to monitor convergence of the
            wavenumbers with Richardson extrapolation, so this may not be the final mesh.
        rmax : float
            Maximum range of the source considered. The modal wavenumbers are multiplied by the source
            range in the pressure field calculation, so this is a natural way to set the numerical precision
            The mesh is refined until the difference in wavenumber from Richardson extrapolation from the previous mesh set
            times rmax, is less than 1 (i.e. a modal phase error of 1 rad)

        c_low : minimum phase speed to include in the calculation
        c_high : maximum phase speed to include in the calculation
        """
        krs, z, phi, ugs = kr.list_input_solve(
            freq,
            self.z_list,
            self.cp_list,
            self.cs_list,
            self.rho_list,
            self.attnp_list,
            self.attns_list,
            self.cp_top,
            self.cs_top,
            self.rho_top,
            self.attnp_top,
            self.attns_top,
            self.cp_bott,
            self.cs_bott,
            self.rho_bott,
            self.attnp_bott,
            self.attns_bott,
            self.attn_units,
            Ng_list,
            rmax,
            c_low,
            c_high,
            self.sigma_arr,
        )

        # pk_krs, phi_z, phi, ugs = kr.list_input_solve(freq, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_top, cs_top, rho_top, attnp_top, attns_top, cp_hs, cs_hs, rho_hs, attnp_hs, attns_hs, 'dbplam', N_list, RMax, c_low, c_high)
        return krs, z, phi, ugs

    def plot_env(self, ax=None, color=None):
        """
        Very basic plotting
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            rhofig, rhoax = plt.subplots(1, 1)
            attnfig, attnax = plt.subplots(1, 1)
        for i in range(len(self.z_list)):
            if color is None:
                ax.plot(self.cp_list[i], self.z_list[i])
                rhoax.plot(self.rho_list[i], self.z_list[i])
                attnax.plot(self.attnp_list[i], self.z_list[i])
            else:
                ax.plot(self.cp_list[i], self.z_list[i], color)
                rhoax.plot(self.rho_list[i], self.z_list[i], color)
                attnax.plot(self.attnp_list[i], self.z_list[i], color)
        ax.set_ylim([self.z_list[-1][-1] + 50, 0])
        rhoax.set_ylim([self.z_list[-1][-1] + 50, 0])
        attnax.set_ylim([self.z_list[-1][-1] + 50, 0])
        for i in range(len(self.z_list)):
            ax.hlines(
                self.z_list[i][-1],
                min([x.min() for x in self.cp_list]),
                max([x.max() for x in self.cp_list]),
                "k",
                alpha=0.5,
            )
            rhoax.hlines(
                self.z_list[i][-1],
                min([x.min() for x in self.rho_list]),
                max([x.max() for x in self.rho_list]),
                "k",
                alpha=0.5,
            )
            attnax.hlines(
                self.z_list[i][-1],
                min([x.min() for x in self.attnp_list]),
                max([x.max() for x in self.attnp_list]),
                "k",
                alpha=0.5,
            )
        ax.hlines(
            self.z_list[-1][-1],
            min([x.min() for x in self.cp_list]),
            max([x.max() for x in self.cp_list]),
            "k",
        )
        rhoax.hlines(
            self.z_list[-1][-1],
            min([x.min() for x in self.rho_list]),
            max([x.max() for x in self.rho_list]),
            "k",
        )
        attnax.hlines(
            self.z_list[-1][-1],
            min([x.min() for x in self.attnp_list]),
            max([x.max() for x in self.attnp_list]),
            "k",
        )
        mean_layer_c = sum([x.mean() for x in self.cp_list]) / len(self.cp_list)
        ax.text(
            mean_layer_c,
            self.z_list[-1][-1] + 30,
            "$c_b$:{0}, \n$\\rho_b$:{1}, \n$\\alpha_b$:{2}".format(
                self.cp_bott, self.rho_bott, self.attnp_bott
            ),
        )
        return


class FluidEnv(Env):
    def __init__(
        self,
        z_list,
        cp_list,
        rho_list,
        attnp_list,
        cp_top,
        rho_top,
        attnp_top,
        cp_bott,
        rho_bott,
        attnp_bott,
        attn_units,
    ):
        cs_list = [np.zeros(x.size) for x in cp_list]
        attns_list = [np.zeros(x.size) for x in attnp_list]  # elastic parameters are 0
        cs_top = 0.0
        attns_top = 0.0
        cs_bott = 0.0
        attns_bott = 0.0
        sigma_arr = np.zeros(len(z_list) + 1)

        super().__init__(
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            cp_top,
            cs_top,
            rho_top,
            attnp_top,
            attns_top,
            cp_bott,
            cs_bott,
            rho_bott,
            attnp_bott,
            attns_bott,
            attn_units,
            sigma_arr=sigma_arr,
        )
