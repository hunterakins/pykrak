"""
Description:
Computing the pressure field from the output of the modal solver

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
from numba import njit, prange
import numba as nb
from pykrak import interp

@njit
def get_pressure_core(krs, zmesh, phimesh, zs_arr, zr_arr, rr_arr, rr_offset, freq, c_ref, beam_pattern, M):
    """
    Compute the pressure field produced by each source at the depth specificed in zs_arr at the origin
    the field is evalued at each point in the direct product of values in zr_arr and rr_arr
    rr_offset is the same size as zr_arr and is a small correction to the range (smaller than resolution cell of your problem)

    Input:
    krs - np 1d array of complex floats
        horizontal wavenumbers
    zmesh - np 1d array
        depths at which the modes are evaluated
    phimesh - np 2d array
        modes evaluated at depths zmesh
        first axis is depth, second axis is mode number
    zs_arr - float or np 1d array
    zr_arr - float or np 1d array
    rr_arr - float or np 1d array
        ranges at which the field is evaluated
    rr_offset - float or np 1d array
        An additional range offset for each receiver depth. 
    c_ref - float
        Reference speed at source depth for beam pattern calculation
    beam_pattern - np 2d array
        First column is angle in degrees
        Second column is beam pattern value at that angle
    Mlim - int
        to use a limited number of modes in the calculation
    Return:
    field - np 3d array
        First axis is source depth, second axis is receiver depth, third axis is range

    Assumes the 'Electrical engineering' convention, not the 'Physicist' Fourier transform convention.
    That is,
    S(omega) = \int s(t) e^{-i omega t} dt
    This is also the convention used by MATLAB and numpy fft libraries.

    This is opposite to the convention used the Ch. 5 notation in Computational Ocean Acoustics (JKPS) 2011
    In particular eq. 5.14

    """
    Ns = zs_arr.size
    Nzr = zr_arr.size
    Nrr = rr_arr.size
    beam_pattern = np.asarray(beam_pattern)
    
    if rr_offset.size == 1: # scalar input
        rr_offset = rr_offset * np.ones(Nzr)


    field = np.zeros((Ns, Nzr, Nrr), dtype=np.complex128)

    #phi_zs = phi_interp(zs_arr, zmesh, phimesh)
    phi_zs = interp.vec_vec_lin_int(zs_arr, zmesh, phimesh)

    #phi_zr = phi_interp(zr_arr, zmesh, phimesh)
    phi_zr = interp.vec_vec_lin_int(zr_arr, zmesh, phimesh)

    modal_term = np.zeros((M, Nrr), dtype=np.complex128)
    for m in range(M):
        krm = krs[m]
        phase = krm * rr_arr
        modal_term[m,:] = np.exp(-1j * krm * rr_arr) / np.sqrt(krm.real * rr_arr)

    for i_s in range(Ns):
        phi_zs_i = phi_zs[i_s, :]
        C_i = phi_zs_i #

        if beam_pattern.shape[0] > 0: # apply beam pattern
            beam_angles = beam_pattern[:,0]
            beam_values = beam_pattern[:,1]
            omega = 2*np.pi*freq
            kz2 = np.real(omega**2 / c_ref**2 - krs**2)  # vertical wavenumber squared
            kz2 = np.maximum(kz2, 0)

            thetaT = np.rad2deg(np.arctan(np.sqrt(kz2) / krs.real))  # calculate the mode angle in degrees
            S = np.interp(thetaT, beam_angles, beam_values) # shading
            S = np.real(S)
            C_i = C_i * S  # apply the shading

        for i_rz in range(Nzr):
            phi_zr_i = phi_zr[i_rz, :]
            total = np.zeros((Nrr), dtype=np.complex128)
            for m in range(M):# sum over modes
                modal_term_m = C_i[m] * phi_zr_i[m] * modal_term[m, :]  
                # add range offset
                offset_phase = np.exp(-1j * krs[m] * rr_offset[i_rz])
                modal_term_m = offset_phase * modal_term_m
                total += modal_term_m
            field[i_s, i_rz, :] = total
    field  = field * np.conj(1j * np.exp(-1j * np.pi/4) / np.sqrt(8*np.pi)) # conj. here because JKPS use difft F.T. convention, see note baove
    return field


def get_pressure(krs, zmesh, phimesh, zs, zr, rr, rr_offset, freq, c_ref=0.0, beam_pattern=0.0, Mlim=10000000):
    M = min(krs.size, Mlim)
    zs_arr = np.asarray(zs)
    Ns = zs_arr.size
    zr_arr = np.asarray(zr)
    Nzr = zr_arr.size
    rr_arr = np.asarray(rr)
    Nrr = rr_arr.size
    rr_offset = np.asarray(rr_offset)
    beam_pattern = np.asarray(beam_pattern)
    
    if rr_offset.ndim == 0: # scalar input
        rr_offset = rr_offset * np.ones(Nzr)
    if zs_arr.ndim == 0: # scalar input
        zs_arr = zs_arr * np.ones(Ns)
    if zr_arr.ndim == 0: # scalar input
        zr_arr = zr_arr * np.ones(Nzr)
    if rr_arr.ndim == 0: # scalar input
        rr_arr = rr_arr * np.ones(Nrr)

    if beam_pattern.ndim == 0:
        beam_pattern = np.empty((0, 2), dtype=np.float64)

    field = get_pressure_core(krs, zmesh, phimesh, zs_arr, zr_arr, rr_arr, rr_offset, freq, c_ref, beam_pattern, M)
    return field
