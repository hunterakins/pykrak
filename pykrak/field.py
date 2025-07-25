"""
Description:
Routines for calculating pressure with various input arguments and in various cases
Has some overkill modal Doppler if you have a fast moving source
The convention is that the source has time dependence of e^{i \omega t},
so e^{-ikr} is the outward spreading wave

get_pressure is for a single source range, and whatever depths phi_zr is sampled at
get_vec_pressure is for multiple source ranges


Date:
8.3.2022

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
from pykrak.envs import factory
from numba import njit, prange
import numba as nb


@njit(cache=True, parallel=True)
def get_phi_zr(zr, phi_z, phi):
    """
    Given array of points zr
    zr supposed to be sorted already!
    Linearly interpolate the function phi(z) at points zr
    phi_z is sorted but not necessarily uniformly spaced
    """
    M = phi.shape[1]
    phi_zr = np.zeros((zr.size, M))
    phi_z_ind = 0
    for i in range(zr.size):
        zri = zr[i]
        while phi_z[phi_z_ind] <= zri:
            phi_z_ind += 1
        if phi_z[i] == zri:  # zr is in the mesh
            phi_zr[i] = phi[phi_z_ind, :]
            phi_z_ind += 1
        else:  # linearly interp
            s = (zri - phi_z[phi_z_ind - 1]) / (phi_z[phi_z_ind] - phi_z[phi_z_ind - 1])
            phi_zr[i, :] = phi[phi_z_ind - 1, :] + s * (
                phi[phi_z_ind, :] - phi[phi_z_ind - 1, :]
            )
    return phi_zr


@njit
def get_delta_R(tilt_angle, zr):
    """
    Positive angle means it's leaning towards the source,
    the lowest (deeper) element of the array is always located at a range r
    (so it's sort of the fixed point)
    Positive angle in DEGREES
    """
    Z = np.max(zr) - zr  # this gives depth above bottom element  (so Z<0)
    deltaR = Z * np.tan(
        tilt_angle * np.pi / 180.0
    )  # tilt_angle > 0 \implies tan(tilt_angle) > 0 , so delta R < 0
    deltaR = deltaR.reshape(zr.size, 1)
    return deltaR


@njit(cache=True, parallel=True)
def get_no_tilt_pressure(phi_zr, phi_zs, krs, r):
    """
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at receiver depths,
    evaluated at source depth,
    wavenumbers kr, and source range r

    Input -
    phi_zr - np 2d array
        jth column is jth mode evaluated at receiver depths
    phi_zs - np 2d array
        jth column is jth mode evaluated at source depth
        only one row
    krs - np 1d array
        wavenumbers
    r - float
        source receiver range
    deltaR
        np 2d array: correction to range for array tilt/deformation
    Return pressure as column vector
    """
    modal_matrix = phi_zr * phi_zs[0, :]
    range_dep = np.exp(-1j * r * krs) / np.sqrt(krs.real * r)
    N = phi_zr.shape[0]
    p = np.zeros((N, 1), dtype=nb.c16)
    for k in prange(N):
        mode_contrib = np.sum(modal_matrix[k, :] * range_dep)
        p[k, 0] = mode_contrib
    p *= 1j * np.exp(1j * np.pi / 4)  # assumes rho is 1 at source depth
    p /= np.sqrt(8 * np.pi)
    return p


@njit(cache=True)
def get_arr_pressure(phi_zs, phi_zr, krs, r_arr, deltaR=np.array([0.0])):
    """
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at source and receiver depths
    wavenumbers kr, and source ranges r

    Input -
    phi_zs - np 2d array
        first index is source depth index, second is mode number
    phi_zr - np 2d array
        first index is receiver depth index, second is mode number
    krs - np 1d array
        wavenumbers
    r_arr - np 1d array
        range between source array and receive array
    deltaR
        np 1d array: correction to range for array tilt/deformation
        either single number or a different number for each receiver depth
    Return pressure as column vector
    """
    Nzr = phi_zr.shape[0]
    Nzs = phi_zs.shape[0]
    M = phi_zr.shape[1]
    phi_zs = np.reshape(phi_zs, (Nzs, 1, M))
    phi_zr = np.reshape(phi_zr, (1, Nzr, M))
    modal_matrix = phi_zs * phi_zr
    if deltaR.size == 1:  # constant offset
        r_arr = r_arr + deltaR
        r_arr = np.reshape(r_arr, (1, r_arr.size))
        krs = np.reshape(krs, (M, 1))
        range_dep = np.exp(-1j * r_arr * krs) / np.sqrt(
            krs.real * r_arr
        )  # num modes x num ranges
        # expand for receiver depth and source depth
        range_dep = np.reshape(
            range_dep, ((1, 1, range_dep.shape[0], range_dep.shape[1]))
        )
        modal_matrix = np.reshape(
            modal_matrix,
            (modal_matrix.shape[0], modal_matrix.shape[1], modal_matrix.shape[2], 1),
        )
        prod = modal_matrix * range_dep
        # sum over modes
        p = np.sum(prod, axis=2)
    else:
        # expand for each mode
        r_arr = np.reshape(r_arr, (1, r_arr.size))
        krs = np.reshape(krs, (krs.size, 1))
        range_dep = np.exp(-1j * r_arr * krs) / np.sqrt(
            krs.real * r_arr
        )  # num modes x num ranges

        # expand for receiver depth and source depth
        range_dep = np.reshape(
            range_dep, ((1, 1, range_dep.shape[0], range_dep.shape[1]))
        )
        modal_matrix = np.reshape(
            modal_matrix,
            (modal_matrix.shape[0], modal_matrix.shape[1], modal_matrix.shape[2], 1),
        )
        prod = modal_matrix * range_dep
        # sum over modes
        p = np.sum(prod, axis=2)
        corr_phase = np.exp(+1j * np.mean(krs) * deltaR)
        corr_phase = np.reshape(corr_phase, (1, Nzr, 1))
        p *= corr_phase
    p *= 1j * np.exp(1j * np.pi / 4)  # assumes rho is 1 at source depth
    p /= np.sqrt(8 * np.pi)
    return p


@njit(cache=True)
def get_pressure(phi_zr, phi_zs, krs, r, deltaR=np.array([0.0])):
    """
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at receiver depths,
    evaluated at source depth,
    wavenumbers kr, and source range r

    Input -
    phi_zr - np 2d array
        jth column is jth mode evaluated at receiver depths
    phi_zs - np 2d array
        first index is a dummy rcvr index
        jth column is jth mode evaluated at source depth
    krs - np 1d array
        wavenumbers
    r - float
        source receiver range
    deltaR
        np 2d array: correction to range for array tilt/deformation
    Return pressure as column vector
    """
    modal_matrix = phi_zs * phi_zr
    if deltaR.size == 1:  # constant offset
        r = r - deltaR
        range_dep = np.exp(-1j * r * krs) / np.sqrt(krs.real * r)
        N = phi_zr.shape[0]
        p = np.zeros((N, 1), dtype=nb.c16)
        for k in prange(N):
            mode_contrib = np.sum(modal_matrix[k, :] * range_dep)
            p[k, 0] = mode_contrib
    else:
        # r = r - deltaR #  minus so that it leans towards the source (range gets closer)
        krs = krs.reshape(1, krs.size)
        range_arg = krs * r
        range_dep = np.exp(-1j * range_arg) / np.sqrt(range_arg.real)
        prod = modal_matrix * range_dep
        p = np.sum(prod, axis=1)
        p = p.reshape(p.size, 1)
        corr_phase = np.exp(-1j * np.mean(krs) * deltaR)
        p *= corr_phase
    p *= 1j * np.exp(1j * np.pi / 4)  # assumes rho is 1 at source depth
    p /= np.sqrt(8 * np.pi)
    return p


@njit
def get_vec_pressure(phi_zr, phi_zs, krs, rgrid, tilt_angles=None, zr=None):
    """
    Designed for source tracks, not replica calculation
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at receiver depths,
    wavenumbers kr,
    and a vector of source depths, ranges, and tilts
    Return pressure as array zr.size, num_pts in track
    Positive angle means it's leaning towards the source,
    the lowest element of the array is always located at a range r
    (so it's sort of the fixed point)
    Positive angle in DEGREES
    It's approximate in that it doesn't correct the range spreading term from tilt,
    only the phase
    """
    num_track_pts = rgrid.size
    num_depths = phi_zr.shape[0]
    full_p = np.zeros((num_depths, num_track_pts), dtype=np.complex128)
    for i in range(num_track_pts):
        modal_matrix = phi_zr * phi_zs[:, i]
        modal_matrix = modal_matrix.astype(np.complex128)
        r = rgrid[i]
        if tilt_angles is not None:
            tilt_angle = tilt_angles[i]
            Z = np.max(zr) - zr
            deltaR = Z * np.tan(tilt_angle * np.pi / 180.0)
            deltaR = deltaR.reshape(zr.size, 1)
            delta_phi = np.exp(-1j * np.outer(deltaR, krs))
            modal_matrix *= delta_phi
        arg = krs.reshape(krs.size, 1) * r
        range_dep = np.exp(-1j * arg) / np.sqrt(arg.real)
        p = modal_matrix @ range_dep
        full_p[:, i] = p[:, 0]
    full_p *= 1j * np.exp(1j * np.pi / 4)
    full_p /= np.sqrt(8 * np.pi)
    return full_p


def get_range_correction(zr, tilt_angle):
    """
    For receiving array tilted at angle tilt_angle (degrees)
    towards the source, compute the range correction for each element in zr
    A negative angle means that the top element is closer to the source than the bottom element
    (leaning forward)
    Positive means its further away (leaning back)
    Input
    zr - np array
    ASSUMED SORTED FROM SMALLEST TO LARGEST (shallowest to deepest)
    tilt_angle - int or float in degrees
    Output
    r_corr - np array same shape as zr
    If the array is at the origin (r=0), then r_corr gives their
    corrected locations relative to the origin
    """
    tilt_rad = np.pi / 180 * tilt_angle
    zr_rel = zr - zr[0]
    r_corr = np.tan(tilt_rad) * zr_rel
    r_corr -= np.mean(r_corr)
    return r_corr


def get_grid_pressure(zr, phi_z, phi, krs, zgrid, rgrid, tilt_angle=None):
    """
    Get the field grid for pressure for receivers at zr
    Using the modes phi evaluated at the grid depths z
    For sources at all positions in the grids zgrid and rgrid
    zr - np 1d array
    phi_z - np 1d array
    phi - np 2d array
        first axis is depth, second axis is mode
    krs - np 1d array
        horizontal wavenumbers
    zgrid - np 1d array
        candidate source depths
    rgrid - np 1d array
        candidate source ranges
    tilt_angles - np 1d array
        tilt angles of the array towards the source
    Return
    pfield - np 3d array
        First axis is receiver depth, second axis is source depth, third axis is source range
    Unnormalized, so really a field calculation
    """
    Nzr = zr.size
    if tilt_angle is not None:
        r_corr = get_range_correction(zr, tilt_angle)
    else:
        r_corr = np.zeros(zr.size)
    pfield = np.zeros((Nzr, zgrid.size, rgrid.size), dtype=np.complex128)

    # interpolate the modes to the grid depths (they are ``receivers'' in reciprocity)
    phi_zr = np.zeros((zgrid.size, krs.size))
    for l in range(krs.size):
        phi_zr[:, l] = np.interp(zgrid, phi_z, phi[:, l])

    # for each receiver depth
    for i in range(Nzr):
        # use reciprocity to get modal field everywhere
        zs = zr[i]
        strength = np.zeros((1, krs.size))
        # get mode value at ``source'' (receiver) depth
        for l in range(krs.size):
            strength[0, l] = np.interp(np.array(zs), phi_z, phi[:, l])
        modal_matrix = strength * phi_zr
        modal_matrix /= np.sqrt(krs.real)
        r_mat = np.outer(krs, rgrid - r_corr[i])
        range_dep = np.exp(-1j * r_mat) / np.sqrt(r_mat.real)
        source_p = modal_matrix @ range_dep
        source_p *= 1j * np.exp(1j * np.pi / 4)
        source_p /= np.sqrt(8 * np.pi)
        pfield[i, ...] = source_p
    return pfield


def get_doppler_vec_pressure(
    phi_zr,
    phi_zs,
    krs,
    r_ret_grid,
    t_ret_grid,
    ums,
    contemp_tgrid,
    tilt_angles=None,
    zr=None,
):
    """
    Use modal Doppler theory get pressure due to source moving along track associated with rgrid
    sample on t_ret_grid retarded time points
    The field is the usual field but using the range r_{s}(t') (the range in the retarded time)
    To get the field in the contemporary time, first find out which time samples I have the field
    using the relationship t = t' + r_{s}(t') / u_{m} , where t is contemporary time, t' is retarded
    time, r_{s}(t') is the range of the source in retarded time, and u_{m} is the group speed of the     mth mode
    Once I have these samples, I put them onto a uniform time grid
    The way I handle tilt is a little inaccurate because I don't use the doppler shifted
    modes to get the range offsets (and I don't change the range spreading appropriately)
    """
    rgrids = np.zeros((krs.size, contemp_tgrid.size))
    phi_zs_dopp = np.zeros((krs.size, contemp_tgrid.size))
    for i in range(krs.size):
        tgrid = (
            t_ret_grid + r_ret_grid / ums[i]
        )  # the contemporary time points associated with my sampled retarded time ranges
        rgrid = np.interp(contemp_tgrid, tgrid, r_ret_grid)
        rgrids[i, :] = rgrid
        phi_zs_mode = np.interp(contemp_tgrid, tgrid, phi_zs[i, :])
        phi_zs_dopp[i, :] = phi_zs_mode

    Z = np.max(zr) - zr

    num_track_pts = contemp_tgrid.size
    num_zrs = phi_zr.shape[0]
    full_p = np.zeros((num_zrs, num_track_pts), dtype=np.complex128)
    for j in range(zr.size):
        modal_matrix = phi_zr[j, :].reshape(krs.size, 1) * phi_zs_dopp
        modal_matrix = modal_matrix.astype(np.complex128)
        if tilt_angles is not None:
            deltaR = Z[j] * np.tan(tilt_angles * np.pi / 180.0)
            zr_rgrids = rgrids + deltaR
        else:
            zr_rgrids = rgrids
        arg = krs.reshape(krs.size, 1) * zr_rgrids
        range_dep = np.exp(-1j * arg) / np.sqrt(arg.real)
        p = np.sum(modal_matrix * range_dep, axis=0)
        full_p[j, :] = p
    full_p *= 1j * np.exp(1j * np.pi / 4)
    full_p /= np.sqrt(8 * np.pi)
    return full_p


@njit
def get_doppler_vec_pressure_deriv(
    phi_zr,
    phi_zs,
    dphir_dk,
    dphis_dk,
    kr,
    r_ret_grid,
    t_ret_grid,
    um,
    contemp_tgrid,
    tilt_angles=None,
    zr=None,
):
    """ """
    tgrid = (
        t_ret_grid + r_ret_grid / um
    )  # the contemporary time points associated with my sampled retarded time ranges
    rgrid = np.interp(contemp_tgrid, tgrid, r_ret_grid)
    phi_zs_dopp = np.interp(contemp_tgrid, tgrid, phi_zs[0, :])
    dphis_dk_dopp = np.interp(contemp_tgrid, tgrid, dphis_dk[0, :])

    Z = np.max(zr) - zr

    num_track_pts = contemp_tgrid.size
    num_zrs = phi_zr.shape[0]
    full_deriv = np.zeros((num_zrs, num_track_pts), dtype=np.complex128)

    for j in range(zr.size):
        A = phi_zr[j, 0] * phi_zs_dopp
        if tilt_angles is not None:
            deltaR = Z[j] * np.tan(tilt_angles * np.pi / 180.0)
            zr_rgrid = rgrid + deltaR
        else:
            zr_rgrid = rgrid
        arg = kr * zr_rgrid
        B = np.exp(-1j * arg) / np.sqrt(arg.real)
        dBdk = -1j * zr_rgrid.real * B
        dAdk = dphir_dk[j] * phi_zs_dopp + phi_zs_dopp * dphis_dk_dopp
        # print(np.linalg.norm(A*dBdk), np.linalg.norm(dAdk*B))
        full_deriv[j, :] = A * dBdk + dAdk * B
    full_deriv *= np.exp(1j * np.pi / 4)
    full_deriv /= np.sqrt(8 * np.pi)
    return full_deriv


@njit
def get_simple_doppler_vec_pressure(
    phi_zr, phi_zs, krs, rgrid, tgrid, ums, tilt_angles=None, zr=None
):
    """
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at receiver depths,
    wavenumbers kr,
    and a vector of source depths, ranges, and tilts
    Return pressure as array zr.size, num_pts in track
    Positive angle means it's leaning towards the source,
    the lowest element of the array is always located at a range r
    (so it's sort of the fixed point)
    Positive angle in DEGREES
    It's approximate in that it doesn't correct the range spreading term from tilt,
    only the phase
    """
    num_track_pts = rgrid.size
    num_depths = phi_zr.shape[0]
    full_p = np.zeros((num_depths, num_track_pts), dtype=np.complex128)
    dt = tgrid[1] - tgrid[0]
    for i in range(num_track_pts):
        modal_matrix = phi_zr * phi_zs[:, i]
        modal_matrix = modal_matrix.astype(np.complex128)
        r = rgrid[i]
        if i == 0:
            v = (rgrid[1] - rgrid[0]) / dt
        elif i == num_track_pts - 1:
            v = (rgrid[-1] - rgrid[-2]) / dt
            print("v", v)
        else:
            v = 0.5 * (rgrid[i + 1] - rgrid[i - 1]) / dt
            print("v", v)
        krsi = krs / (1 + v / ums)  # doppler shifted krs...
        if tilt_angles is not None:
            tilt_angle = tilt_angles[i]
            Z = np.max(zr) - zr
            deltaR = Z * np.tan(tilt_angle * np.pi / 180.0)
            deltaR = deltaR.reshape(zr.size, 1)
            delta_phi = np.exp(-1j * np.outer(deltaR, krsi))
            modal_matrix *= delta_phi
        arg = krsi.reshape(krsi.size, 1) * r
        range_dep = np.exp(-1j * arg) / np.sqrt(arg.real)
        p = modal_matrix @ range_dep
        full_p[:, i] = p[:, 0]
    full_p *= np.exp(1j * np.pi / 4)
    full_p /= np.sqrt(8 * np.pi)
    return full_p


def quick_tilt_test():
    env = factory.create("swellex")()
    zr = np.linspace(100, 200, 30)
    zs = 50.0
    freq = 100.0
    env.add_freq(freq)
    krs = env.get_krs(**{"cmax": 1800.0})
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(np.array(zs))
    r = np.arange(1000, 1500, 0.1)
    phi_zs = phi_zs.T * np.ones(r.size)
    tilt_angles = 45 * (1500 - r) / 500

    p = get_vec_pressure(phi_zr, phi_zs, krs, r)
    p1 = get_vec_pressure(phi_zr, phi_zs, krs, r, tilt_angles, zr=zr)
    p1 = get_vec_pressure(phi_zr, phi_zs, krs, r, tilt_angles, zr=zr)
    import time

    now = time.time()
    p1 = get_vec_pressure(phi_zr, phi_zs, krs, r, tilt_angles, zr=zr)
    print("time to run", time.time() - now)

    plt.figure()
    plt.pcolormesh(r, zr, abs(p))
    for j in range(r.size):
        if abs(r[j] % 100) < 1e-8:
            print(r[j])
            for i in range(zr.size):
                delta_r = (zr[-1] - zr) * np.tan(tilt_angles[j] * np.pi / 180)
                plt.plot(r[j] + delta_r[i], zr[i], "k+")
    plt.gca().invert_yaxis()
    plt.figure()
    plt.pcolormesh(r, zr, abs(p1))
    plt.gca().invert_yaxis()
    plt.show()


def quick_dopp_test():
    """Compare gridded numerical solution to uniform exact solution"""
    env = factory.create("swellex")()
    zr = np.linspace(100, 200, 30)
    zs = 50.0
    freq = 100
    env.add_freq(freq)
    krs = env.get_krs(**{"cmax": 1800.0})
    ums = env.get_ugs()
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(np.array(zs))
    t_ret_grid = np.arange(0, 10, 0.001)  # 4 seconds sampling for ten mintues

    t0 = t_ret_grid[0] + r_ret_grid[0] / np.min(
        ums
    )  # time of last mode arrival at t'=0
    tf = t_ret_grid[-1] + r_ret_grid[-1] / np.max(
        ums
    )  # time of first mode arrival at t' = -1
    tgrid = np.linspace(t0, tf, t_ret_grid.size)  # contemporary time

    r0 = 1000
    v = 500.0
    r_ret = r0 + v * t_ret_grid

    phi_zs = phi_zs.T * np.ones(r_ret.size)
    tilt0 = 1.0
    dtilt = 0.001
    tilt_angles = tilt0 + dtilt * t_ret_grid

    p = get_doppler_vec_pressure(phi_zr, phi_zs, krs, r_ret, t_ret_grid, ums, unif_grid)
    print(tgrid[0])

    r_grid = r0 + v * tgrid
    p1 = get_vec_pressure(phi_zr, phi_zs, krs, r_grid)
    kr_factors = 1 / (1 + v / ums)
    krs = krs * kr_factors
    p2 = get_vec_pressure(phi_zr, phi_zs, krs, r_grid)

    plt.figure()
    plt.pcolormesh(tgrid, zr, abs(p))
    plt.gca().invert_yaxis()

    plt.figure()
    plt.pcolormesh(t_ret_grid, zr, abs(p1))
    plt.gca().invert_yaxis()

    plt.figure()
    plt.pcolormesh(t_ret_grid, zr, abs(p2))
    plt.gca().invert_yaxis()

    plt.figure()
    plt.pcolormesh(t_ret_grid, zr, abs(p2 - p))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()


def acc_dopp_test():
    env = factory.create("swellex")()
    zr = np.linspace(100, 200, 30)
    zs = 50.0
    freq = 100.0
    env.add_freq(freq)
    krs = env.get_krs(**{"cmax": 1800.0})
    ums = env.get_ugs()
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(np.array(zs))

    t_ret_grid = np.arange(0, 600, 4.0)  # 4 seconds sampling for ten mintues

    r0 = 1000
    v = 2.0
    alpha = 1 / 600
    r_ret = r0 + v * t_ret_grid + 0.5 * alpha * np.square(t_ret_grid)
    plt.plot(t_ret_grid, r_ret)
    plt.show()

    phi_zs = phi_zs.T * np.ones(r_ret.size)
    tilt0 = 1.0
    dtilt = 0.001
    tilt_angles = tilt0 + dtilt * t_ret_grid

    tgrid, p = get_doppler_vec_pressure(phi_zr, phi_zs, krs, r_ret, t_ret_grid, ums)
    print(tgrid[0])

    r_grid = r0 + v * tgrid + 0.5 * alpha * np.square(tgrid)
    mean_v = v + alpha * np.mean(tgrid)
    kr_factors = 1 / (1 + mean_v / ums)
    krs = krs * kr_factors
    p1 = get_vec_pressure(phi_zr, phi_zs, krs, r_grid)
    plt.figure()
    plt.pcolormesh(abs(p1 - p) / abs(p))
    plt.colorbar()
    plt.suptitle(
        "Difference between Doppler solution and simply using the ranges with a mean correction to the wavenumbers"
    )
    plt.show()

    plt.figure()
    plt.suptitle("Same difference, but incoherent difference")
    plt.pcolormesh((abs(p1) - (p)) / abs(p))
    plt.colorbar()
    plt.show()


def two_dopp_comp():
    env = factory.create("swellex")()
    zr = np.linspace(100, 200, 60)
    zs = 50.0
    freq = 400.0
    env.add_freq(freq)
    krs = env.get_krs(**{"cmax": 1800.0})
    ums = env.get_ugs()
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(np.array(zs))

    t_ret_grid = np.arange(0, 600, 2.7)  # 4 seconds sampling for ten mintues
    t_grid = np.arange(2.7, 600, 2.7)
    print(t_grid[1] - t_grid[0], t_grid[-1] - t_grid[-2])
    tilt_angles = np.zeros((t_grid.size))

    r0 = 1000
    v = 2.0
    alpha = 1 / 600
    r_ret = r0 + v * t_ret_grid + 0.5 * alpha * np.square(t_ret_grid)
    r = r_ret[1:].copy()

    phi_zs_track = env.get_phi_zr(np.array([zs] * len(t_ret_grid)))
    import time

    dopp_p = get_doppler_vec_pressure(
        phi_zr,
        phi_zs_track.T,
        krs,
        r_ret,
        t_ret_grid,
        ums,
        t_grid,
        tilt_angles=tilt_angles,
        zr=zr,
    )
    now = time.time()
    for i in range(10):
        dopp_p = get_doppler_vec_pressure(
            phi_zr,
            phi_zs_track.T,
            krs,
            r_ret,
            t_ret_grid,
            ums,
            t_grid,
            tilt_angles=tilt_angles,
            zr=zr,
        )
    d_t = (time.time() - now) / 10
    simple_dopp_p = get_simple_doppler_vec_pressure(
        phi_zr, phi_zs_track.T, krs, r, t_grid, ums, tilt_angles=tilt_angles, zr=zr
    )
    now = time.time()
    for i in range(10):
        simple_dopp_p = get_simple_doppler_vec_pressure(
            phi_zr, phi_zs_track.T, krs, r, t_grid, ums, tilt_angles=tilt_angles, zr=zr
        )
    s_t = (time.time() - now) / 10
    print("dt ,st", d_t, s_t)

    plt.figure()
    plt.pcolormesh(dopp_p.real)
    plt.figure()
    plt.pcolormesh(simple_dopp_p.real)
    plt.show()

    plt.plot(abs(dopp_p - simple_dopp_p))
    plt.plot(abs(dopp_p - simple_dopp_p))
    plt.plot(abs(dopp_p - simple_dopp_p))
    plt.plot(abs(dopp_p), "k")
    plt.show()


if __name__ == "__main__":
    two_dopp_comp()
    acc_dopp_test()
    quick_dopp_test()
    quick_tilt_test()
