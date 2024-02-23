"""
Description:
Routines for coupled mode field calculation
Follows the one-way single scattering approach in JKPS (equation 5.264)
Note that this differs from KRAKEN, which I believe only enforces continuity of pressure. 
This approach here enforces both continuity of pressure and of radial particle velocity.

Has not been tested

Date:
1/8/2023

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
from interp import interp
from numba import jit, njit
from pykrak.attn_pert import get_c_imag

@njit
def advance_a(a0, krs0, r0, r1):
    """
    Advance the mode amplitudes to the new interface at 
    r=r1 from their value at r=r0
    """
    range_dep = np.exp(-1j * krs0 * (r1 - r0)) #* np.sqrt(r0 / r1)
    a0_adv = a0*range_dep
    return a0_adv

@njit
def compute_p_left(a0, krs0, phi0, r0, r1):
    """
    Given amplitudes a0 of modes at range r0
    wavenumbers and modes in krs0 and phi0
    defined in the region between r0 and r1

    Compute the pressure field at the interface range r1

    Note here that the convention is that the range dependence does not 
    include the 1/sqrt(km) terms
    This must be accounted for in the initial condition for the amplitudes 
    in the first segment
    """
    a0_adv = advance_a(a0, krs0, r0, r1) 
    weighted_modes = a0_adv * phi0
    p_left = np.sum(weighted_modes, axis=1)
    return p_left

@njit
def compute_p_weighted_left(a0, krs0, phi0, r0, r1):
    """
    Given amplitude a0 of modes at range r0
    wavenumbers and modes in krs0 and phi0
    defined in the region between r0 and r1

    Compute the weighted pressure field required to enforce continuity
    of radial particle velocity at the interface range r1
    """
    tmp_a0 = a0 * krs0
    p_weighted_left = compute_p_left(tmp_a0, krs0, phi0, r0, r1)
    return p_weighted_left

@njit
def get_on_new_grid(z0, z1, rho0, rho1, rho_hs0, rho_hs1, phi0, phi1, gammas0, gammas1):
    """
    The grids from a mode run end at the lower halfspace which may differ from segment to segment
    This function interpolates the modes and densities onto the same grid using the 
    exponentially decaying tail
    of the modes in the halfspace to extend the shallower grid
    """
    Z1 = np.max(z1)
    Z0 = np.max(z0)

    if Z1 > Z0: # need to extend modes from z0
        z_extra = z1[z1 > Z0]
        rho_extra = rho_hs0 * np.ones(z_extra.size)
        tg = np.reshape(gammas0, (1, gammas0.size))
        tz = np.reshape(z_extra, (z_extra.size, 1))
        phi_extra = np.exp(-tg * (tz - Z0))* phi0[-1,:]
        new_z0 = np.concatenate((z0, z_extra)) # be careful here because arrays are mutable...
        # this approach should maintain contiguity
        new_phi0 = np.zeros((new_z0.size, phi0.shape[1]))
        new_phi0[:z0.size, :] = phi0.copy()
        new_phi0[z0.size:, :] = phi_extra.copy()
        z0 = new_z0
        phi0 = new_phi0
        rho0 = np.concatenate((rho0, rho_extra))
    elif Z0 > Z1: # need to extend modes from z1
        z_extra = z0[z0 > Z1]
        rho_extra = rho_hs1 * np.ones(z_extra.size)
        tg = np.reshape(gammas1, (1, gammas1.size))
        tz = np.reshape(z_extra, (z_extra.size, 1))
        phi_extra = np.exp(-tg * (tz - Z1))* phi1[-1,:]
        z1 = np.concatenate((z1, z_extra))
        new_phi1 = np.zeros((z1.size, phi1.shape[1]))
        new_phi1[:phi1.shape[0], :] = phi1.copy()
        new_phi1[phi1.shape[1]:, :] = phi_extra.copy()
        phi1 = new_phi1

        rho1 = np.concatenate((rho1, rho_extra))

    # now both grids extend to same depths
    rho0_new = interp.vec_lin_int(z1, z0, rho0)
    phi0_new = np.zeros((z1.size, phi0.shape[1]))
    for i in range(phi0.shape[1]):
        phi0_new[:,i] = interp.vec_lin_int(z1, z0, phi0[:,i])
    return z1, rho0_new, phi0_new, rho1, phi1

@njit
def update_a0(a0, krs0, gammas0, phi0, z0, rho0, rho_hs0, krs1, gammas1, phi1, z1, rho1, rho_hs1, r0, r1, same_grid, cont_part_velocity=True):
    """
    Given vector of amplitudes a0
    modal information in the left segment
    krs0 - np 1d array complex
    gammas0 - np 1d array (sqrt(krm^2 - k_hs^2).real)
    phi0 - np 2d array (Nz x M)
    z0 - np 1d array
    rho0 - np 1d array
    rho_hs0 - float
    modal information in the right segment
    same format

    r0 - float
        range at which the left segment began
    r1 - float
        range at which the new segment begines (and therefore the interface between left and right segments)
    same_grid - Boolean
        flag to indicate whether or not the grid is equal for all ranges
    cont_part_velocity - Boolean
        True to enforce continuity of particle velocity at segment interfaces
        Set to False to do comparisons with KRAKEN
    """
    Z1 = np.max(z1) # get these before updating them in get on same grid...
    Z0 = np.max(z0)
    if not same_grid: # need to interpolate
        z1, rho0, phi0, rho1, phi1 = get_on_new_grid(z0, z1, rho0, rho1, rho_hs0, rho_hs1, phi0, phi1, gammas0, gammas1)
    Z1_new = np.max(z1)
    a0_adv = advance_a(a0, krs0, r0, r1) # 
    p_left = np.sum(a0_adv * phi0, axis=1)
    a0_adv_weighted = krs0 * a0_adv
    p_weighted_left = np.sum(a0_adv_weighted * phi0, axis=1)

    # these are the for the integral from Z to infinity
    tail_values = a0_adv * phi0[-1,:]*np.exp(-gammas0 * (Z1_new - Z0))
    tail_values_weighted = a0_adv_weighted * phi0[-1,:]*np.exp(-gammas0 * (Z1_new - Z0))

    # now iterate over modes to compute the new mode amplitudes for the right segment 
    M1 = krs1.size
    anew = np.zeros((M1), dtype=np.complex128)
    for l in range(M1):
        phi1_l = phi1[:,l]
        # first integrate down to Z1_new
        term1 = np.trapz(p_left / rho1 * phi1_l, z1) #
        # now add the contribution to the integral from the tail
        tail1 = 1 / rho_hs1 * phi1_l[-1] * np.exp(-gammas1[l] *(Z1_new - Z1)) * np.sum(tail_values   / (gammas1[l] + gammas0))
        term1 += tail1

        # now do the particle velocity matching integral
        term2 = np.trapz(p_weighted_left / rho0 * phi1_l, z1)
        tail2 = 1 / rho_hs0 * phi1_l[-1] * np.exp(-gammas1[l] *(Z1_new - Z1)) * np.sum(tail_values_weighted   / (gammas1[l] + gammas0))
        term2 += tail2
        term2 /= krs1[l]
        if cont_part_velocity:
            al = 0.5*(term1 + term2)
        else:
            al = term1
        anew[l] = al
    return anew

def compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs, same_grid, cont_part_velocity=True):
    """
    Compute the pressure field at the receiver depths in zr
    Due to the source at zs 
    receiver is at range rs
    rgrid is the grid at which the environmental segments are centered?
    rgrid[0] must be 0
    same_grid - boolean Flag 
        True if all the modes have been evaluated on same grid (and so interpolation is not necessary)
    """
    Nr = zr.size
    krs0 = krs_list[0]
    phi0 = phi_list[0]
    zgrid0 = zgrid_list[0]
    rho0 = rho_list[0]
    rho_hs0 = rho_hs_list[0]
    c_hs0 = c_hs_list[0]
    gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
    gammas0 = np.ascontiguousarray(gammas0)
   
    phi_zs = np.zeros((krs0.size))
    for i in range(krs0.size):
        phi_zs[i] = interp.lin_int(zs, zgrid0, phi0[:,i])
    # the initial value is a bit 
    a0 = phi_zs * np.exp(-1j * krs0 * rgrid[1]) / np.sqrt(krs0) 
    a0 *= 1j*np.exp(1j*np.pi/4) # assumes rho is 1 at source depth
    a0 /= np.sqrt(8*np.pi)


    for i in range(1, rgrid.size):
        r0 = rgrid[i-1]
        r1 = rgrid[i]

        phi0 = phi_list[i-1]
        zgrid0 = zgrid_list[i-1]
        rho0 = rho_list[i-1]
        rho_hs0 = rho_hs_list[i-1]
        c_hs0 = c_hs_list[i-1]
        krs0 = krs_list[i-1]
        gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
        gammas0 = np.ascontiguousarray(gammas0)

        if rs <= r1: # source is in this segment
            if i == 1:
                a0 = advance_a(a0, krs0, r1, rs)
            else:
                a0 = advance_a(a0, krs0, r0, rs)
            p = np.sum(a0 * phi0, axis=1)
            p /= np.sqrt(rs)
            p_zr_real = interp.vec_lin_int(zr, zgrid0, p.real)
            p_zr_imag = interp.vec_lin_int(zr, zgrid0, p.imag)
            return p_zr_real + 1j*p_zr_imag
        else: # advance a0 to the next segment

            phi1 = phi_list[i]
            zgrid1 = zgrid_list[i]
            rho1 = rho_list[i]
            rho_hs1 = rho_hs_list[i]
            c_hs1 = c_hs_list[i]
            krs1 = krs_list[i]
            gammas1 = np.sqrt(krs1**2 - (omega**2 / c_hs1**2)).real
            gammas1 = np.ascontiguousarray(gammas1)
            if i == 1: # first segment is special case (phase of a0 is for source-receiver range equal to the first segment length r1)
                a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r1, r1, same_grid, cont_part_velocity)
            else:
                a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r0, r1, same_grid, cont_part_velocity)
    # rs > rgrid[-1] so the source is beyond the last segment
    a0 = advance_a(a0, krs1, r1, rs) 
    p = np.sum(a0 * phi1, axis=1)
    p /= np.sqrt(rs)
    p_zr_real = interp.vec_lin_int(zr, zgrid1, p.real) # zgrid should be very fine so I don't see any issue here...
    p_zr_imag = interp.vec_lin_int(zr, zgrid1, p.imag)
    p_zr = p_zr_real + 1j*p_zr_imag
    return p_zr

def compute_arr_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs_grid, same_grid, cont_part_velocity=True):
    """
    Compute the pressure field at the receiver depths in zr
    Due to the source at zs 
    receivers at all ranges in rs_grid
    rgrid is the grid of environmental segment interfaes (see get_seg_interface_grid)
    rgrid[0] must be 0
    same_grid - boolean Flag 
        True if all the modes have been evaluated on same grid (and so interpolation is not necessary)
    """
    Nr = zr.size
    krs0 = krs_list[0]
    phi0 = phi_list[0]
    zgrid0 = zgrid_list[0]
    rho0 = rho_list[0]
    rho_hs0 = rho_hs_list[0]
    c_hs0 = c_hs_list[0]


    pressure_out = np.zeros((Nr, rs_grid.size), dtype=np.complex128)

    gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
    gammas0 = np.ascontiguousarray(gammas0)
   
    phi_zs = np.zeros((krs0.size))
    for i in range(krs0.size):
        phi_zs[i] = interp.lin_int(zs, zgrid0, phi0[:,i])
    # the initial value is a bit 
    a0 = phi_zs * np.exp(-1j * krs0 * rgrid[1]) / np.sqrt(krs0) 
    a0 *= 1j*np.exp(1j*np.pi/4) # assumes rho is 1 at source depth
    a0 /= np.sqrt(8*np.pi)

    rcvr_range_index = 0
    for i in range(1, rgrid.size):
        r0 = rgrid[i-1]
        r1 = rgrid[i]

        phi0 = phi_list[i-1]
        zgrid0 = zgrid_list[i-1]
        rho0 = rho_list[i-1]
        rho_hs0 = rho_hs_list[i-1]
        c_hs0 = c_hs_list[i-1]
        krs0 = krs_list[i-1]
        gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
        gammas0 = np.ascontiguousarray(gammas0)

        while (rcvr_range_index < rs_grid.size) and rs_grid[rcvr_range_index] <= r1:
            rs = rs_grid[rcvr_range_index]
            if i == 1:
                p0 = advance_a(a0, krs0, r1, rs)
            else:
                p0 = advance_a(a0, krs0, r0, rs)
            p = np.sum(p0 * phi0, axis=1)
            p /= np.sqrt(rs)
            p_zr_real = interp.vec_lin_int(zr, zgrid0, p.real)
            p_zr_imag = interp.vec_lin_int(zr, zgrid0, p.imag)
            pressure_out[:, rcvr_range_index] = p_zr_real + 1j*p_zr_imag
            rcvr_range_index += 1
        if rcvr_range_index == rs_grid.size:
            break
        # advance a0 to the next segment
        phi1 = phi_list[i]
        zgrid1 = zgrid_list[i]
        rho1 = rho_list[i]
        rho_hs1 = rho_hs_list[i]
        c_hs1 = c_hs_list[i]
        krs1 = krs_list[i]
        gammas1 = np.sqrt(krs1**2 - (omega**2 / c_hs1**2)).real
        gammas1 = np.ascontiguousarray(gammas1)
        if i == 1: # first segment is special case (phase of a0 is for source-receiver range equal to the first segment length r1)
            a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r1, r1, same_grid, cont_part_velocity)
        else:
            a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r0, r1, same_grid, cont_part_velocity)

    # now finish for receivers in final segment
    while rcvr_range_index < rs_grid.size:
        rs = rs_grid[rcvr_range_index]
        p0 = advance_a(a0, krs1, r1, rs) 
        p = np.sum(p0 * phi1, axis=1)
        p /= np.sqrt(rs)
        p_zr_real = interp.vec_lin_int(zr, zgrid1, p.real) # zgrid should be very fine so I don't see any issue here...
        p_zr_imag = interp.vec_lin_int(zr, zgrid1, p.imag)
        p_zr = p_zr_real + 1j*p_zr_imag
        pressure_out[:, rcvr_range_index] = p_zr
        rcvr_range_index += 1
    return pressure_out

def get_seg_interface_grid(rgrid):
    """
    rgrid is the ranges at which the 
    environments are centered
    interfaces are halfway between them...
    """
    r_out = np.zeros((rgrid.size))
    r_out[1:] = 0.5*(rgrid[1:] + rgrid[:-1])
    return r_out

def downslope_coupled_mode_code():
    """
    # 10 km source range
    # sloping bottom
    # 100 meters deep at source
    # 200 meters deep at receiver
    """
    from pykrak.linearized_model import LinearizedEnv
    
    freq = 40.0
    omega = 2*np.pi*freq
    
    rgrid = np.linspace(0, 10e3, 100)
    Zvals = 100 + (rgrid * 100 / 10e3)

    c_hs = 1800. 
    rho_hs = 1.8
    attn_hs = .2
    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    dz = (1500 / freq / 20) # lambda / 20
    for Z in Zvals:
        z_list = [np.array([0, Z]), np.array([Z, max(Zvals) + 20.0])]
        c_list = [np.array([1500., 1470.]), np.array([1800.0, 1800.0])]
        env_rho_list = [np.array([1.0, 1.0]), np.array([rho_hs, rho_hs])]
        attn_list = [np.array([.0, .0]), np.array([attn_hs, attn_hs])]
        cmin= min([c_list[i].min() for i in range(len(c_list))])
        cmax = 1799.0

        N_list = [max(int(np.ceil((z_list[i][1] - z_list[i][0]) / dz)), 4)  for i in range(len(z_list))]

        env = LinearizedEnv(freq, z_list, c_list, env_rho_list, attn_list, c_hs, rho_hs, attn_hs, 'dbpkmhz', N_list, cmin, cmax)
        krs = env.get_krs()
        phi = env.get_phi(N_list)
        zgrid = env.get_phi_z(N_list)
        rhogrid = env.get_rho_grid(N_list)
    
        krs_list.append(krs)
        phi_list.append(phi)
        rho_list.append(rhogrid)
        zgrid_list.append(zgrid)
        c_hs_list.append(c_hs)
        rho_hs_list.append(rho_hs)
        
    zs = 25.    

    same_grid = False

    zr = np.linspace(5., Zvals.max(), 200)
    p_arr = np.zeros((zr.size, rgrid.size-10), dtype=np.complex128)
    rcm_grid = get_seg_interface_grid(rgrid)
    for i in range(10,rgrid.size):
        rs = rgrid[i]
        p = compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, rs, same_grid)
        p_arr[:,i-10] = p
   
    tl = 20*np.log10(abs(p_arr))


    plt.figure()
    plt.pcolormesh(rgrid[10:], zr, tl, vmax=np.max(tl), vmin=np.max(tl)-60)
    plt.colorbar()
    plt.gca().invert_yaxis()

def upslope_coupled_mode_code():
    """
    # 10 km source range
    # sloping bottom
    # 100 meters deep at source
    # 200 meters deep at receiver
    """
    from pykrak.linearized_model import LinearizedEnv
    
    freq = 40.0
    omega = 2*np.pi*freq
    
    rgrid = np.linspace(0, 10e3, 100)
    Zvals = 200 - (rgrid * 100 / 10e3)

    c_hs = 1800. 
    rho_hs = 1.8
    attn_hs = .2
    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    dz = (1500 / freq / 20) # lambda / 20
    for Z in Zvals:
        z_list = [np.array([0, Z]), np.array([Z, max(Zvals) + 20.0])]
        c_list = [np.array([1500., 1470.]), np.array([1800.0, 1800.0])]
        env_rho_list = [np.array([1.0, 1.0]), np.array([rho_hs, rho_hs])]
        attn_list = [np.array([.0, .0]), np.array([attn_hs, attn_hs])]
        cmin= min([c_list[i].min() for i in range(len(c_list))])
        cmax = 1799.0

        N_list = [max(int(np.ceil((z_list[i][1] - z_list[i][0]) / dz)), 4)  for i in range(len(z_list))]

        env = LinearizedEnv(freq, z_list, c_list, env_rho_list, attn_list, c_hs, rho_hs, attn_hs, 'dbpkmhz', N_list, cmin, cmax)
        krs = env.get_krs()
        phi = env.get_phi(N_list)
        zgrid = env.get_phi_z(N_list)
        rhogrid = env.get_rho_grid(N_list)
    
        krs_list.append(krs)
        phi_list.append(phi)
        rho_list.append(rhogrid)
        zgrid_list.append(zgrid)
        c_hs_list.append(c_hs)
        rho_hs_list.append(rho_hs)
        
    zs = 25.    

    zr = np.linspace(5., Zvals.max(), 200)
    p_arr = np.zeros((zr.size, rgrid.size-10), dtype=np.complex128)
    rcm_grid = get_seg_interface_grid(rgrid)
    for i in range(10,rgrid.size):
        rs = rgrid[i]
        p = compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, rs,False)
        p_arr[:,i-10] = p
   
    tl = 20*np.log10(abs(p_arr))


    plt.figure()
    plt.pcolormesh(rgrid[10:], zr, tl, vmax=np.max(tl), vmin=np.max(tl)-60)
    plt.colorbar()
    plt.gca().invert_yaxis()

def range_independent_check():
    """
    # 100 km source range
    """
    from pykrak.linearized_model import LinearizedEnv
    
    freq = 35.0
    omega = 2*np.pi*freq
    
    rgrid = np.linspace(0, 10e4, 100)
    Z = 200.0

    c_hs = 1800. 
    rho_hs = 1.8
    attn_hs = .2
    attn_units = 'dbpkmhz'

    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    dz = (1500 / freq / 80) # lambda / 20
    z_list = [np.array([0, Z]), np.array([Z, Z + 20.0])]
    c_list = [np.array([1500., 1470.]), np.array([1800.0, 1800.0])]
    env_rho_list = [np.array([1.0, 1.0]), np.array([rho_hs, rho_hs])]
    attn_list = [np.array([.0, .0]), np.array([attn_hs, attn_hs])]
    cmin= min([c_list[i].min() for i in range(len(c_list))])
    cmax = 1799.0

    N_list = [max(int(np.ceil((z_list[i][1] - z_list[i][0]) / dz)), 4)  for i in range(len(z_list))]

    env = LinearizedEnv(freq, z_list, c_list, env_rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
    krs = env.get_krs()
    phi = env.get_phi(N_list)
    zgrid = env.get_phi_z(N_list)
    rhogrid = env.get_rho_grid(N_list)

    zs = 25.    
    zr = np.linspace(5., Z, 200)

    """ Calculate usinga  range-independent model """
    from pykrak.pressure_calc import get_arr_pressure
    phi_zs = env.get_phi_zr(np.array([zs]))
    phi_zr = env.get_phi_zr(zr)
    p2 = get_arr_pressure(phi_zs, phi_zr, krs, rgrid[1:])
    p2 = np.squeeze(p2)
    tl2 = 20*np.log10(abs(p2))

    plt.figure()
    plt.pcolormesh(rgrid[1:], zr, tl2)
    #plt.plot(rgrid[10:], tl2[np.argmin(np.abs(env.phi_z - zs))])
    #plt.plot(rgrid[10:], -10*np.log10(rgrid[10:]))
    plt.colorbar()
    plt.gca().invert_yaxis()

    c_hs_imag = get_c_imag(c_hs, attn_hs, attn_units, 2*np.pi*freq)
    c_hs_complex = c_hs + 1j*c_hs_imag

    for i in range(rgrid.size):
        krs_list.append(krs)
        phi_list.append(phi)
        rho_list.append(rhogrid)
        zgrid_list.append(zgrid)
        c_hs_list.append(c_hs_complex)
        rho_hs_list.append(rho_hs)
        
    p_arr = np.zeros((zr.size, rgrid.size-1), dtype=np.complex128)
    rcm_grid = get_seg_interface_grid(rgrid)
    for i in range(1,rgrid.size):
        rs = rgrid[i]
        p = compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, rs, True)
        p_arr[:,i-1] = p
   
    tl = 20*np.log10(abs(p_arr))



    print('krs', krs)


    plt.figure()
    plt.pcolormesh(rgrid[1:], zr, tl)
    plt.colorbar()
    plt.gca().invert_yaxis()

    plt.figure()

    plt.plot(rgrid[1:], tl[np.argmin(np.abs(env.phi_z - zs))])
    plt.plot(rgrid[1:], tl2[np.argmin(np.abs(env.phi_z - zs))])
    plt.plot(rgrid[1:], -10*np.log10(rgrid[1:]))
    plt.plot(rgrid[1:], -20*np.log10(rgrid[1:]))


    fig, axes = plt.subplots(2,1, sharex=True)
    z_ind = np.argmin(np.abs(env.phi_z - zs))
    axes[0].plot(rgrid[1:], np.abs(p_arr[z_ind, :]),'k')
    axes[0].plot(rgrid[1:], np.abs(p2[z_ind, :]), 'b')
    axes[1].plot(rgrid[1:], np.angle(p_arr[z_ind, :]), 'k')
    axes[1].plot(rgrid[1:], np.angle(p2[z_ind, :]), 'b')


if __name__ == '__main__':
    upslope_coupled_mode_code()
    plt.show()
    range_independent_check()
    downslope_coupled_mode_code()
    plt.show()
