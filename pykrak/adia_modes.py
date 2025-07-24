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
from pykark.interp import interp
from numba import jit, njit
from pykrak.attn_pert import get_c_imag
from pykrak.pressure_calc import get_arr_pressure
import time

@njit(cache=True)
def range_interp_krs_phi(rpt, rgrid, kr_arr, phi_arr):
    """
    kr_arr is values of horiz. wavenumbers at different
    ranges
    rgrid is the range grid at which the environments that these
    modes are definesd
    rpt is the desired new point to interpolate the wavenumbers
    """
    if rpt > rgrid.max() or rpt < rgrid.min():
        raise ValueError("rpt is outside of the range of the model")
    else:
        klo, khi = interp.get_klo_khi(rpt, rgrid)
        dr = rgrid[khi] - rgrid[klo]
        kr_left = kr_arr[klo,:]
        kr_right = kr_arr[khi,:]
        a = (rgrid[khi] - rpt) / dr
        b = (rpt - rgrid[klo]) / dr
        # only include modes that exist at both ranges
        kr_inds = (kr_left.real > 0) & (kr_right.real > 0)
        kr_vals = a*kr_left[kr_inds] + b*kr_right[kr_inds]

        phi_left = phi_arr[klo,:,:]
        phi_right = phi_arr[khi,:,:]
        phi_vals = a*phi_left[kr_inds,:] + b*phi_right[kr_inds,:]
    return kr_vals, phi_vals

@njit(cache=True)
def get_M(krs):
    return np.sum(krs != -1.0)

@njit(cache=True)
def compute_arr_adia_pressure(krs_arr, phi_zs, phi_zr, rgrid, rs_grid):
    """
    krs - array of wavenumbers, padded to maximum mode num
        shape (num environment segments, M_max)
    phi_zs - array of mode source amplitues
        shape (num sources, num modes)
    phi_zr - array of mode shapes padded to maximum mode num
        shape (num env segments, M_max, num rcvr depths)
    rgrid - array of ranges at which the environment segments are centered
    rs_grid - array
        source-receiver ranges (source assumed at rgrid[0])
        
    Assumes all modes have been evaluated on same grid 
    """
    Nrr = rs_grid.size

    #print('Ns, Nr, Nrr', Ns, Nr, Nrr)
    Ns = phi_zs.shape[0]
    Nr = phi_zr.shape[1] # first axis is environment

    pressure_out = np.zeros((Ns, Nr, Nrr), dtype=np.complex128)
    M_max = krs_arr.shape[-1]

    curr_env_seg = 0
    krs_curr = krs_arr[0,:]
    M = get_M(krs_curr)
    rs_i = 0
    rs_j = 0
    while curr_env_seg < rgrid.size-1:
        #print('computing field in seg ', curr_env_seg)
        Ri0 = rgrid[curr_env_seg]
        Ri1 = rgrid[curr_env_seg+1]
        #print('Ri0, Ri1', Ri0, Ri1)
        while rs_grid[rs_j] < Ri1:
            rs_j += 1
        num_pts = rs_j - rs_i
        krsa = krs_arr[curr_env_seg,:]
        krsb = krs_arr[curr_env_seg+1,:]
        phi_a = phi_zr[curr_env_seg,:,:]
        phi_b = phi_zr[curr_env_seg+1,:,:]
        Ma = get_M(krsa)
        Mb = get_M(krsb)

        M = min(Ma, Mb, M)
        krsa = krsa[:M]
        krsb = krsb[:M]
        phi_a = phi_a[:, :M]
        phi_b = phi_b[:, :M]
        krs_curr = krs_curr[:M]
        phi_zs = phi_zs[:,:M]

        dkrdr = (krsb - krsa) / (Ri1 - Ri0)
        dphidr = (phi_b - phi_a) / (Ri1 - Ri0)
        for r_ind in range(rs_i, rs_j):
            #now = time.time()
            rpt = rs_grid[r_ind]
            #print('getting field at', rpt)
            krs_seg = krsa + 0.5*dkrdr*(rpt - Ri0)
            #print('krs seg shape', krs_seg.shape)
            krs_eff = 1/rpt*(Ri0 *krs_curr  + (krs_seg)*(rpt - Ri0))
            phi_seg = phi_a + dphidr*(rs_grid[r_ind] - Ri0)
            #print('phi_seg.shape', phi_seg.shape)
            rpt_arr = np.zeros((1,1))
            rpt_arr[0,0] = rpt
            #print('phi zs shape, phi zr shape, krs eff shape', phi_zs.shape, phi_zr.shape, krs_eff.shape)
            phi_zs = np.ascontiguousarray(phi_zs)
            phi_seg = np.ascontiguousarray(phi_seg)
            p = get_arr_pressure(phi_zs, phi_seg, krs_eff, rpt_arr)
            pressure_out[...,r_ind] = p[...,0]
            #print('single loop press time', time.time()-now)
        rs_i = rs_j
        krs_curr = 1/Ri1*(Ri0 *krs_curr  + (krsa + 0.5*dkrdr*(Ri1 - Ri0))*(Ri1 - Ri0))
        curr_env_seg += 1
    if rs_grid[rs_i] == rgrid[-1]:
        rpt = rs_grid[rs_i]
        krs_eff = krs_curr
        #print('krs_curr', krs_curr)
        #print('mean krs', .5*(krsa + krsb))
        phi_seg = phi_b
        rpt_arr = np.zeros((1,1))
        rpt_arr[0,0] = rpt
        phi_zs = np.ascontiguousarray(phi_zs)
        phi_seg = np.ascontiguousarray(phi_seg)
        p = get_arr_pressure(phi_zs, phi_seg, krs_eff, rpt_arr)
        pressure_out[...,rs_i] = p[...,0]
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

def downslope_adia_mode_code():
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

def upslope_adia_mode_code():
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
    first = True
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
    upslope_adia_mode_code()
    plt.show()
    range_independent_check()
    downslope_adia_mode_code()
    plt.show()
