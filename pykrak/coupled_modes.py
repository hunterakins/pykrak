"""
Description:
Routines for coupled mode field calculation
Follows the forward scattering, sequential approach used in Kraken
Specifically, it updates the modal weights used to calculate pressure 
by enforcing continuity of pressure at each interface between range-independent
segments

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
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def get_pressure(phi_zr, phi_zs, krs, r, tilt_angle=0, zr=None):
    """
    Consistent with fourier transform of the form
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    From modes evaluated at receiver depths,
    evaluated at source depth, 
    wavenumbers kr, and source range r
    Return pressure as column vector
    Positive angle means it's leaning towards the source, 
    the lowest element of the array is always located at a range r 
    (so it's sort of the fixed point)
    Positive angle in DEGREES 
    Input
    phi_zr - np 2d array
        first axis is receiver depth, second is mode number
        amplitude of modes at receivers
    phi_zs - np 2d array
        I think first axis is usually just a single depth value?
        Second is mode number
    krs - np 1d array
        horizontal wavenumbers
    r - float
        source range
    tilt_angle - optional int/ float
        Array tilt. Positive means array isleaning towards the source (top element is closer than bott. eleent
    zr - optional
        If providing tilt, I need array positions
    """
    modal_matrix = phi_zs*phi_zr
    if tilt_angle != 0:
        Z = np.max(zr) - zr
        deltaR = Z*np.tan(tilt_angle * np.pi / 180.)
        deltaR = deltaR.reshape(zr.size,1)
        r = r - deltaR #  minus so that it leans towards the source (range gets closer)
        krs = krs.reshape(1, krs.size)
        range_arg = krs * r
        range_dep = np.exp(-1j*range_arg) / np.sqrt(range_arg.real)
        prod = modal_matrix*range_dep
        p = np.sum(prod, axis=1)[:, np.newaxis]
    else:
        range_dep = np.exp(-1j*r*krs) / np.sqrt(krs.real*r)
        range_dep = range_dep.reshape(krs.size,1)
        p = modal_matrix@range_dep
    p *= np.exp(1j*np.pi/4)
    p /= np.sqrt(8*np.pi)
    return p

def get_interface_pts(rgrid):
    """ 
    Given a range-dependent environment with N updated points
    given in rgrid, break the transect into N range-independent segments
    defined by N-1 interface points r_int_grid
    Input 
    rgrid - np 1d array
        points r at which the environment is updated
    Output 
    r_int_grid  - np 1d array
        points of the interface between the range-independent subsegments
    """
    r_int_grid = (rgrid[1:] + rgrid[:-1]) / 2
    return r_int_grid

def compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs):
    """
    krs_list - list of krs for range-independent segments
    phi_list - list of mode shape functions for range-independent segments
    zgrid_list - list of z grids for the mode shapes (and pressure fields)
            for each range-independent segments
    rho_list - list of rho grid for the range-independent segs
    rgrid consists of the points for which the range-dependent model
    is specified (range-indendependent segments bracket each point)
    zs - source depth
    zr - receiver depths
    rs - receiver range (transect starts at source, so first point in grid is the 
        environment at source position)

    A will be used to keep track of the weights 
        like in KRAKEN, it is basically the weights "a" in e.g. eq. 239 in JKPS
        but also with the phase at the beginning of the interface 
    P holds the pressure at the interface ranges as computed from the segment to the left
        (we march left to right)
        P is weighted by the grid spacing and the density at the grid points...
        It is then interpolated onto the new grid before updating...
    """

    # get interface ranges (halfway between profile points)
    ri_pts = get_interface_pts(rgrid)
    #plt.figure()
    #plt.plot(rgrid, 'k+')
    #plt.plot(ri_pts, 'b+')
    #plt.show()
    
    # restrict to relevant ranges...
    rel_inds =  ri_pts < rs 
    if rel_inds.sum() == 0: # receiver is in first range-independent segment
        num_segs = 1
    else:
        ri_pts = ri_pts[rel_inds]
        num_segs = ri_pts.size + 1
    
    # first compute pressure in the first segment
    krs0 = krs_list[0]
    phi0 = phi_list[0]
    M = krs0.size
    z0 = zgrid_list[0]
    #rho_hs0 = rho_hs_list[0]
    c_hs0 = c_hs_list[0]
    phi_zs = np.zeros((1, M))
    for i in range(M):
        phi_zs[:,i] = np.interp(zs, z0, phi0[:,i])

    krs0, phi0, z0 = krs_list[0].copy(), phi_list[0].copy(), zgrid_list[0].copy()
    rho0, rho_hs0, c_hs0 = rho_list[0].copy(), rho_hs_list[0], c_hs_list[0]
    if num_segs == 1:
        P = get_pressure(phi0, phi_zs, krs0, rs)
        P = np.interp(zr, z0, P[:,0])
    else:
        #  now loop over the range-independent segments, tracking the weights...
        for i in range(num_segs-1):
            # get modes, density, and grid for the new range-ind segment
            krs1, phi1, z1 = krs_list[i+1].copy(), phi_list[i+1].copy(), zgrid_list[i+1].copy()
            rho1, rho_hs1, c_hs1 = rho_list[i+1].copy(), rho_hs_list[i+1], c_hs_list[i+1]

            # compute pressure at  interface using last range-ind. seg., interpolating
            # it onto the current grid (z1)
            gamma0 = np.sqrt(np.square(krs0.real) - omega*omega / c_hs0 / c_hs0)
            gamma1 = np.sqrt(np.square(krs1.real) - omega*omega / c_hs1 / c_hs1)
            phi0_bott = phi0[-1,:]
            phi1_bott = phi1[-1,:]
            z_bott0 = z0[-1] # store bottom pt...
            z_bott1 = z1[-1] # store bottom pt...

            # if new segment is deeper, use evanescent tail to get old seg at all points on new grid
            if z_bott1 > z_bott0: 
                z_extra = z1[z1 > z_bott0] # extra_zpts
                exp_decay = np.exp(-np.outer((z_extra - z_bott0), gamma0))
                phi0_extra = phi0[-1,:] * exp_decay
                phi0 = np.vstack((phi0, phi0_extra))
                #print('New seg deeper. Num new points is {0}. Extra mode values has shape {1}. New mode matrix has shape {2}'.format(z_extra.size, phi0_extra.shape, phi0.shape))
                z0 = np.hstack((z0, z_extra))
            elif z_bott0 > z_bott1: # old segment was deeper, so use evanescent tail to extend it to deepest point in old grid. This is necessary for the integral.
                z_extra = z0[z0 > z_bott1] # extra_zpts
                exp_decay = np.exp(-np.outer((z_extra - z_bott1), gamma1))
                phi1_extra = phi1[-1,:] * exp_decay
                phi1 = np.vstack((phi1, phi1_extra))
                #print('Old seg deeper. Num new points is {0}. Extra mode values has shape {1}. New mode matrix has shape {2}'.format(z_extra.size, phi1_extra.shape, phi1.shape))
                z1 = np.hstack((z1, z_extra))
                rho1 = np.hstack((rho1, rho_hs1*np.ones(z_extra.size))) # pad rho1 with halfpsace val

            zr_max = zr.max() # if receiver depth is deeper than grid (in halfspace), theb extend grid
            if zr_max > z1[-1]: # need to extend grid to deepest reeiver point
                f = omega/2/np.pi
                lam = 1500 / f
                dz = lam / 20
                Z = (zr_max - z1[-1])
                N = max(10, int(Z/dz))
                z_extra = np.linspace(z1[-1], zr_max, N)
                #plt.figure()
                #plt.plot(z1, phi0[:,0])
                phi0_extra = phi0[-1,:]*np.exp(-np.outer(z_extra - z1[-1], gamma0))
                phi1_extra = phi1[-1,:]*np.exp(-np.outer(z_extra - z1[-1], gamma1))
                phi0 = np.vstack((phi0, phi0_extra))
                phi1 = np.vstack((phi1, phi1_extra))
                z0 = np.hstack((z0, z_extra))
                z1 = np.hstack((z1, z_extra))
                rho1 = np.hstack((rho1, rho_hs1*np.ones(z_extra.size))) # pad rho1 with halfpsace val
                #plt.plot(z1, phi0[:,0])
                #plt.gca().invert_yaxis()
                #plt.show()
                

            if i == 0:# first range...     
                A0 = np.exp(1j*np.pi/4)*phi_zs[0,:] /np.sqrt(8*np.pi) /np.sqrt(krs0)
                A0 *= np.exp(-1j*ri_pts[0]*krs0)
                matrix = A0  * phi0
                P = np.sum(matrix,axis=1)
            else:
                A0 *= np.exp(-1j*(ri_pts[i] - ri_pts[i-1])*krs0)
                matrix = A0*phi0
                P = np.sum(matrix, axis=1)


            P = np.interp(z1, z0, P)[:, np.newaxis]
            #plt.figure()
            #plt.plot(P[:,0].real, z1)
            #plt.plot(P[:,0].imag, z1)
            #plt.show()

            # compute mode_weights A by first integrating down to halfspace
            h_pts = z1[1:] - z1[:-1] # get interval widths for integration
            rho_pts = .5*(rho1[1:] +rho1[:-1]) # use average density for intervals

            integrand = P*phi1
            integrand = .5*(integrand[1:] + integrand[:-1]) * (h_pts / rho_pts)[:, np.newaxis]


            #for l in range(phi1.shape[1]):
            #    integrand1 = phi1[:,0]*phi1[:,l]
            #    integrand1 = .5*(integrand1[1:] + integrand1[:-1]) * (h_pts / rho_pts)
            #    tail1 = (phi1[-1,0] * phi1[-1,l]) /2/rho_hs1 / (gamma1[0] + gamma1[l])
            #    print(np.sum(integrand1))
            #sys.exit(0)
            #plt.figure()
            #plt.plot(integrand[:,0].real, .5*(z1[1:] + z1[:-1]))
            #plt.plot(integrand[:,0].imag, .5*(z1[1:] + z1[:-1]))
            #plt.show()
            
            # then add contribution to the integral of the tail 
            tail = np.zeros(krs1.size,dtype=np.complex_)
            for mode_num in range(krs1.size):
                if i == 0:
                    tail_matrix = A0*phi0_bott*np.exp(-(z1[-1] - z_bott0))
                    tail[mode_num] = phi1[-1,mode_num] * np.sum(tail_matrix / (gamma0 + gamma1[mode_num]) )
                else:
                    tail_matrix = A0*phi0_bott*np.exp(-(z1[-1] - z_bott0))
                    tail[mode_num] = phi1[-1,mode_num] * np.sum(tail_matrix/ (gamma0 + gamma1[mode_num]) )
            tail /= rho_hs1
            A = np.sum(integrand, axis=0)
            A += tail

            A0 = A # save weights to compute tail and pressure field in next segment

            # save current modes and whatnot for next segment
            krs0, phi0, z0 = krs1.copy(), phi1.copy(), z1.copy()
            rho0, rho_hs0, c_hs0 = rho1.copy(), rho_hs1, c_hs1
            
            # restore original grid
            if i < num_segs-1:
                z_inds = z0 <= z_bott1
                phi0 = phi0[z_inds,:]
                z0 = z0[z_inds]
                rho0 = rho0[z_inds]

        # compute final field and interpolate onto receiver depths
        zr_max = zr.max()
        z0_bott = z0.max()
        if zr_max > z0.max(): # use tail to compute phi0 down to bottom of zr_max
            z_extra = zr[zr > z0.max()]
            gamma0 = np.sqrt(np.square(krs0.real) - omega*omega / c_hs0 / c_hs0)
            exp_decay = np.exp(-np.outer((z_extra - z0_bott), gamma0))
            phi0_extra = phi0[-1,:] * exp_decay
            phi0 = np.vstack((phi0, phi0_extra))
            z0 = np.hstack((z0, z_extra))
        #plt.figure()
        #plt.plot(phi0[:,0],z0)
        #plt.plot(phi0[:,-1],z0)
        #plt.show()

        matrix = A0*np.exp(-1j*(rs - ri_pts[-1])) / np.sqrt(rs)* phi0
        P = np.sum(matrix, axis=1)
        P = np.interp(zr, z0, P)
    return P

def downslope_coupled_mode_code():
    """
    # 10 km source range
    # sloping bottom
    # 100 meters deep at source
    # 200 meters deep at receiver
    """
    from envs import factory
    builder=factory.create('hs')
    
    freq = 200
    omega = 2*np.pi*freq
    
    rgrid = np.linspace(0, 10e3, 100)
    Zvals = 100 + (rgrid * 100 / 10e3)

    c_hs = 1800. 
    rho_hs = 1.8
    attn_hs = 1.
    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    for Z in Zvals:
        zw  = np.array([0., Z])
        cw  = np.array([1500., 1500.])
        dz = 1.0
        env = builder(zw, cw, c_hs, rho_hs, attn_hs, 'dbpkmhz', pert=False)
        env.add_freq(freq)
        krs = env.get_krs(**{'cmax':1799., 'Nh':1})
        phi = env.get_phi()
        zgrid = env.get_phi_z()
        rhogrid = env.get_rho_grid()
    

        krs_list.append(krs)
        phi_list.append(phi)
        rho_list.append(rhogrid)
        zgrid_list.append(zgrid)
        c_hs_list.append(c_hs)
        rho_hs_list.append(rho_hs)
        
    zs = 25.    

    zr = np.linspace(5., Zvals.max(), 200)
    p_arr = np.zeros((zr.size, rgrid.size-10), dtype=np.complex128)
    for i in range(10,rgrid.size):
        rs = rgrid[i]
        p = compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs)
        p_arr[:,i-10] = p
    
    tl = 20*np.log10(abs(p_arr))

    plt.figure()
    plt.pcolormesh(rgrid[10:], zr, tl, vmax=-60, vmin=-90)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

def ri_check():
    from envs import factory
    builder=factory.create('hs')
    
    freq = 100
    omega = 2*np.pi*freq
    
    Z = 200.
    """ Coupled mode check """
    rgrid = np.linspace(0, 10e3, 1000)
    Zvals = 200 + np.zeros((rgrid.size))

    c_hs = 1800. 
    rho_hs = 1.8
    attn_hs = 1.
    #krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    zw  = np.array([0., Z])
    cw  = np.array([1500., 1500.])
    dz = .1
    env = builder(zw, cw, c_hs, rho_hs, attn_hs, 'dbpkmhz', dz=dz, pert=False)
    env.add_freq(freq)
    krs = env.get_krs(**{'cmax':1799., 'Nh':1})
    phi = env.get_phi()
    zgrid = env.get_phi_z()
    rhogrid = env.get_rho_grid()
    

    krs_list = [krs] * rgrid.size
    phi_list = [phi] * rgrid.size
    rho_list = [rhogrid] * rgrid.size
    zgrid_list = [zgrid] * rgrid.size
    c_hs_list = [c_hs] * rgrid.size
    rho_hs_list = [rho_hs]*rgrid.size
        
    zs = 25.    

    zr = np.linspace(5., 195., 200)
    p_arr = np.zeros((zr.size, rgrid.size-10), dtype=np.complex128)
    p_ri_arr = np.zeros((zr.size, rgrid.size-10), dtype=np.complex128)
    phi_zr = np.zeros((zr.size, krs.size))
    phi_zs = np.zeros((1, krs.size))
    for i in range(krs.size):
        phi_zr[:,i] = np.interp(zr, zgrid, phi[:,i])
        phi_zs[0,i] = np.interp(zs, zgrid, phi[:,i])
   
    corr_grid = np.zeros((rgrid.size-10)) 
    for i in range(10,rgrid.size):
        rs = rgrid[i]
        p = compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs)
        p_ri = get_pressure(phi_zr, phi_zs, krs, rs)[:,0]
        p_ri_arr[:,i-10] = p_ri
        p_arr[:,i-10] = p
        corr = np.square(abs(np.sum(p_ri.conj()*p))) / np.square(np.linalg.norm(p_ri)) / np.square(np.linalg.norm(p))
        #if corr < .3:
        #    plt.plot(p_ri.real, 'b--')
        #    plt.plot(p.real, 'b')
        #    plt.plot(p_ri.imag, 'r--')
        #    plt.plot(p.imag, 'r')
        #plt.show()
        corr_grid[i-10] = corr
    
    tl = 20*np.log10(abs(p_arr))
    tl_ri = 20*np.log10(abs(p_ri_arr))

    plt.figure()
    plt.plot(corr_grid)
    plt.show()


    fig, axes =plt.subplots(2,1, sharex=True)
    cs0 = axes[0].pcolormesh(rgrid[10:], zr, tl, vmax=-60, vmin=-90)
    fig.colorbar(cs0, ax=axes[0])
    plt.gca().invert_yaxis()
    cs1 = axes[1].pcolormesh(rgrid[10:], zr, tl_ri, vmax=-60, vmin=-90)
    fig.colorbar(cs1, ax=axes[1])
    plt.gca().invert_yaxis()

    fig, axes =plt.subplots(2,1, sharex=True)
    cs0 = axes[0].pcolormesh(rgrid[10:], zr, p_arr.real)
    fig.colorbar(cs0, ax=axes[0])
    plt.gca().invert_yaxis()
    cs1 = axes[1].pcolormesh(rgrid[10:], zr, p_ri_arr.real)
    fig.colorbar(cs1, ax=axes[1])
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    ri_check()
    downslope_coupled_mode_code()
