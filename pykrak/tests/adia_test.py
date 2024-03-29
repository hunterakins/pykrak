"""
Description:
    Compare PyKrak coupled mode run KRAKEN

Date:
    10/9/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pyat.pyat.readwrite import read_env, write_env, write_fieldflp, read_shd, write_bathy

from pykrak import coupled_modes as cm
from pykrak.test_helpers import get_krak_inputs

from pykrak.linearized_model import LinearizedEnv
from pyat.pyat import env as pyat_env
import os


os.system('rm cm_log.txt')
def downslope_test():
    freq = 100.0
    omega = 2*np.pi*freq
    Z0 = 100.0
    Z1 = 200.0
    R = 10*1e3
    num_segs = 21
    Zvals = np.linspace(Z0, Z1, num_segs)
    Zmax = Zvals.max()
    rgrid = np.linspace(0.0, R, num_segs)
    rcm_grid = cm.get_seg_interface_grid(rgrid)
    cw = 1500.0
    rho_w = 1.0
    c_hs = 1800.0
    rho_hs = 2.0
    attn_hs = 0.01
    #attn_units = 'dbpkmhz'
    attn_units = 'npm'
    mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

    cmin = 1500.0
    cmax = 1799.0

    # Pekeris waveguide at each segment
    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    nmesh_list = []
    env_list = []
    # Loop over each segment and run pykrak to get the necessary values
    for seg_i in range(num_segs):
        Z = Zvals[seg_i]
        env_z_list = [np.array([0.0, Z]), np.array([Z, Zmax])]
        env_c_list = [np.array([cw, cw]), np.array([c_hs, c_hs])]
        env_rho_list = [np.array([rho_w, rho_w]), np.array([rho_hs, rho_hs])]
        env_attn_list = [np.array([0.0, 0.0]), np.array([attn_hs, attn_hs])]
        N_list = [max(int(np.ceil(Z / mesh_dz)), 20), max(int(np.ceil(Zmax - Z)), 10)]

        if Z == Zmax:
            env_z_list = [env_z_list[0]]
            env_c_list = [env_c_list[0]]
            env_rho_list = [env_rho_list[0]]
            env_attn_list = [env_attn_list[0]]
            N_list = [N_list[0]]


        nmesh_list.append(N_list)


        env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
        
        env_list.append(env)
        env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
        env.add_x0(np.array([0.0]))
        #tmp_krs = env.get_krs(**{'N_list': N_list, 'Nh':1})
        modes = env.full_forward_modes()
        krs = modes.krs
        krs_str = krs.astype(str)
        with open('cm_log.txt', 'a') as f:
            f.write('Running pykrak for depth {0} with mesh N: {1}\n'.format(Z, N_list))
            for tmp_j in range(env.z_arr.size):
                f.write('{}, {}, {}, {}\n'.format(env.z_arr[tmp_j], env.c_arr[tmp_j], env.rho_arr[tmp_j], env.attn_arr[tmp_j]))
            for i in range(krs.size):
                f.write('{0}  {1}\n'.format(i + 1, krs_str[i]))
        phi = modes.phi
        zgrid = modes.z
        rhogrid = env.get_rho_grid(N_list)
    
        krs_list.append(krs)
        phi_list.append(phi)
        rho_list.append(rhogrid)
        zgrid_list.append(zgrid)
        c_hs_list.append(c_hs)
        rho_hs_list.append(rho_hs)


    # Now we have all the values we need to run the coupled mode model

    zs = 25.    

    same_grid = False
    ranges = np.linspace(100.0, 10*1e3, 1000)

    zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
    zr = zout[1:]
    p_arr = np.zeros((zr.size, ranges.size-1), dtype=np.complex128)
    #for i in range(1,ranges.size):
    #    rs = ranges[i]
    #    p = cm.compute_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, rs, same_grid, cont_part_velocity=False) # False for KRAKEN model comp
    #    p_arr[:,i-1] = p

    p_arr = cm.compute_arr_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, ranges[1:], same_grid, cont_part_velocity=False) # False for KRAKEN model comp

    p_tl = 20*np.log10(np.abs(p_arr)) 
    plt.figure()
    plt.pcolormesh(ranges[1:]*1e-3, zr, p_tl)
    plt.gca().invert_yaxis()
    plt.colorbar()

    zind = np.argmin(np.abs(zr - 100.0))
    plt.figure()
    plt.plot(ranges[1:]*1e-3, p_tl[zind,:], 'k')

    p_100 = p_arr[zind,:]


    # now use KRAKEN
    name = 'at_files/cm_pekeris_test.env'
    model='kraken'
    rmax = 0 # force use of the first mesh
    source = pyat_env.Source(np.array([zs]))
    dom = pyat_env.Dom(ranges*1e-3, zout)
    pos = pyat_env.Pos(source, dom)
    beam = None
    cint= pyat_env.cInt(cmin, cmax)

    for seg_i in range(num_segs):
        env = env_list[seg_i]
        ssp, bdy = get_krak_inputs(env, twod=True)

        NMESH = nmesh_list[seg_i]
        NMESH = [x for x in NMESH]
        if seg_i == 0:
            append = False
        else:
            append=True
        write_env(name, model, 'Auto gen from Env object', freq, ssp, bdy, pos, beam, cint, rmax, NMESH=NMESH, append=append)
        

    #bathy_grid = np.zeros((rgrid.size, 2))
    #bathy_grid[:,0] = rgrid*1e-3
    #bathy_grid[:,1] = Zvals
    #write_bathy('at_files/cm_pekeris_test.bty', bathy_grid)
    
    # run kraken
    import os
    os.system('cd at_files && kraken.exe cm_pekeris_test')

    # run field
    #kwargs = {'rProf':rcm_grid*1e-3, 'NProf':rgrid.size}
    kwargs = {'rProf':rgrid*1e-3, 'NProf':rgrid.size}
    field_dom = pyat_env.Dom(ranges*1e-3,zr)
    pos_field = pyat_env.Pos(source, field_dom)
    write_fieldflp('at_files/cm_pekeris_test.flp', 'RCOC', pos_field, **kwargs)
    os.system('cd at_files && field.exe cm_pekeris_test')


    [ PlotTitle, PlotType, freqVec, atten, pos, pressure ] = read_shd('at_files/cm_pekeris_test.shd')
    pressure = np.squeeze(pressure)
    # correct KRAKEN scaling to agree with mine
    pressure /= np.sqrt(2*np.pi) 
    pressure /= np.sqrt(8 *np.pi)
    k_tl = 20*np.log10(np.abs(pressure))
    kp_100 = pressure[zind,:][1:]
    zind = np.argmin(np.abs(pos.r.depth - 100.0))
    plt.plot(ranges[1:]*1e-3, k_tl[zind,:][1:], 'b')

    plt.figure()
    plt.pcolormesh(ranges, zout[1:], k_tl)
    plt.gca().invert_yaxis()
    plt.colorbar()
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(ranges[1:]*1e-3, np.abs(p_100), 'k')
    axes[0].plot(ranges[1:]*1e-3, np.abs(kp_100), 'b')

    axes[1].plot(ranges[1:]*1e-3, np.angle(p_100), 'k')
    axes[1].plot(ranges[1:]*1e-3, np.angle(kp_100), 'b')

    plt.show()


downslope_test()
