"""
Description:
    Coupled mode with range_dep_model and MPI

Date:
    10/9/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pyat.pyat.readwrite import read_env, write_env, write_fieldflp, read_shd, write_bathy
from pykrak import coupled_modes as cm
from pykrak.linearized_model import LinearizedEnv
from pykrak.test_helpers import get_krak_inputs
from pykrak.cm_model import CMModel
from pyat.pyat import env as pyat_env
from mpi4py import MPI
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    os.system('rm cm_log.txt')

def downslope_test():
    freq = 100.0
    omega = 2*np.pi*freq
    Z0 = 100.0
    Z1 = 200.0
    R = 10*1e3
    num_segs = size
    Zvals = np.linspace(Z0, Z1, num_segs)
    Zmax = Zvals.max()
    rgrid = np.linspace(0.0, R, num_segs)
    rcm_grid = cm.get_seg_interface_grid(rgrid)
    cw = 1500.0
    rho_w = 1.0
    c_hs = 1800.0
    rho_hs = 2.0
    attn_hs = 0.2
    attn_units = 'dbplam'
    mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

    cmin = 1500.0
    cmax = 1799.0

    # Pekeris waveguide at each segment
    krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
    x0_list = []
    range_list = [x for x in rgrid]
    seg_i = rank
    # Make env_list 
    Z = Zvals[seg_i]
    env_z_list = [np.array([0.0, Z]), np.array([Z, Zmax])]
    env_c_list = [np.array([cw, cw]), np.array([c_hs, c_hs])]
    env_rho_list = [np.array([rho_w, rho_w]), np.array([rho_hs, rho_hs])]
    env_attn_list = [np.array([0.0, 0.0]), np.array([attn_hs, attn_hs])]
    N_list = [max(int(np.ceil(Z / mesh_dz)), 20), max(int(np.ceil(Zmax - Z)), 10)]

    if Z == Zmax: # don't need the layer domain extension
        env_z_list = [env_z_list[0]]
        env_c_list = [env_c_list[0]]
        env_rho_list = [env_rho_list[0]]
        env_attn_list = [env_attn_list[0]]
        N_list = [N_list[0]]


    nmesh_list = []
    nmesh_list = comm.gather(N_list, root=0)
    nmesh_list =  comm.bcast(nmesh_list, root=0)

    print('nmesh_list', nmesh_list)

    env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
    
    env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
    x0 = np.array([0.0])
    x0_list = []
    x0_list = comm.gather(x0, root=0)
    x0_list = comm.bcast(x0_list, root=0)

            
    env_list = comm.gather(env, root=0)


    # Now we have all the values we need to run the coupled mode model

    print('running da model')
    rdm = CMModel(range_list, env, comm)

    modes_list = rdm.run_models(x0_list)
    print('done')

    zs = np.array([25.])    
    same_grid = False
    ranges = np.linspace(100.0, 10*1e3, 1000)

    zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
    zr = zout[1:]

    p_arr = rdm.compute_field(zs, zr, ranges[1:], same_grid=same_grid, cont_part_velocity=False)
    if rank == 0:
        print([x.krs for x in modes_list])

        #p_arr = cm.compute_arr_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rcm_grid, zs, zr, ranges[1:], same_grid, cont_part_velocity=False) # False for KRAKEN model comp

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
        source = pyat_env.Source(zs)
        dom = pyat_env.Dom(ranges*1e-3, zout)
        pos = pyat_env.Pos(source, dom)
        beam = None
        cint= pyat_env.cInt(cmin, cmax)

        for seg_i in range(num_segs):
            env = env_list[seg_i]
            ssp, bdy = get_krak_inputs(env, twod=True)

            NMESH = nmesh_list[seg_i]
            NMESH = [x for x in NMESH]

            print('seg_i', seg_i)
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


if __name__ == '__main__':
    downslope_test()
