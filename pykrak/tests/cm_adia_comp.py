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
from pykrak.adia_model import AdiabaticModel
from pyat.pyat import env as pyat_env
from mpi4py import MPI
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# for grid calculations
def speed_test1():
    for i in range(3):
        now = time.time()
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
        attn_hs = 0.01
        attn_units = 'dbpkmhz'
        mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

        cmin = 1500.0
        cmax = 1799.0

        # Pekeris waveguide at each segment
        krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
        nmesh_list = []
        env_list = []
        x0_list = []
        range_list = [x for x in rgrid]
        # Make env_list 
        for seg_i in range(num_segs):
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


            nmesh_list.append(N_list)


            env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
            
            env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
            env_list.append(env)
            x0_list.append(np.array([0.0]))

        # broadcast it
        #comm.bcast(env_list, root=0)
        #comm.bcast(range_list, root=0)
        #comm.bcast(x0_list, root=0)

        if rank == 0:
            print('forming environemnt objs', time.time()-now)


        # Now we have all the values we need to run the coupled mode model

        rdm = CMModel(range_list, env_list, comm)
        now = time.time()
        modes_list = rdm.run_models(x0_list)
        if rank == 0:
            print('cm model run time', time.time() - now)

        zs = np.array([25.])    
        same_grid = False
        ranges = np.linspace(100.0, 10*1e3, 1000)

        zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
        zr = zout[1:]

        now = time.time()
        p_arr = rdm.compute_field(zs, zr, ranges[1:], same_grid=same_grid, cont_part_velocity=False)
        if rank == 0:
            print('cm field comp time', time.time() - now)

        if rank == 0:
            if i == 0:
                p_tl = 20*np.log10(np.abs(p_arr)) 
                plt.figure()
                plt.suptitle('CM')
                plt.pcolormesh(ranges[1:]*1e-3, zr, p_tl)
                plt.gca().invert_yaxis()
                plt.colorbar()

        # do it with the adia model
        rdm = AdiabaticModel(range_list, env_list, comm)
        now = time.time()
        modes_list = rdm.run_models(x0_list)
        if rank == 0:
            print('adia model run time', time.time() - now)

        zs = np.array([25.])    
        same_grid = False

        zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
        zr = zout[1:]

        now = time.time()
        adia_p_arr = rdm.compute_field(zs, zr, ranges[1:])
        if rank == 0:
            print('adia field comp time', time.time() - now)

        if rank == 0:
            if i == 0:
                p_tl = 20*np.log10(np.abs(adia_p_arr)) 
                plt.figure()
                plt.suptitle('Adiabatic')
                plt.pcolormesh(ranges[1:]*1e-3, zr, p_tl)
                plt.gca().invert_yaxis()
                plt.colorbar()
    plt.show()

# for single range calculations
def speed_test2():
    for i in range(3):
        now = time.time()
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
        attn_hs = 0.01
        attn_units = 'dbpkmhz'
        mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

        cmin = 1500.0
        cmax = 1799.0

        # Pekeris waveguide at each segment
        krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
        nmesh_list = []
        env_list = []
        x0_list = []
        range_list = [x for x in rgrid]
        # Make env_list 
        for seg_i in range(num_segs):
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


            nmesh_list.append(N_list)


            env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
            
            env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
            env_list.append(env)
            x0_list.append(np.array([0.0]))

        # broadcast it
        #comm.bcast(env_list, root=0)
        #comm.bcast(range_list, root=0)
        #comm.bcast(x0_list, root=0)

        if rank == 0:
            print('forming environemnt objs', time.time()-now)


        # Now we have all the values we need to run the coupled mode model

        rdm = CMModel(range_list, env_list, comm)
        now = time.time()
        modes_list = rdm.run_models(x0_list)
        if rank == 0:
            print('cm model run time', time.time() - now)

        zs = np.array([25.])    
        same_grid = False
        rs_grid = np.array([10000.0])

        zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
        zr = zout[1:]

        now = time.time()
        p_arr = rdm.compute_field(zs, zr, rs_grid, same_grid=same_grid, cont_part_velocity=False)
        if rank == 0:
            print('cm field comp time', time.time() - now)

        if rank == 0:
            if i == 0:
                p_tl = 20*np.log10(np.abs(p_arr)) 
                fig, axes = plt.subplots(2,1, sharex=True)
                plt.suptitle('CM')
                axes[0].plot(zr, p_tl)
                axes[1].plot(zr, np.angle(p_arr))

        # do it with the adia model
        rdm = AdiabaticModel(range_list, env_list, comm)
        now = time.time()
        modes_list = rdm.run_models(x0_list)
        if rank == 0:
            print('adia model run time', time.time() - now)

        zs = np.array([25.])    
        zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
        zr = zout[1:]

        now = time.time()
        adia_p_arr = rdm.compute_field(zs, zr, rs_grid)
        if rank == 0:
            print('adia field comp time', time.time() - now)

        if rank == 0:
            if i == 0:
                p_tl = 20*np.log10(np.abs(adia_p_arr)) 
                fig, axes = plt.subplots(2,1, sharex=True)
                plt.suptitle('Adia')
                axes[0].plot(zr, p_tl)
                axes[1].plot(zr, np.angle(adia_p_arr))
    plt.show()



if __name__ == '__main__':
    speed_test2()
