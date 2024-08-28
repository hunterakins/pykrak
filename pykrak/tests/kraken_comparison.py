"""
Description:
    Compare pykrak wavenumber estimates to kraken using same mesh

Date:
    10/29/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt

from pykrak import test_helpers as th
from pyat.pyat import readwrite as rw
from pykrak import pykrak_env as pke

import os


def kr_comp():
    env_files = ['pekeris_layer_attn.env', 'pekeris.env', 'pekeris_attn.env']
    for env in env_files:
        os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env('at_files/{}.env'.format(env[:-4]), 'kraken')

        modes = rw.read_modes(**{'fname':'at_files/{}.mod'.format(env[:-4]),
                                'freq':freq})
        krak_k = modes.k
        krak_modes = modes.phi
        z=modes.z
        print('z0', z[0])

        krak_prt_k, mode_nums, vps, vgs = th.read_krs_from_prt_file('at_files/{}.prt'.format(env[:-4]), verbose=True)
        print('krak M', modes.M)

        pykrak_env, N_list = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
        pk_krs = pykrak_env.get_krs(**{'N_list':N_list, 'Nh':1, 'cmax':pykrak_env.c_hs-1e-10})
        phi_z, phi = pykrak_env.phi_z, pykrak_env.phi
        ugs = pykrak_env.get_ugs(N_list=N_list)
        
        print('pykrak M', pk_krs.size)
        for i in range(len(krak_prt_k)):
            mode_num = mode_nums[i]
            krak_k = str(krak_prt_k[i].real)
            pk_k = str(pk_krs[mode_num-1].real)
            print('krak, pk, krak vg, pvg', krak_k, pk_k, vgs[i], ugs[mode_num-1])
            num_dig= 0 
            count = 0
            while count < (min(len(krak_k), len(pk_k))) and krak_k[count] == pk_k[count]:
                count += 1
            plt.plot(mode_num, count-1, 'ko') # exclude decimal
        plt.xlabel('Mode num')
        plt.ylabel('Number of digits in agreement \n(Includes leading zero')
        plt.grid()
        plt.suptitle('Comparison of pykrak and kraken wavenumbers for {}'.format(env))
        plt.show()


def phi_comp():
    env_files = ['pekeris_attn.env', 'pekeris.env', 'pekeris_layer_attn.env']
    for env in env_files:
        os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env('at_files/{}.env'.format(env[:-4]), 'kraken')

        modes = rw.read_modes(**{'fname':'at_files/{}.mod'.format(env[:-4]),
                                'freq':freq})
        krak_k = modes.k
        krak_modes = modes.phi
        rho = modes.rho


        z=modes.z
        print('z0', z[0])

        print('krak M', modes.M)

        pykrak_env, N_list = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
        pk_krs = pykrak_env.get_krs(**{'N_list':N_list, 'Nh':1, 'cmax':pykrak_env.c_hs-1e-10})
        phi_z, phi = pykrak_env.phi_z, pykrak_env.phi
        print('phiz', phi_z)  

        plt.figure()
        plt.plot(phi[:,0], phi_z)
        plt.plot(krak_modes[:,0].real, phi_z)

        plt.figure()
        plt.plot(phi[:,0] - krak_modes[:,0], phi_z)
        plt.figure()
        plt.plot(phi_z[1:], phi[1:,0] / krak_modes[1:,0])

        plt.figure()
        plt.plot(phi[:,1] - krak_modes[:,1], phi_z)
        plt.figure()
        plt.plot(phi_z[1:], phi[1:,1] / krak_modes[1:,1])
        plt.show()

kr_comp()
#phi_comp()
