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
from pykrak import krak_routines as kr
import os


def kr_comp():
    env_files = ['ice_env.env', 'solve2test.env', 'pekeris_layer_attn.env','pekeris.env',  'pekeris_attn.env']
    for env in env_files:
        os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env('at_files/{}.env'.format(env[:-4]), 'kraken')
        c_low, c_high = cint.Low, cint.High
        attn_units = 'dbplam'

        modes = rw.read_modes(**{'fname':'at_files/{}.mod'.format(env[:-4]),
                                'freq':freq})
        krak_k = modes.k
        krak_modes = modes.phi
        z=modes.z
        print('z0', z[0])

        krak_prt_k, mode_nums, vps, vgs = th.read_krs_from_prt_file('at_files/{}.prt'.format(env[:-4]), verbose=True)
        print('krak M', modes.M)

        RMax = 0.0


        pykrak_env, N_list, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_hs, cs_hs, rho_hs, attnp_hs, attns_hs, pos, beam, cint, RMax = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
        print('N_list', N_list)

        rho_top = 0.0
        cp_top = 0.0
        cs_top = 0.0
        attnp_top = 0.0
        attns_top = 0.0

        pk_krs, phi_z, phi, ugs = kr.list_input_solve(freq, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_top, cs_top, rho_top, attnp_top, attns_top, cp_hs, cs_hs, rho_hs, attnp_hs, attns_hs, 'dbplam', N_list, 0.0, c_low, c_high)
        #pk_krs = pykrak_env.get_krs(**{'N_list':N_list, 'Nh':1, 'cmax':pykrak_env.c_hs-1e-10})
        #phi_z, phi = pykrak_env.phi_z, pykrak_env.phi
        #ugs = pykrak_env.get_ugs(N_list=N_list)
        
        print('pykrak M', pk_krs.size)
        for i in range(len(krak_prt_k)):
            mode_num = mode_nums[i]
            krak_k = str(krak_prt_k[i].real)
            pk_k = str(pk_krs[mode_num-1].real)
            print(krak_prt_k[i], pk_krs[mode_num-1])
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
    env_files = ['solve2test.env', 'pekeris_layer_attn.env', 'pekeris_attn.env', 'pekeris.env']
    for env in env_files:
        os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env('at_files/{}.env'.format(env[:-4]), 'kraken')

        c_low, c_high = cint.Low, cint.High

        modes = rw.read_modes(**{'fname':'at_files/{}.mod'.format(env[:-4]),
                                'freq':freq})
        krak_k = modes.k
        krak_modes = modes.phi
        rho = modes.rho


        z=modes.z
        print('z0', z[0])

        print('krak M', modes.M)

        #pykrak_env, N_list = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
        pykrak_env, N_list, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_hs, cs_hs, rho_hs, attnp_hs, attns_hs, pos, beam, cint, RMax = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
        print('N_list', N_list)

        rho_top = 0.0
        cp_top = 0.0
        cs_top = 0.0
        attnp_top = 0.0
        attns_top = 0.0
    
        pk_krs, phi_z, phi, ugs = kr.list_input_solve(freq, z_list, cp_list, cs_list, rho_list, attnp_list, attns_list, cp_top, cs_top, rho_top, attnp_top, attns_top, cp_hs, cs_hs, rho_hs, attnp_hs, attns_hs, 'dbplam', N_list, 0.0, c_low, c_high)
        #pk_krs = pykrak_env.get_krs(**{'N_list':N_list, 'Nh':1, 'cmax':pykrak_env.c_hs-1e-10})
        #phi_z, phi = pykrak_env.phi_z, pykrak_env.phi
        #print('phiz', phi_z)  

        print(z, phi_z)


        phi_new = np.zeros((z.size, phi.shape[1]))
        for i in range(phi.shape[1]):
            phi_new[:,i] = np.interp(z, phi_z, phi[:,i])

        phi = phi_new
        phi_z = z
        plt.figure()
        plt.plot(phi[:,0]/np.sum(np.abs(phi[:,0])), phi_z, label='PyKrak')
        plt.plot(krak_modes[:,0].real/np.sum(np.abs(phi[:,0])), z, label='Kraken')
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(phi[:,1], phi_z, label='PyKrak')
        plt.plot(krak_modes[:,1].real, z, label='KRAKEN')
        plt.legend()
        plt.ylabel('z (m)')
        plt.figure()
        plt.plot(phi[:,2], phi_z, label='PyKrak')
        plt.plot(krak_modes[:,2].real, z, label='KRAKEN')
        plt.legend()
        plt.ylabel('z (m)')

        phi /= np.sum(np.abs(phi), axis=0)
        krak_modes /= np.sum(np.abs(krak_modes), axis=0)
        plt.figure()
        plt.suptitle('Difference in mode shape from PyKrak and KRAKEN\nMode 1')
        plt.plot(phi[:,0] - krak_modes[:,0], phi_z)
        plt.ylabel('z (m)')

        plt.figure()
        plt.plot(phi[:,1] - krak_modes[:,1], phi_z)
        plt.suptitle('Difference in mode shape from PyKrak and KRAKEN\nMode 2')
        plt.ylabel('z (m)')
        plt.figure()
        plt.plot(phi[:,2] - krak_modes[:,2], phi_z)
        plt.suptitle('Difference in mode shape from PyKrak and KRAKEN\nMode 3')
        plt.ylabel('z (m)')
        plt.show()

kr_comp()
phi_comp()
