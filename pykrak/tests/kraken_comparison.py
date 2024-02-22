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


env_files = ['pekeris_layer_attn.env', 'pekeris.env', 'pekeris_attn.env']
for env in env_files:
    os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
    TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env('at_files/{}.env'.format(env[:-4]), 'kraken')

    modes = rw.read_modes(**{'fname':'at_files/{}.mod'.format(env[:-4]),
                            'freq':freq})
    krak_k = modes.k
    krak_modes = modes.phi
    print(krak_modes.shape)
    z=modes.z
    print('z0', z[0])

    krak_prt_k, mode_nums = th.read_krs_from_prt_file('at_files/{}.prt'.format(env[:-4]))
    print('krak M', modes.M)

    pykrak_env, N_list = th.init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax)
    pk_krs = pykrak_env.get_krs(**{'N_list':N_list, 'Nh':1, 'cmax':pykrak_env.c_hs-1e-10})
    phi_z, phi = pykrak_env.phi_z, pykrak_env.phi
    
    print('pykrak M', pk_krs.size)
    for i in range(len(krak_prt_k)):
        mode_num = mode_nums[i]
        krak_k = str(krak_prt_k[i].real)
        pk_k = str(pk_krs[mode_num-1].real)
        print('krak, pk', krak_k, pk_k)
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



    

