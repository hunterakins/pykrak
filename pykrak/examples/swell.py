"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from env.env.envs import factory
from pykrak.sturm_seq import get_krs, get_A_numba, get_arrs
from pykrak.inverse_iteration import get_phi
import time

def test_swell_sturm():
    loader = factory.create('swellex')
    env = loader()
    print(env.rhob)

    cw = env.cw
    cb = env.cb
    rhob = env.rhob
    env.rhob[:2] = 2.5
    env.rhob[2:] = 3.2
    rhob = np.round(rhob, 1)

    krs = []
    for h1 in [.5, .1, .01]:
        now = time.time()
        z_layer1 = np.arange(0.0, env.z_ss[-1] + h1, h1)
        cw =  cw.reshape(cw.size)
        c_layer1 = np.interp(z_layer1, env.z_ss, cw) 
        
        h = h1
        h2 = 1
        h3 = 5
        z_layer_2 = np.arange(env.z_sb[0], env.z_sb[1] + h2, h2)
        z_layer_3 = np.arange(env.z_sb[2], env.z_sb[3] + h3, h3)

        c_layer2 = cb[0] + (cb[1]-cb[0])/(env.z_sb[1] - env.z_sb[0])*(z_layer_2-env.z_sb[0]) #linear ssps...
        c_layer3 = cb[2] + (cb[3]-cb[2])/(env.z_sb[3] - env.z_sb[2])*(z_layer_3 - env.z_sb[2])
        c_hs = cb[-1]

        rho_layer1 = 1.0*np.ones(z_layer1.size)
        rho_layer2 = rhob[0]+ (rhob[1]-rhob[0])/(env.z_sb[1] - env.z_sb[0])*z_layer_2 #
        rho_layer3 = rhob[2]+ (rhob[3]-rhob[2])/(env.z_sb[3] - env.z_sb[2])*z_layer_3 #
        rho_hs = env.rhob[-1]



        h_list = [h1, h2, h3]
        z_list = [z_layer1, z_layer_2, z_layer_3]
        c_list  = [c_layer1, c_layer2, c_layer3]
        rho_list = [rho_layer1, rho_layer2, rho_layer3]
        h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(h_list, z_list, c_list, rho_list)

        #plt.figure()
        #for i in range(3):
        #    plt.plot(c_list[i], z_list[i],'.')
        #plt.gca().invert_yaxis()

        #plt.figure()
        #for i in range(3):
        #    plt.plot(rho_list[i], z_list[i])
        #plt.gca().invert_yaxis()

        freq = 53.0
        omega = 2*np.pi*freq

        kmin = omega / c_layer3[0] 
        kmin = omega / 1800.0
        kmax = omega / np.min(cw)  - 1e-10

        lam_min = np.square(h*kmin)
        lam_max = np.square(h*kmax)
        c_hs = float(c_hs)
        rho_hs = float(rho_hs)
        a_diag, e, d = get_A_numba(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, .5*(lam_min+lam_max))
        kr = get_krs(omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs, lam_min, lam_max)
        kr = np.array(kr)
        phi=  get_phi(kr, omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, c_hs, rho_hs)
        krs.append(kr)
        print('running sturm', time.time()-now)


    zs = 50.0
    zr = [50.0]
    env.add_source_params(freq, zs, zr) # 
    #env.add_field_params(h, np.max(z), 10.0, 1000.0)
    env.add_field_params(h, 216.5+h, 10.0, 1000.0)
    env.attn *= 0.0
    custom_r = np.array([10000]) # just  a dummy range...I just want modes
    now = time.time()
    p, pos = env.run_model('kraken_custom_r', 'at_files/', 'iso', zr_flag=False, \
                                zr_range_flag=False, custom_r=custom_r)
    print('kraken time', time.time()-now)
    print('num kraken modes', env.modes.k.size)
    print('num sturm modes', len(kr))
    plt.figure()
    plt.plot(env.modes.k.real[::-1],[1]*env.modes.M, 'k+')
    for kr in krs:
        plt.plot(kr, [1]*len(kr), '.')
    plt.grid()
    plt.show()


    plt.figure()
    plt.plot(phi, color='b')
    plt.plot(env.modes.phi.real, color='r')
    plt.grid()
    plt.show()

test_swell_sturm()
