"""
Description:
Test Pekeris, compare with KRAKEN manual

Date:
1/9/2023

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

from pykrak.envs import factory

builder = factory.create('hs')
cw = np.array([1500., 1500.])
zw = np.array([0., 5000.])
attn_hs=0
N_list = [501]
rho_w = np.array([1.0, 1.0])
c_hs = 2000.
rho_hs = 2.
freq = 10.0

env = builder(zw, cw, c_hs, rho_hs, attn_hs,'dbpkmhz', pert=False)
env.add_freq(freq)
krs = env.get_krs(verbose=True, **{'N_list':N_list, 'rmax':1e8})
print('I\tK\talpha\tphase speed')
for i in range(krs.size):
    kr = krs[i]
    print('{0}\t{1}\t0.0\t{2}'.format(i+1, kr.real, 2*np.pi*freq / kr))
