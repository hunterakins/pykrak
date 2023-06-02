"""
Description:
Calculate normal modes and TL for Pekeris waveguide

Date:
4/18/2023

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

from pykrak.pykrak_env import Env

z_list = [np.array([0, 5000.])]
c_list = [np.array([1500., 1500.])]
rho_list = [np.ones(2)]
attn_list = [.001*np.ones(2)]
c_hs = 2000.
rho_hs = 2.0
attn_hs = 0.0
attn_units ='dbpkmhz'
freq = 10.

env = Env(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units)
env.add_freq(freq)
rmax = 1e3
krs = env.get_krs(**{'rmax':rmax, 'Nh':1})
print(krs)
phi = env.phi

zr = np.array([2500.])
zs = np.array([500.])
phi_z = env.get_phi_z()
phi = env.phi
r = np.linspace(2*1e5, 2.2*1e5, 1000)

from pykrak import pressure_calc as pc
p = pc.get_grid_pressure(zr, phi_z, phi, krs, zs, r)
tmp_r = np.array([1.0])
tmp_p = pc.get_grid_pressure(zr, phi_z, phi, krs, zr, tmp_r)
print(abs(tmp_p), 1/(4*np.pi))

print(p.shape)
p = np.squeeze(p)
tl = -20*np.log10(np.sqrt(2*np.pi)*abs(p)) # this is actually incorrect, it should be 4pi
# this factor gets agreement with the KRAKEN manual. I think the figure in the manual does not account for the fact that the pressure field calculation has a factor of 1/sqrt(8 pi) in it 
plt.figure()
plt.plot(r*1e-3, tl,'k')
plt.xlabel('Range (km)')
plt.ylabel('Loss (dB)')
plt.annotate('f = 10 Hz\nsd=500 m\n rd=2500 m', xy=(0.9, 0.9), xycoords='axes fraction')
plt.ylim([110, 70])
plt.show()



