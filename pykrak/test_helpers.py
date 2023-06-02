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
from pyat.pyat.readwrite import read_env, read_modes
from pyat.pyat.env import SSP, plot_ssp
from pykrak.pykrak_env import Env
import os

def read_krs_from_prt_file(prt_file):
    """
    .mod files have krs in single precision
    .prt files are 10 digit rounded krs in double precision
    which is more accurate
    """
    with open(prt_file, 'r') as x:
        i = 0
        lines = x.readlines()
        for line in lines:
            line = line.strip(' ')
            if line[0] == 'I':
                print(i)
                break
            i += 1
        i += 1 # now i is the first mode one
        krs = []
        while True:
            line = lines[i]
            line = line.strip(' ')
            if line[0] == '_':
                return krs      
            split_line = line.split(' ')
            krs.append(float(split_line[2]))
            i += 1
            
def init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax):
    """
    Initialize a pykrak env obj from the values read in from the env files
    """

    N_list = ssp.N
    z_list = [np.array(x.z) for x in ssp.raw]
    c_list = [np.array(x.alphaR) for x in ssp.raw]
    rho_list = [np.array(x.rho) for x in ssp.raw]
    attn_list = [np.array(x.alphaI) for x in ssp.raw]
    bot_bdry = bdry.Bot
    opt = bot_bdry.Opt
    hs = bot_bdry.hs
    # I don't handle shear
    c_hs = hs.alphaR
    if hs.betaR != 0:
        print('Warning. Ignoring halfspace shear')
    rho_hs = hs.rho
    c_hs = float(c_hs)
    rho_hs = float(rho_hs)
    attn_hs = hs.alphaI
    top_opt = bdry.Top.Opt
    top_opt = top_opt.strip()
    atten_opt = top_opt[2]
    atten_opts = ['N', 'F', 'M', 'W', 'Q']
    py_opts = ['npm', 'dbpm', 'dbplam', 'dbpkmhz', 'q']
    attn_units = py_opts[atten_opts.index(atten_opt)]
    env= Env(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units)
    env.add_freq(freq)
    return env, N_list
