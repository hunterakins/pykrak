"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib
from pyat.pyat.readwrite import read_env, read_modes
from pyat.pyat import env as pyat_env
from pykrak.pykrak_env import Env
import os

def read_krs_from_prt_file(prt_file,verbose=True):
    """
    .mod files have krs in single precision
    .prt files are 16 digit rounded krs in double precision
    which is more accurate
    """
    with open(prt_file, 'r') as x:
        i = 0
        lines = x.readlines()
        for line in lines:
            line = line.strip(' ')
            if line[0] == 'I':
                if verbose:
                    print(i)
                i += 1
                break
            i += 1
        i += 1 # now i is the first mode one
        krs = []
        mode_nums = [] # krak will only print subset
        vps = []
        vgs = []
        while True:
            line = lines[i]
            line = line.strip(' ')
            if line[0] == '_':
                return krs, mode_nums, vps, vgs
            split_line = line.split(' ')
            split_line = [x for x in split_line if x != '']
            if verbose:
                print(split_line)
            if float(split_line[2]) == 0.0:
                krs.append(float(split_line[1]))
            else:
                krs.append(float(split_line[1])+1j*float(split_line[2])) 
            vp = float(split_line[3])
            vg = float(split_line[4])
            vps.append(vp)
            vgs.append(vg)

            mode_nums.append(int(split_line[0]))
            i += 1
    return krs, mode_nums, vps, vgs
            
def init_pykrak_env(freq, ssp, bdry, pos, beam, cint, RMax):
    """
    Initialize a pykrak env obj from the values read in from the env files
    """

    N_list = ssp.N
    z_list = [np.array(x.z) for x in ssp.raw]
    c_list = [np.array(x.alphaR) for x in ssp.raw]
    cs_list = [np.array(x.betaR) for x in ssp.raw]
    rho_list = [np.array(x.rho) for x in ssp.raw]
    attn_list = [np.array(x.alphaI) for x in ssp.raw]
    attns_list = [np.array(x.betaI) for x in ssp.raw]
    #attn_list = [2*np.pi*freq *x / y**2 for x,y in zip(cI_list, c_list)]
    # this is npm ...so need to reverse convert?
    bot_bdry = bdry.Bot
    opt = bot_bdry.Opt
    hs = bot_bdry.hs
    # I don't handle shear
    if opt[0] == 'A':
        c_hs = hs.alphaR
        rho_hs = hs.rho
        c_hs = float(c_hs)
        cs_hs = float(hs.betaR)
        rho_hs = float(rho_hs)
        attn_hs = hs.alphaI
        attns_hs = hs.betaI
    elif opt[0] == 'R':
        rho_hs = 1e10
        c_hs = 000.0
        attn_hs = 000.0
        cs_hs = 0.0
        attns_hs = 0.0


    top_opt = bdry.Top.Opt
    top_opt = top_opt.strip()
    atten_opt = top_opt[2]
    atten_opts = ['N', 'F', 'M', 'W', 'Q']
    py_opts = ['npm', 'dbpkmhz', 'dbpm', 'dbplam', 'q']
    attn_units = py_opts[atten_opts.index(atten_opt)]
    print('attn units', attn_units)
    env= Env(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units)
    env.add_freq(freq)
    N_list = [x+1 for x in N_list] # kraken doesn't count end points?
    return env, N_list, z_list, c_list, cs_list, rho_list, attn_list, attns_list, c_hs, cs_hs, rho_hs, attn_hs, attns_hs, pos, beam, cint, RMax

def get_krak_inputs(env, twod=False):
    """
    Get ssp object and bdy object for acoustics toolbox
    from pykrak object
    twod flag is for bottom halfspace 
    """
    layer_list = []
    N_list = []
    for i in range(len(env.z_list)): # for each layer
        z = env.z_list[i]
        c = env.c_list[i]
        rho = env.rho_list[i]
        attn = env.attn_list[i]

        ssp = pyat_env.SSPraw(z, c, np.zeros(z.size), rho, attn, np.zeros(z.size))
        layer_list.append(ssp)
        N_list.append(z.size)
    depths = [0] + [x[-1] for x in env.z_list] # layer depths
    Nmedia = len(layer_list)
    ssp = pyat_env.SSP(layer_list, depths, Nmedia, N=N_list)

    atten_opts = ['N', 'F', 'M', 'W', 'Q']
    py_opts = ['npm', 'dbpkmhz', 'dbpm', 'dbplam', 'q']
    attn_units = atten_opts[py_opts.index(env.attn_units)]
    print('attn units', attn_units)

    topbdry = pyat_env.TopBndry('CV' + attn_units)
    hs = pyat_env.HS(env.c_hs, 0, env.rho_hs, env.attn_hs, 0)
    bott_opt = 'A'
    if twod:
        bott_opt += '~'
    botbdry = pyat_env.BotBndry(bott_opt, hs)
    bndry = pyat_env.Bndry(topbdry, botbdry)
    return ssp, bndry

