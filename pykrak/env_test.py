import numpy as np
from matplotlib import pyplot as plt
from pykrak.envs import factory

"""
Description:
Test out envs

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def get_kraken_krs():
    """ Kraken krs for 100 Hz source """
    krs = [0.4217748 -2.9627795e-06j,0.42086965-1.0816424e-05j, 0.4195981 -1.5834639e-05j,0.4178734 -2.6450240e-05j, 0.4157315 -3.5546836e-05j,0.4130856 -4.7818343e-05j, 0.40996996-6.0344391e-05j,0.4063682 -7.7885117e-05j, 0.40226972-1.0424954e-04j]
    return krs


if __name__=='__main__':
    env_builder = factory.create('swellex')
    env = env_builder()
    env.add_freq(100.0)
    cmin = np.min(env.c_list[0])
    cmax = np.min(env.c_list[1])
    env.get_krs(**{'cmin': cmin, 'cmax': cmax})
    env.get_phi()
    krs = env.krs
    krak_krs = np.sort(get_kraken_krs())
    diffs = krs - krak_krs
    print("Relative difference", abs(krs - krak_krs) / abs(krs))
