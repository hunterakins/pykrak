"""
Description:
    Helpers for the internal wave mode solver

Date:
    5/19/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib
import gsw



def get_Nz(z, p, t, c):
    """
    Given grid of depths z with temperatue c and salinity s:
    get buoyancy frequency on that grid
    """
    sp = gsw.SP_from_C(c, t, p)
    sa = gsw.SA_from_SP(sp)
    ct = gsw.CT_from_t(sa, t, p)
    rho = gsw.density.rho(sa,ct,p)

    # comopute deriv
    drho = rho[1:] - rho[:-1]
    dz = z[1:] - z[:-1]
    z_grid = (z[1:] + z[:-1])/2
    drhodz = drho / dz

    # smooth?
    Nz_sq = -9.81 / rho[1:] * drhodz
    Nz = np.sqrt(Nz_sq) # assume its stable?
    return Nz


def get_wI(lat):
    """
    latitude in degrees
    get inertial frequency in rad/s
    """
    freqi=np.sin(lat*np.pi/180)/12.0 # 2 Omega sin(lat)...2*Omega is twice earth's rotation rate which in hertz is 2 cycles / 24 hours...so this is in cycles / hrs
    fac =2.0*np.pi/3600.0 # convert to rad / s
    omegai=fac*freqi
    return omegai

def dt_power_spec(k, E0):
    """
    Dozier Tappert power spectrum
    """
    #P•(k) 2 No•w,E ø (t- 1)j/j* k • (• +j/j )  (• + 
    return

        
