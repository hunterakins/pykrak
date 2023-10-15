"""
Description:
    Range dependent modal model
    Use MPI to run range-independent normal mode calculations in parallel

Date:
    9/22/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak import pykrak_env, pressure_calc
from pykrak import linearized_model as lm
import mpi4py.MPI as MPI
from interp import interp
import time
from numba import njit, jit

@njit(cache=True)
def range_interp_krs(rpt, rgrid, kr_arr):
    """
    kr_arr is values of horiz. wavenumbers at different
    ranges
    rgrid is the range grid at which the environments that these
    modes are definesd
    rpt is the desired new point to interpolate the wavenumbers
    """
    if rpt > rgrid.max() or rpt < rgrid.min():
        raise ValueError("rpt is outside of the range of the model")
    else:
        klo, khi = interp.get_klo_khi(rpt, rgrid)
        dr = rgrid[khi] - rgrid[klo]
        kr_left = kr_arr[:, klo]
        kr_right = kr_arr[:, khi]
        a = (rgrid[khi] - rpt) / dr
        b = (rpt - rgrid[klo]) / dr
        # only include modes that exist at both ranges
        kr_inds = (kr_left.real > 0) & (kr_right.real > 0)
        kr_vals = a*kr_left[kr_inds] + b*kr_right[kr_inds]
    return kr_vals

class RangeDepModel:
    def __init__(self, range_list, env_list, comm):
        self.env_list = env_list # list of range-independent linearized env obnjects
        self.n_env = len(env_list)
        self.range_list = range_list # list of range that represents the env
        self.comm = comm


    def run_models(self, freq, x0_list):
        """
        Solve for krs and mode shapes for each environment
        """
        modes_list = []
        rank = self.comm.Get_rank()
        env = self.env_list[rank]
        x0 = x0_list[rank]
        env.add_freq(freq)
        env.add_x0(x0)
        modes = env.full_forward_modes()
        modes_list = self.comm.gather(modes, root=0)
        self.modes_list = modes_list
        return modes_list

    def _get_interface_ranges(self):
        """
        Ranges in range_list are centered on a range-independent region
        This function returns the ranges of the interfaces between regions, 
        taken to be halfway between the updated ranges
        """
        interface_range_list = [self.range_list[0]]
        for i in range(self.n_env-1):
            interface_range_list.append((self.range_list[i+1] + self.range_list[i])/2)
        """
        Now add final interface range list with 100 meters of play
        """
        interface_range_list.append(self.range_list[-1] + 100) 
        return interface_range_list

    def _get_kr_arr(self, M_max):
        """
        Get a matrix of kr values
        Each column corresponds to an environmnet range
        Return the range values as well
        """
        rgrid = np.array(self.range_list)
        kr_arr = -np.ones((M_max, rgrid.size), dtype=np.complex128)
        for i in range(self.n_env):
            M = self.modes_list[i].M
            kr_arr[:M, i] = self.modes_list[i].krs[:M]
        return rgrid, kr_arr

    def _get_mean_krs(self, rs, M_max):
        """
        Get the averaged wavenumber for a receiver at range rs
        Using the trapezoid rule
        """
        rgrid, kr_arr = self._get_kr_arr(M_max)
        if rs > rgrid[-1]:
            raise ValueError("rs is outside of the range of the model")
        # insert rs into the grid of model ranges and interpolate kr wavenumbers
        # to that range
        if rs not in rgrid: 
            kr_vals = range_interp_krs(rs, rgrid, kr_arr)
            tmp = -1.0*np.ones((M_max), dtype=np.complex128)
            tmp[:kr_vals.size] = kr_vals
            kr_vals = tmp
            klo, khi = interp.get_klo_khi(rs, rgrid)
            rgrid = np.insert(rgrid, khi, rs)
            kr_arr = np.insert(kr_arr,  khi, kr_vals, axis=1)



        # now truncate to ranges less than or equal to rs
        inds = rgrid <= rs
        rgrid = rgrid[inds]
        if rgrid[-1] != rs:
            raise ValueError('Bug in the code in get_mean_krs')
        kr_arr = kr_arr[:, inds]

        # now eliminate modes that don't exist at all ranges
        kr_inds = np.all(kr_arr.real > 0, axis=1)

        kr_arr = kr_arr[kr_inds, :]
        M = kr_arr.shape[0]
        mean_krs = np.zeros(M, dtype=np.complex128)
        for i in range(M):
            mean_krs[i] = np.trapz(kr_arr[i,:], rgrid)/rgrid[-1]
        return mean_krs

    def get_phi_zs(self, zs, M):
        src_modes = self.modes_list[0]
        phi_zs = src_modes.get_phi_zr(zs, M)
        #phi_zs = phi_zs[:,:M]
        return phi_zs

    def get_phi_zr(self, zr, M, rgrid, rs):
        rk_rcvr = np.argmin(np.abs(rgrid - rs)) # this is the range-independent segment that the receiver is in
        rcvr_modes = self.modes_list[rk_rcvr]
        phi_zr = rcvr_modes.get_phi_zr(zr, M)
        #phi_zr = phi_zr[:,:M]
        return phi_zr


