"""
Description:
    Adiabatic model object for managing an adiabatic model run.

Date:
    9/22/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak import pykrak_env, pressure_calc
from pykrak import linearized_model as lm
from pykrak import range_dep_model as rdm
from pykrak.adia_modes import compute_arr_adia_pressure
from matplotlib import pyplot as plt
import mpi4py.MPI as MPI
from interp.interp import get_spline, splint, vec_splint, vec_lin_int
import time

class AdiabaticModel(rdm.RangeDepModel):
    def __init__(self, range_list, env, comm):
        super().__init__(range_list, env, comm)

    def get_phi_zr_arr(self, zr): 
        rank = self.comm.Get_rank()
        modes = self.modes
        phi = modes.phi
        Nr = zr.size
        phi_zr = np.zeros((Nr, modes.M))
        for mi in range(modes.M):
            phi_zr[:,mi] = vec_lin_int(zr, modes.z, phi[:,mi])
        phi_zr_list = self.comm.gather(phi_zr, root=0)
        if rank == 0:
            M_max = max([x.M for x in self.modes_list])
            phi_zr_arr = np.zeros((self.n_env, Nr, M_max))
            for i in range(self.n_env):
                M = phi_zr_list[i].shape[1]
                phi_zr_arr[i,:,:M] = phi_zr_list[i]
            return phi_zr_arr
        else:
            return None

    def compute_field(self, zs, zr, rs_grid):
        """
        Compute the field at a set of receiver locations for a set of source depths
        Assume that the environments have been supplied in order, with the first environment
        corresponding to the source and the last environment corresponding to the receiver.
        The range associated with the environment is considered to be centered on the region
        it describes.
        """ 
        now = time.time()
        phi_zr_arr = self.get_phi_zr_arr(zr)
        if self.comm.rank == 0:
            interface_range_list = self._get_interface_ranges()
            M_list = [x.M for x in self.modes_list]
            modes = self.modes # for source segment this is always rank 0 
            M_max = max(M_list)
            phi_zs = np.zeros((zs.size, M_max))
            for i in range(modes.M):
                phi_zs[:,i] = vec_lin_int(zs, modes.z, modes.phi[:,i])

            rgrid, kr_arr = self._get_kr_arr(M_max) # throw out modes that don't exist at every range
            # compute the field at the receiver
            
            field = np.zeros((zs.size, zr.size), dtype=np.complex128)
            p = compute_arr_adia_pressure(kr_arr, phi_zs, phi_zr_arr, rgrid, rs_grid)
            field = np.squeeze(p)
        else:
            field = np.zeros((1))
        return field

