"""
Description:
    Coupled mode model object for managing an coupled model run.

Date:
    9/22/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak import pykrak_env, pressure_calc
from pykrak import linearized_model as lm
from pykrak import coupled_modes as cm
from pykrak import range_dep_model as rdm
from matplotlib import pyplot as plt
import mpi4py.MPI as MPI
from interp.interp import get_spline, splint, vec_splint, vec_lin_int
import time
from numba import njit, jit

class CMModel(rdm.RangeDepModel):
    def __init__(self, range_list, env_list, comm):
        super().__init__(range_list, env_list, comm)
        #self.env_list = env_list # list of range-iidependent linearized env obnjects
        #self.n_env = len(env_list)
        #self.range_list = range_list # list of range that represents the env
        #self.ri = ri

    def compute_field(self, zs, zr, rs_grid, same_grid=False, cont_part_velocity=True):
        """
        Compute the field at a set of receiver locations for a set of source depths
        Assume that the environments have been supplied in order, with the first environment
        corresponding to the source and the last environment corresponding to the receiver.
        The range associated with the environment is considered to be centered on the region
        it describes.

        same_grid - True if you have made all the environments use the same mesh.
            saves time by not having to interpolate
        """
        if self.comm.rank == 0:
            interface_range_list = self._get_interface_ranges()
            rgrid = np.array(interface_range_list)
            M_list = [x.M for x in self.modes_list]
            M_max = max(M_list)
            modes_list = self.modes_list
            env_list = self.env_list

            rho_list = [x.get_rho_grid(x.N_list) for x in env_list]
            krs_list = [x.krs for x in modes_list]
            phi_list = [x.phi for x in modes_list]
            zgrid_list= [x.z for x in modes_list]
            c_hs_list = [x.c_hs for x in env_list]
            rho_hs_list = [x.rho_hs for x in env_list]

            omega = 2*np.pi*self.env_list[0].freq
            p = cm.compute_arr_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs_grid, same_grid, cont_part_velocity=cont_part_velocity)
            field = np.squeeze(p)
            return field
        else:
            return np.array([0.0])

