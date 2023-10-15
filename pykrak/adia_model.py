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
from matplotlib import pyplot as plt
import mpi4py.MPI as MPI
from interp.interp import get_spline, splint, vec_splint, vec_lin_int
import time
from numba import njit, jit

class AdiabaticModel(rdm.RangeDepModel):
    def __init__(self, range_list, env_list, comm):
        super().__init__(range_list, env_list, comm)
        #self.env_list = env_list # list of range-iidependent linearized env obnjects
        #self.n_env = len(env_list)
        #self.range_list = range_list # list of range that represents the env
        #self.ri = ri

    def compute_field(self, zs, zr, rs):
        """
        Compute the field at a set of receiver locations for a set of source depths
        Assume that the environments have been supplied in order, with the first environment
        corresponding to the source and the last environment corresponding to the receiver.
        The range associated with the environment is considered to be centered on the region
        it describes.
        """
        if self.comm.rank == 0:
            interface_range_list = self._get_interface_ranges()
            M_list = [x.M for x in self.modes_list]
            M_max = max(M_list)
            #M = min(M_list)

            rgrid, kr_arr = self._get_kr_arr(M_max) # throw out modes that don't exist at every range
            mean_krs = self._get_mean_krs(rs, M_max) # integrate over ranges to get mean kr to use with rs in the pressure calc
            M = mean_krs.size
            #plt.figure()
            #plt.plot(rgrid, kr_arr[0,:])
            #plt.plot(rs/2, mean_krs[0], 'o')
            #plt.show()

            # compute the field at the receiver
            
            field = np.zeros((zs.size, zr.size), dtype=np.complex128)
            phi_zs = self.get_phi_zs(zs, M)
            phi_zr = self.get_phi_zr(zr, M, rgrid, rs)
            p = pressure_calc.get_arr_pressure(phi_zs, phi_zr, mean_krs, np.array([rs]))
            field = np.squeeze(p)
        return field

@jit
def single_mode_freq_spl_interp(des_freq_arr, freq_arr, kr_arr, phi_zs_vals_arr, phi_zr_vals_arr):
    """
    Interpolate a single modes values onto a frequency grid
    des_freq_arr - desired frequencies, assumed to be wihtin the bounds of freq_arr
    freq_arr- frequencies at which the modes have been computed
    kr_arr - kr value at each frequency
    phi_zs_vals_arr - phi_zs values at each frequency
    phi_zr_vals_arr - phi_zr values at each frequency
    """
    if des_freq_arr.min() < freq_arr.min() or (des_freq_arr.max() > freq_arr.max()):
        raise ValueError('Attemptimg to extrapolate beyond frequency bounds')

    num_zs = phi_zs_vals_arr.shape[0]
    num_zr = phi_zr_vals_arr.shape[0]
    num_freq_out = des_freq_arr.size
    phi_zs_out = np.zeros((num_zs, num_freq_out))
    phi_zr_out = np.zeros((num_zr, num_freq_out))

    """
    First interpolate the wave number
    """
    kr_r_spline = get_spline(freq_arr, kr_arr.real, 1e30, 1e30) # uses default 0 second derivative on edges
    kr_i_spline = get_spline(freq_arr, kr_arr.imag, 1e30, 1e30) # uses default 0 second derivative on edges
    kr_r_out, dkr_df_out = vec_splint(des_freq_arr, freq_arr, kr_arr.real, kr_r_spline)
    kr_i_out, dkr_df_out = vec_splint(des_freq_arr, freq_arr, kr_arr.imag, kr_i_spline)
    kr_out = kr_r_out + 1j*kr_i_out

    """
    Now interpolate the mode amplitudes
    """
    for i in range(num_zs):
        phi_zs_vals = phi_zs_vals_arr[i,:]
        phi_zs_spline = get_spline(freq_arr, phi_zs_vals, 1e30, 1e30)
        phi_zs_out[i,:] = vec_splint(des_freq_arr, freq_arr, phi_zs_vals, phi_zs_spline)[0]

    for i in range(num_zr):
        phi_zr_vals = phi_zr_vals_arr[i,:]
        phi_zr_spline = get_spline(freq_arr, phi_zr_vals, 1e30, 1e30)
        phi_zr_out[i,:] = vec_splint(des_freq_arr, freq_arr, phi_zr_vals, phi_zr_spline)[0]
    return kr_out, phi_zs_out, phi_zr_out

@jit
def single_mode_freq_lin_interp(des_freq_arr, freq_arr, kr_arr, phi_zs_vals_arr, phi_zr_vals_arr):
    """
    Interpolate a single modes values onto a frequency grid
    des_freq_arr - desired frequencies, assumed to be wihtin the bounds of freq_arr
    freq_arr- frequencies at which the modes have been computed
    kr_arr - kr value at each frequency
    phi_zs_vals_arr - phi_zs values at each frequency
    phi_zr_vals_arr - phi_zr values at each frequency
    """
    if des_freq_arr.min() < freq_arr.min() or (des_freq_arr.max() > freq_arr.max()):
        raise ValueError('Attemptimg to extrapolate beyond frequency bounds')

    num_zs = phi_zs_vals_arr.shape[0]
    num_zr = phi_zr_vals_arr.shape[0]
    num_freq_out = des_freq_arr.size
    phi_zs_out = np.zeros((num_zs, num_freq_out))
    phi_zr_out = np.zeros((num_zr, num_freq_out))

    """
    First interpolate the wave number
    """
    kr_r_out = vec_lin_int(des_freq_arr, freq_arr, kr_arr.real)
    kr_i_out = vec_lin_int(des_freq_arr, freq_arr, kr_arr.imag)
    kr_out = kr_r_out + 1j*kr_i_out

    """
    Now interpolate the mode amplitudes
    """
    for i in range(num_zs):
        phi_zs_vals = phi_zs_vals_arr[i,:]
        phi_zs_out[i,:] = vec_lin_int(des_freq_arr, freq_arr, phi_zs_vals)

    for i in range(num_zr):
        phi_zr_vals = phi_zr_vals_arr[i,:]
        phi_zr_out[i,:] = vec_lin_int(des_freq_arr, freq_arr, phi_zr_vals)
    return kr_out, phi_zs_out, phi_zr_out

@njit(cache=True)
def full_freq_interp(des_freq_arr, freq_arr, mean_krs_arr, phi_zs_vals_arr, phi_zr_vals_arr):
    """
    des_freq_arr - np 1d array 
        desired frequencies
    freq_arr - np 1d array
        frequencies at which the modes have been computed
        must be monotonically increasing
    mean_krs_arr - np 2d array
        first axis is krs
        second axis is frequency
        filled with -1 if mode doesn't exist at that frequency
    phi_zs_vals_arr - np 3d array
        first axis is z index
        second axis is phi_zs
        third axis is frequency
    phi_zr_vals_arr - np 3d array
        first axis is z index
        second axis is phi_zr
        third axis is frequency
    """
    M_max = mean_krs_arr.shape[0]
    mean_krs_full_arr = -1*np.ones((M_max, des_freq_arr.size), dtype=np.complex128)
    phi_zs_full_arr = np.zeros((phi_zs_vals_arr.shape[0], M_max, des_freq_arr.size))
    phi_zr_full_arr = np.zeros((phi_zr_vals_arr.shape[0], M_max, des_freq_arr.size))
    for i in range(M_max):
        mode_freqs = freq_arr[mean_krs_arr[i,:].real > 0] # frequencies at which this mode exists
        fmin = mode_freqs.min()
        eval_inds = freq_arr >= fmin
        des_inds = des_freq_arr >= fmin
        kr_vals = mean_krs_arr[i,:][eval_inds]
        phi_zs_vals = phi_zs_vals_arr[:,i, :][:, eval_inds]
        phi_zr_vals = phi_zr_vals_arr[:,i, :][:, eval_inds]
        mode_des_freqs = des_freq_arr[des_inds]
        mode_eval_freqs = freq_arr[eval_inds]
        #print(mode_eval_freqs.shape, 'num eval f')
        if mode_eval_freqs.size == 1:
            continue
        kr_interp, phi_zs_interp, phi_zr_interp = single_mode_freq_spl_interp(mode_des_freqs, mode_eval_freqs, kr_vals, phi_zs_vals, phi_zr_vals)
        mean_krs_full_arr[i,:][des_inds] = kr_interp[:]
        phi_zs_full_arr[:,i,:][:,des_inds] = phi_zs_interp[:,:]
        phi_zr_full_arr[:,i,:][:,des_inds] = phi_zr_interp[:,:]
    return mean_krs_full_arr, phi_zs_full_arr, phi_zr_full_arr

class MultiFrequencyAdiabaticModel:
    def __init__(self, range_list, env_list, world_comm, model_freqs, pulse_freqs, model_comm):
        self.env_list = env_list
        self.range_list = range_list
        self.world_comm = world_comm
        self.world_rank = self.world_comm.Get_rank()
        self.num_freqs = len(model_freqs)
        self.num_ranges = len(range_list)
        self.model_freqs = model_freqs
        self.pulse_freqs = pulse_freqs # array
        self.freq = self.model_freqs[(self.world_rank // self.num_ranges)]
        self.model = AdiabaticModel(range_list, env_list, model_comm)
        self.range_rank = model_comm.Get_rank()
        self.is_primary = self.range_rank == 0
        #print('rank, freq', self.rank, self.freq)

    def run_model(self, zs, zr, rs, x0_list):
        """
        Run the forward model to get modes at each frequency
        in model_freqs
        zs is np array
        zr is np array
        rs is scalar 
        x0_list - list of perturbation coefficient array for
            each env...
        Then, interpolate these to get the 
        - modes at source position
        - modes at receiver positions
        - mean propagation wavenumbers 
        at each frequency in pulse freqs using ?
            spline?
        """

        # step 1: run model at the model frequencies
        now = time.time()
        modes_list = self.model.run_models(self.freq, x0_list) # modes objects in list correspond to environment at given ranges
        if self.is_primary:
            M_list = [x.M for x in modes_list]
            M_max = max(M_list)
            mean_krs = self.model._get_mean_krs(rs, M_max)
            M = mean_krs.size
            phi_zs = self.model.get_phi_zs(zs, M)
            phi_zr = self.model.get_phi_zr(zr, M, np.array(self.range_list), rs)

            if self.world_rank > 0:
                self.world_comm.send(mean_krs, dest=0, tag=0)
                self.world_comm.send(phi_zs, dest=0, tag=1)
                self.world_comm.send(phi_zr, dest=0, tag=2)

        # step 2: use the rank 0 process to collect modal information from all envs
        self.world_comm.Barrier()
        if self.world_rank == 0:
            mean_krs_freq_list = [mean_krs]
            phi_zs_freqs_list = [phi_zs]
            phi_zr_freqs_list = [phi_zr]
            for f_i in range(1, self.num_freqs):
                f_rank = f_i*self.num_ranges
                mean_krs_freq_list.append(self.world_comm.recv(source=f_rank, tag=0))
                phi_zs_freqs_list.append(self.world_comm.recv(source=f_rank, tag=1))
                phi_zr_freqs_list.append(self.world_comm.recv(source=f_rank, tag=2))

        # step 3: interpolate the modal values onto the pulse frequencies
        if self.world_rank == 0:
            now = time.time()
            M_freq_list = [x.size for x in mean_krs_freq_list] #
            #print('M list for freqs in the model', M_freq_list)
            M_max = max(M_freq_list)
            mean_krs_arr = -1*np.ones((M_max, self.num_freqs), dtype=np.complex128)
            phi_zs_arr = np.zeros((zs.size, M_max, self.num_freqs))
            phi_zr_arr = np.zeros((zr.size, M_max, self.num_freqs))

            # pack the modes from each frequency into arrays
            for i in range(self.num_freqs):
                mean_krs_arr[:M_freq_list[i], i] = mean_krs_freq_list[i]
                phi_zs_arr[:, :M_freq_list[i], i] = phi_zs_freqs_list[i]
                phi_zr_arr[:, :M_freq_list[i], i] = phi_zr_freqs_list[i]
   
            now = time.time()
            pulse_freq_mask = (self.pulse_freqs >= min(self.model_freqs)) & (self.pulse_freqs <= max(self.model_freqs)) 
            bounded_pulse_freqs = self.pulse_freqs[pulse_freq_mask]
            mean_krs_full_arr, phi_zs_full_arr, phi_zr_full_arr =  full_freq_interp(bounded_pulse_freqs, self.model_freqs, mean_krs_arr, phi_zs_arr, phi_zr_arr)
            #print('this interp time', time.time() - now)
            

        fields = np.zeros((zs.size, zr.size, len(self.pulse_freqs)), dtype=np.complex128)
        # step 4: use interpolated modal values to get fields at each pulse frequency
        count = 0
        if self.world_rank == 0:
            for pulse_freq_index in range(len(self.pulse_freqs)):
                if self.pulse_freqs[pulse_freq_index] < min(self.model_freqs) or self.pulse_freqs[pulse_freq_index] > max(self.model_freqs):
                    continue
                else:
                    pf = self.pulse_freqs[pulse_freq_index]
                    bf_ind = np.argmin(np.abs(bounded_pulse_freqs - pf))
                    bf_krs = mean_krs_full_arr[:, bf_ind]
                    mode_inds = bf_krs != -1
                    bf_krs = bf_krs[bf_krs != -1]
                    field = np.zeros((zs.size, zr.size), dtype=np.complex128)
                    phi_zs_arr = phi_zs_full_arr[:, :, bf_ind][:,mode_inds]
                    phi_zr_arr = phi_zr_full_arr[:, :, bf_ind][:,mode_inds]
                    M = bf_krs.size
                    if M == 0:
                        pass
                    else:
                        
                        #print('pf - bf', pf, bounded_pulse_freqs[bf_ind], mean_krs -bf_krs)
                        #plt.plot(self.pulse_freqs[pulse_freq_index]*np.ones(M), mean_krs.real, 'ro')
                        p = pressure_calc.get_arr_pressure(phi_zs_arr, phi_zr_arr, bf_krs, rs)
                        field = np.squeeze(p)
                        count += 1
                    fields[...,pulse_freq_index] = field


        return fields
