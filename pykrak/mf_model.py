"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from pykrak import pykrak_env, pressure_calc
from pykrak import linearized_model as lm
from pykrak import range_dep_model as rdm
from pykrak.adia_model import AdiabaticModel
from pykrak.cm_model import CMModel
import time
from numba import njit, jit
from interp.interp import get_spline, splint, vec_splint, vec_lin_int

@njit(cache=True)
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

@njit(cache=True)
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
        if mode_eval_freqs.size == 1:
            continue
        kr_interp, phi_zs_interp, phi_zr_interp = single_mode_freq_spl_interp(mode_des_freqs, mode_eval_freqs, kr_vals, phi_zs_vals, phi_zr_vals)
        mean_krs_full_arr[i,:][des_inds] = kr_interp[:]
        phi_zs_full_arr[:,i,:][:,des_inds] = phi_zs_interp[:,:]
        phi_zr_full_arr[:,i,:][:,des_inds] = phi_zr_interp[:,:]
    return mean_krs_full_arr, phi_zs_full_arr, phi_zr_full_arr

def get_field(zs, zr, rs, x0_list, env, model_freqs, pulse_freqs, bounded_pulse_freqs, mean_krs_full_arr, phi_zs_full_arr, phi_zr_full_arr):
    count = 0
    fields = np.zeros((zs.size, zr.size, len(pulse_freqs)), dtype=np.complex128)
    for pulse_freq_index in range(len(pulse_freqs)):
        if pulse_freqs[pulse_freq_index] < min(model_freqs) or pulse_freqs[pulse_freq_index] > max(model_freqs):
            continue
        else:
            pf = pulse_freqs[pulse_freq_index]
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
                p = pressure_calc.get_arr_pressure(phi_zs_arr, phi_zr_arr, bf_krs, np.array([rs]))
                field = np.squeeze(p)
                count += 1
            fields[...,pulse_freq_index] = field

    return fields

def compute_mf_arr_cm_pressure(omega, krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list, rgrid, zs, zr, rs_grid, same_grid, cont_part_velocity=True):
    """
    Compute the pressure field at the receiver depths in zr
    Due to the source at zs 
    receivers at all ranges in rs_grid
    rgrid is the grid of environmental segment interfaes (see get_seg_interface_grid)
    rgrid[0] must be 0
    same_grid - boolean Flag 
        True if all the modes have been evaluated on same grid (and so interpolation is not necessary)
    """
    Nr = zr.size
    krs0 = krs_list[0]
    phi0 = phi_list[0]
    zgrid0 = zgrid_list[0]
    rho0 = rho_list[0]
    rho_hs0 = rho_hs_list[0]
    c_hs0 = c_hs_list[0]


    pressure_out = np.zeros((Nr, rs_grid.size), dtype=np.complex128)

    gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
    gammas0 = np.ascontiguousarray(gammas0)
   
    phi_zs = np.zeros((krs0.size))
    for i in range(krs0.size):
        phi_zs[i] = interp.lin_int(zs, zgrid0, phi0[:,i])
    # the initial value is a bit 
    a0 = phi_zs * np.exp(-1j * krs0 * rgrid[1]) / np.sqrt(krs0) 
    a0 *= 1j*np.exp(1j*np.pi/4) # assumes rho is 1 at source depth
    a0 /= np.sqrt(8*np.pi)

    rcvr_range_index = 0
    for i in range(1, rgrid.size):
        r0 = rgrid[i-1]
        r1 = rgrid[i]

        phi0 = phi_list[i-1]
        zgrid0 = zgrid_list[i-1]
        rho0 = rho_list[i-1]
        rho_hs0 = rho_hs_list[i-1]
        c_hs0 = c_hs_list[i-1]
        krs0 = krs_list[i-1]
        gammas0 = np.sqrt(krs0**2 - (omega**2 / c_hs0**2)).real
        gammas0 = np.ascontiguousarray(gammas0)

        while (rcvr_range_index < rs_grid.size) and rs_grid[rcvr_range_index] <= r1:
            rs = rs_grid[rcvr_range_index]
            if i == 1:
                p0 = advance_a(a0, krs0, r1, rs)
            else:
                p0 = advance_a(a0, krs0, r0, rs)
            p = np.sum(p0 * phi0, axis=1)
            p /= np.sqrt(rs)
            p_zr_real = interp.vec_lin_int(zr, zgrid0, p.real)
            p_zr_imag = interp.vec_lin_int(zr, zgrid0, p.imag)
            pressure_out[:, rcvr_range_index] = p_zr_real + 1j*p_zr_imag
            rcvr_range_index += 1
        if rcvr_range_index == rs_grid.size:
            break
        # advance a0 to the next segment
        phi1 = phi_list[i]
        zgrid1 = zgrid_list[i]
        rho1 = rho_list[i]
        rho_hs1 = rho_hs_list[i]
        c_hs1 = c_hs_list[i]
        krs1 = krs_list[i]
        gammas1 = np.sqrt(krs1**2 - (omega**2 / c_hs1**2)).real
        gammas1 = np.ascontiguousarray(gammas1)
        if i == 1: # first segment is special case (phase of a0 is for source-receiver range equal to the first segment length r1)
            a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r1, r1, same_grid, cont_part_velocity)
        else:
            a0 = update_a0(a0, krs0, gammas0, phi0, zgrid0, rho0, rho_hs0, krs1, gammas1, phi1, zgrid1, rho1, rho_hs1, r0, r1, same_grid, cont_part_velocity)

    # now finish for receivers in final segment
    while rcvr_range_index < rs_grid.size:
        rs = rs_grid[rcvr_range_index]
        p0 = advance_a(a0, krs1, r1, rs) 
        p = np.sum(p0 * phi1, axis=1)
        p /= np.sqrt(rs)
        p_zr_real = interp.vec_lin_int(zr, zgrid1, p.real) # zgrid should be very fine so I don't see any issue here...
        p_zr_imag = interp.vec_lin_int(zr, zgrid1, p.imag)
        p_zr = p_zr_real + 1j*p_zr_imag
        pressure_out[:, rcvr_range_index] = p_zr
        rcvr_range_index += 1
    return pressure_out

class MultiFrequencyModel:
    def __init__(self, range_list, env, world_comm, model_freqs, pulse_freqs, model_comm,mod_type):
        self.env = env # one list for each frequency 
        self.range_list = range_list
        self.world_comm = world_comm
        self.world_rank = self.world_comm.Get_rank()
        self.num_freqs = len(model_freqs)
        self.num_ranges = len(range_list)
        self.model_freqs = model_freqs
        self.pulse_freqs = pulse_freqs # array
        self.freq_ind = self.world_rank // self.num_ranges
        self.freq = self.model_freqs[self.freq_ind]
        self.mod_type = mod_type
        if mod_type == 'adia':
            self.model = AdiabaticModel(range_list, env, model_comm)
        elif mod_type == 'cm':
            self.model = CMModel(range_list, env, model_comm)
        else:
            raise ValueError("Model type not recognized : must be adia or cm")
        self.range_rank = model_comm.Get_rank()
        self.is_primary = self.range_rank == 0

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

        # step 1: run model at the model frequencies to get all modes
        modes_list = self.model.run_models(x0_list) # modes objects in list correspond to environment at given ranges
        if self.mod_type == 'adia':
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
            #self.world_comm.Barrier()
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
                

            fields = np.zeros((zs.size, zr.size, len(self.pulse_freqs)), dtype=np.complex128)
            # step 4: use interpolated modal values to get fields at each pulse frequency
            count = 0
            if self.world_rank == 0:
                fields = get_field(zs, zr, rs, x0_list, self.env, self.model_freqs, self.pulse_freqs, bounded_pulse_freqs, mean_krs_full_arr, phi_zs_full_arr, phi_zr_full_arr)
        elif self.mod_type == 'cm':
            range_list = self.range_list
            #fields = np.zeros((zs.size, zr.size, len(self.pulse_freqs)), dtype=np.complex128)
            fields = np.zeros((zs.size, zr.size, len(self.model_freqs)), dtype=np.complex128)
            if self.is_primary: # this is for each frequency
                model = self.model
                p = model.compute_field(zs, zr, rs, same_grid=True, cont_part_velocity=True)
                #model_fields[:,:,self.freq_ind] = p
                if self.world_rank > 0:
                    self.world_comm.send(p, dest=0, tag=0)
            if self.world_rank == 0: # now do the f interpolation...
                p_list = [p]
                model_fields[:,:,self.freq_ind] = p.copy()
                for f_i in range(1, self.num_freqs):
                    f_rank = f_i*self.num_ranges
                    pf = self.world_comm.recv(source=f_rank, tag=0)
                    p_list.append(pf)
                    fields[:,:,f_i] = pf.copy()

        else:
            raise ValueError("Model type not recognized : must be adia or cm")
        return fields
