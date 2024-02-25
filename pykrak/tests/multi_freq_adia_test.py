""" Description:
    Multiple frequency -- pulse synthesis

Date:
    2/23/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pyat.pyat.readwrite import read_env, write_env, write_fieldflp, read_shd, write_bathy
from pykrak import coupled_modes as cm
from pykrak.linearized_model import LinearizedEnv
from pykrak.test_helpers import get_krak_inputs
from pykrak.adia_model import MultiFrequencyAdiabaticModel
from pyat.pyat import env as pyat_env
from mpi4py import MPI
import os
import time

from fft import fft

def get_source_waveform(fc, samp_per_cyc, Q, num_digits):
    """
    Get samples from the electronic driving signal for the source
    Signal is a sinusoid with frequency fc
    Sampling rate is determined by samp_per_cyc
    Q is the number of cycles per digit
    num_digits is the total number of digits to use
    The digits transmitted are 0, 1, 0, 0, 0, ...
    Return
    tgrid, signal, fgrid, S
    S is the discrete Fourier transform of the signal
    """
    N_dig = Q*samp_per_cyc
    N_tot = num_digits*N_dig
    fs = samp_per_cyc*fc 
    dt = 1/(fs)
    tgrid = np.linspace(0, (N_tot-1)*dt, N_tot)

    t_digit = np.linspace(0, (N_dig-1)*dt, N_dig)
    digit_waveform = np.sin(2*np.pi*fc*t_digit)

    digits = np.zeros(num_digits)
    digits[0] = 1

    digits = np.reshape(digits, (num_digits, 1))
    digit_waveform = np.reshape(digit_waveform, (1, N_dig))
    signal = digits * digit_waveform
    signal = signal.reshape(N_tot)
    normalized_f_grid, S = fft.fft1(signal, -1) # -1 is the forward transform
    fgrid = normalized_f_grid*fs
    return tgrid, signal, fgrid, S

def get_transducer_H(fgrid, fr, Qr, A, M=1):
    """
    """
    omega = 2*np.pi*fgrid
    omegar = 2*np.pi*fr
    nonzero_inds = np.where(omega != 0)

    K = omegar**2  * M
    R = omegar*M * Qr # 
    Y = 1j*omega[nonzero_inds]/K + 1/R + 1/(1j*omega[nonzero_inds] * M)
    H = np.zeros(omega.size, dtype=np.complex128)
    H[nonzero_inds] = 1j*omega[nonzero_inds]/Y
    H /= np.max(np.abs(H))
    H *= A
    return H

def get_pulse_weights():
    fc= 35
    samp_per_cyc = 8
    Q = 8
    T = 80
    num_digits = int(T / (Q/fc))
    tgrid, signal, fgrid, S = get_source_waveform(fc, samp_per_cyc, Q, num_digits)
    fr = 35
    Qr = 8
    A = 1
    H = get_transducer_H(fgrid, fr, Qr, A)
    S = H*S
    return tgrid, signal, fgrid, S

def downslope_test(num_freqs):
    tgrid, signal, fgrid, S = get_pulse_weights()

    _, trans_sig = fft.fft1(S, +1)  # inverse FFT
    trans_sig = trans_sig[:signal.size] / trans_sig.size # remove zero padding and transform scaling


    fmin, fmax = 30,40
    model_freqs = np.linspace(fmin, fmax, num_freqs)
    pulse_inds = (fgrid >= fmin) & (fgrid <= fmax)
    pulse_freqs = fgrid[pulse_inds]
    sym_inds = (fgrid <= -fmin) & (fgrid >= -fmax)

    # load up communicator and split for each model freq.
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    size = world_comm.Get_size()
    num_envs = int(size / num_freqs)

    color = int(world_rank // num_envs) # this gives me a model comm for each frequcny
    model_comm = world_comm.Split(color, world_rank)

    freq_i = world_rank % num_freqs
    freq = model_freqs[freq_i]

    if world_rank == 0:
        fig, axes = plt.subplots(1,1, sharex=True)
        axes.plot(tgrid, signal)


        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(tgrid, np.abs(trans_sig))
        axes[1].plot(tgrid, np.angle(trans_sig))


        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(fgrid, np.abs(S))
        axes[1].plot(fgrid, np.angle(S))
    
    # form list of environment segments  for each frequency
    Z0 = 100.0
    Z1 = 200.0
    R = 100*1e3
    Zvals = np.linspace(Z0, Z1, num_envs)
    Zmax = Zvals.max()
    rgrid = np.linspace(0.0, R, num_envs)
    rcm_grid = cm.get_seg_interface_grid(rgrid)
    cw = 1500.0
    rho_w = 1.0
    c_hs = 1800.0
    rho_hs = 2.0
    attn_hs = 0.01
    attn_units = 'dbpkmhz'
    range_list = [x for x in rgrid]


    full_env_list = []
    for freq in model_freqs:
        mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

        cmin = 1500.0
        cmax = 1799.0

        # Pekeris waveguide at each segment
        krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
        nmesh_list = []
        env_list = []
        x0_list = []
        # Make env_list 
        Ntot = int(Zmax / mesh_dz)
        for seg_i in range(num_envs):
            Z = Zvals[seg_i]
            env_z_list = [np.array([0.0, Z]), np.array([Z, Zmax])]
            env_c_list = [np.array([cw, cw]), np.array([c_hs, c_hs])]
            env_rho_list = [np.array([rho_w, rho_w]), np.array([rho_hs, rho_hs])]
            env_attn_list = [np.array([0.0, 0.0]), np.array([attn_hs, attn_hs])]
            Nwtr = int(Z / mesh_dz)
            Nlyr = Ntot - Nwtr + 1
            N_list = [Nwtr, Nlyr]

            if Z == Zmax: # don't need the layer domain extension
                env_z_list = [env_z_list[0]]
                env_c_list = [env_c_list[0]]
                env_rho_list = [env_rho_list[0]]
                env_attn_list = [env_attn_list[0]]
                N_list = [Ntot]
            print('N_list', N_list)
            print("Ntot", sum(N_list))


            nmesh_list.append(N_list)


            env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
            
            env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
            env_list.append(env)
            x0_list.append(np.array([0.0]))

        full_env_list.append(env_list)


    # now set the model frequencies and the pulse frequencies
    rdm = MultiFrequencyAdiabaticModel(range_list, full_env_list, world_comm, model_freqs, pulse_freqs, model_comm)

    zs = np.array([25.])    
    same_grid = False
    rs = 100*1e3
    zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
    zr = zout[1:]

    #


    wg_weights = rdm.run_model(zs, zr, rs, x0_list)
    if world_rank == 0:
        wg_weights = np.squeeze(wg_weights)
        print('weights shape', wg_weights.shape)
        Hwg= np.zeros((zr.size, fgrid.size), dtype=np.complex128)
        Hwg[:, pulse_inds] = wg_weights
        Hwg[:, sym_inds] = wg_weights[::-1].conj()
        print(fgrid[pulse_inds])
        print('sym inds..ordering', fgrid[sym_inds][::-1])
        Swg = S[None,:]*Hwg

        rcv_arr = np.zeros((zr.size, tgrid.size), dtype=np.complex128)
        for i in range(zr.size):
            _, signal_rcv = fft.fft1(Swg[i,:], +1)  # inverse FFT
            signal_rcv = signal_rcv[:signal.size] / Swg.shape[1] # remove zero padding and transform scaling
            rcv_arr[i,:] = signal_rcv

        plt.figure()
        plt.pcolormesh(tgrid, zr, np.abs(rcv_arr))
        plt.colorbar()

        z_ind = np.argmin(np.abs(25 - zr))
        plt.figure()
        plt.plot(pulse_freqs, np.abs(wg_weights[z_ind,:]))

        fig, axes = plt.subplots(2,1, sharex=True)
        plt.suptitle('spectrum at 25 m depth')
        axes[0].plot(fgrid, np.abs(Swg[z_ind,:]))
        axes[1].plot(fgrid, np.angle(Swg[z_ind,:]))
        fig, axes = plt.subplots(2,1, sharex=True)
        plt.suptitle('spectrum at 25 m depth')
        axes[0].plot(fgrid, np.abs(S))
        axes[1].plot(fgrid, np.angle(S))
        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(tgrid, np.abs(rcv_arr[z_ind, :]))
        axes[1].plot(tgrid, np.angle(rcv_arr[z_ind, :]))

        

        plt.show()

def ri_test(num_freqs):
    tgrid, signal, fgrid, S = get_pulse_weights()

    _, trans_sig = fft.fft1(S, +1)  # inverse FFT
    trans_sig = trans_sig[:signal.size] / trans_sig.size # remove zero padding and transform scaling


    fmin, fmax = 30,40
    model_freqs = np.linspace(fmin, fmax, num_freqs)
    pulse_inds = (fgrid >= fmin) & (fgrid <= fmax)
    pulse_freqs = fgrid[pulse_inds]
    sym_inds = (fgrid <= -fmin) & (fgrid >= -fmax)

    # load up communicator and split for each model freq.
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    size = world_comm.Get_size()
    num_envs = 1

    color = int(world_rank // num_envs) # this gives me a model comm for each frequcny
    print('color', color)
    model_comm = world_comm.Split(color, world_rank)

    freq_i = world_rank % num_freqs
    freq = model_freqs[freq_i]

    if world_rank == 0:
        fig, axes = plt.subplots(1,1, sharex=True)
        axes.plot(tgrid, signal)


        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(tgrid, np.abs(trans_sig))
        axes[1].plot(tgrid, np.angle(trans_sig))


        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(fgrid, np.abs(S))
        axes[1].plot(fgrid, np.angle(S))
    
    # form list of environment segments  for each frequency
    Z = 200.0
    R = 100*1e3
    rgrid=np.array([0.0])
    cw = 1500.0
    rho_w = 1.0
    c_hs = 1800.0
    rho_hs = 2.0
    attn_hs = 0.01
    attn_units = 'dbpkmhz'
    range_list = [x for x in rgrid]
    Zvals = np.array([Z])
    Zmax = Z
    full_env_list = []
    for freq in model_freqs:
        mesh_dz = (1500 / freq) / 20 # lambda /20 spacing

        cmin = 1500.0
        cmax = 1799.0

        # Pekeris waveguide at each segment
        krs_list, phi_list, zgrid_list, rho_list, rho_hs_list, c_hs_list = [], [], [], [], [], []
        nmesh_list = []
        env_list = []
        x0_list = []
        # Make env_list 
        Ntot = int(Zmax / mesh_dz)
        for seg_i in range(num_envs):
            Z = Zvals[seg_i]
            env_z_list = [np.array([0.0, Z]), np.array([Z, Zmax])]
            env_c_list = [np.array([cw, cw]), np.array([c_hs, c_hs])]
            env_rho_list = [np.array([rho_w, rho_w]), np.array([rho_hs, rho_hs])]
            env_attn_list = [np.array([0.0, 0.0]), np.array([attn_hs, attn_hs])]
            Nwtr = int(Z / mesh_dz)
            Nlyr = Ntot - Nwtr + 1
            N_list = [Nwtr, Nlyr]

            if Z == Zmax: # don't need the layer domain extension
                env_z_list = [env_z_list[0]]
                env_c_list = [env_c_list[0]]
                env_rho_list = [env_rho_list[0]]
                env_attn_list = [env_attn_list[0]]
                N_list = [Ntot]


            nmesh_list.append(N_list)


            env = LinearizedEnv(freq, env_z_list, env_c_list, env_rho_list, env_attn_list, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
            
            env.add_c_pert_matrix(env.z_arr, np.zeros((env.z_arr.size,1)))
            env_list.append(env)
            x0_list.append(np.array([0.0]))

        full_env_list.append(env_list)


    print('env list', full_env_list)
    # now set the model frequencies and the pulse frequencies
    rdm = MultiFrequencyAdiabaticModel(range_list, full_env_list, world_comm, model_freqs, pulse_freqs, model_comm)

    zs = np.array([25.])    
    same_grid = False
    rs = 100*1e3
    zout = np.linspace(0.0, Zvals.max(), nmesh_list[-1][0])
    zr = zout[1:]

    #


    wg_weights = rdm.run_model(zs, zr, rs, x0_list)
    print('got weights')
    if world_rank == 0:
        wg_weights = np.squeeze(wg_weights)
        print('weights shape', wg_weights.shape)
        Hwg= np.zeros((zr.size, fgrid.size), dtype=np.complex128)
        Hwg[:, pulse_inds] = wg_weights
        Hwg[:, sym_inds] = wg_weights[::-1].conj()
        print(fgrid[pulse_inds])
        print('sym inds..ordering', fgrid[sym_inds][::-1])
        Swg = S[None,:]*Hwg

        rcv_arr = np.zeros((zr.size, tgrid.size), dtype=np.complex128)
        for i in range(zr.size):
            _, signal_rcv = fft.fft1(Swg[i,:], +1)  # inverse FFT
            signal_rcv = signal_rcv[:signal.size] / Swg.shape[1] # remove zero padding and transform scaling
            rcv_arr[i,:] = signal_rcv

        plt.figure()
        plt.pcolormesh(tgrid, zr, np.abs(rcv_arr))
        plt.colorbar()

        z_ind = np.argmin(np.abs(25 - zr))
        plt.figure()
        plt.plot(pulse_freqs, np.abs(wg_weights[z_ind,:]))

        fig, axes = plt.subplots(2,1, sharex=True)
        plt.suptitle('spectrum at 25 m depth')
        axes[0].plot(fgrid, np.abs(Swg[z_ind,:]))
        axes[1].plot(fgrid, np.angle(Swg[z_ind,:]))
        fig, axes = plt.subplots(2,1, sharex=True)
        plt.suptitle('spectrum at 25 m depth')
        axes[0].plot(fgrid, np.abs(S))
        axes[1].plot(fgrid, np.angle(S))
        fig, axes = plt.subplots(2,1, sharex=True)
        axes[0].plot(tgrid, np.abs(rcv_arr[z_ind, :]))
        axes[1].plot(tgrid, np.angle(rcv_arr[z_ind, :]))

        

        plt.show()

if __name__ == '__main__':
    ri_test(12)
    downslope_test(6)
