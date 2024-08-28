"""
Description:
    Inherit env model
    add linearization

Date:
    10/6/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak.pykrak_env import Env, Modes
from pykrak import adjoint
from pykrak.sturm_seq import cat_list_to_arr
from pykrak.raw_pykrak import get_modes
from numba import njit
from pykrak import pressure_calc as pc
from pykrak.attn_pert import get_c_imag, get_attn_conv_factor, get_c_imag_npm
from interp import interp
from pykrak.misc import get_simpsons_integrator, get_layer_N

@njit
def get_layered_dkdc_integrator(h_arr, ind_arr, z_arr, k_sq_arr, c_arr, rho_arr):
    """
    Input
    h_arr - array of mesh spacings for each layer
    ind_arr - array of index of first value for each layer
    z_arr - depths of the layer meshes concatenated
    k_sq_arr - wavenumber square omega^2 / c^2 for the layer meshes concatenated
    rho_arr - density of the layer meshes concatenated

    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for i in range(len(layer_N)):
        if i < num_layers -1 :
            k_sq_i = k_sq_arr[ind_arr[i]:ind_arr[i+1]]
            rho_i = rho_arr[ind_arr[i]:ind_arr[i+1]]
            c_i = c_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            k_sq_i = k_sq_arr[ind_arr[i]:]
            rho_i = rho_arr[ind_arr[i]:]
            c_i = c_arr[ind_arr[i]:]
        integrator_i = get_simpsons_integrator(layer_N[i], h_arr[i])[0,:]
        integrator_i *= k_sq_i / (rho_i * c_i)
        if i == 0:
            integrator = integrator_i
        else:
            integrator[-1] += integrator_i[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_i[1:]))
    return -integrator

class LinEnvError(Exception):
    pass

def get_ml_noise_lh(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Use columns of model matrix to get new sound speeds
    d is data vector
    """
    @njit
    def lh(r, zs, zr, tilt, x):
        """
        """
        delta_c = model_matrix@x
        krs, phi, phi_z, _ = get_modes(freq, h_arr, ind_arr, z_arr, c_arr+delta_c, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs = get_phi_zr(np.array([zs]), phi_z, phi)
        deltaR = get_delta_R(tilt, zr)
        p =get_pressure(phi_zr, phi_zs, krs, r,deltaR)
        alpha_sq = get_alpha_sq(d[:,0], p[:,0])
        log_lh = -.5*np.log(1- alpha_sq)
        return log_lh
    return lh

class LinearizedEnv(Env):
    def __init__(self, freq, z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units, N_list, cmin, cmax):
        super().__init__(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units)
        self.add_freq(freq)
        self._fix_mesh(N_list)
        self._get_env_arrs()
        self.cmin = cmin
        self.cmax = cmax
        self.zs_arr = None
        self.zr_arr = None
        self.modes = None
        self.phi_zs = None
        self.phi_zr = None
        self.x0 = None
        self.add_attn_conv_factor()

    def _fix_mesh(self, N_list):
        """
        Fix the mesh of the model
        """
        self.N_list = N_list
        z_list, c_list, rho_list, attn_list, kr_sq_list = self.interp_env_vals(N_list)
        self.z_list = z_list
        self.c_list = c_list
        self.rho_list = rho_list
        self.attn_list = attn_list
        self.kr_sq_list = kr_sq_list # complex 
        self.N_list = N_list
        self.h_list = [x[1] - x[0] for x in self.z_list]
        return
    
    def _get_env_arrs(self):
        """
        Get the array-ified lists
        """
        self.h_arr = np.array(self.h_list)
        self.z_arr, self.ind_arr = cat_list_to_arr(self.z_list)
        self.c_arr, _  = cat_list_to_arr(self.c_list)
        self.rho_arr, _ = cat_list_to_arr(self.rho_list)
        self.attn_arr, _ = cat_list_to_arr(self.attn_list)
        self.kr_sq_arr, _ = cat_list_to_arr(self.kr_sq_list)
        return

    def add_c_pert_matrix(self, pert_z_arr, pert_c_arr):
        """
        Add a model matrix to the environment
        pert_z_arr - grid of depths at which the perturbation basis vectors
            are defined
        pert_c_arr - matrix of perturbation basis vectors
            each column is a perturbation basis vector
        Interpolate these onto the mesh using linear interpolation
        """
        L = pert_c_arr.shape[1]
        
        self.P = L
        interp_pert_c_arr = np.zeros((self.z_arr.size, L))
        for i in range(L):
            interp_pert_c_arr[:, i] = np.interp(self.z_arr, pert_z_arr, pert_c_arr[:, i])
        self.pert_c_arr = interp_pert_c_arr
        return
       
    def add_x0(self, x0):
        self.x0 = x0
        return

    def update_x0(self, x0):
        """
        Update the x0 point around which linearization
        will be done
        Update the phi_zs and phi_Zr variables as well 
        """
        self.add_x0(x0)
        self.full_forward_modes()
        if self.zs_arr is not None:
            self.add_zs_arr(self.zs_arr)
        if self.zr_arr is not None:
            self.add_zr_arr(self.zr_arr)
        return

    def full_forward_modes(self):
        """
        Run the model on the environment with the parameters perturbed by vector x0
        """
        x0 = self.x0
        freq = self.freq
        ind_arr, z_arr, c_arr, rho_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr
        attn_arr = self.attn_arr
        attn_arr_npm = self.conv_factor * attn_arr
        tmp_c_arr = c_arr + self.pert_c_arr @ x0
        omega = 2*np.pi*self.freq
        #tmp_c_imag = get_c_imag(tmp_c_arr, attn_arr, self.attn_units, omega)
        tmp_c_imag = get_c_imag_npm(tmp_c_arr, attn_arr_npm, omega)
        tmp_c_arr = tmp_c_arr + 1j * tmp_c_imag
        k_sq_arr = omega**2 / tmp_c_arr**2

        attn_arr = self.attn_arr
        rho_hs, c_hs, attn_hs = self.rho_hs, self.c_hs, self.attn_hs
        attn_hs_npm = attn_hs*self.conv_factor
        c_hs_imag = get_c_imag_npm(c_hs, attn_hs_npm, omega)
        tmp_c_hs  = c_hs + 1j * c_hs_imag
        k_hs_sq = omega**2 / tmp_c_hs**2


        cmin, cmax = self.cmin, self.cmax
        h_arr = self.h_arr

        krs, phi, phi_z, ugs = get_modes(freq, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, cmin, cmax)
        M = krs.size
        self.modes = Modes(self.freq, krs, phi, M, phi_z, ugs)
        return self.modes

    def add_zs_arr(self, zs_arr):
        self.zs_arr = zs_arr
        modes = self.modes
        if modes is None:
            raise LinEnvError("Need to run full_forward_modes before adding zs arr")
        self.phi_zs = modes.get_phi_zr(zs_arr)
        return

    def add_zr_arr(self, zr_arr):
        self.zr_arr = zr_arr
        modes = self.modes
        if modes is None:
            raise LinEnvError("Need to run full_forward_modes before adding zs arr")
        self.phi_zr = modes.get_phi_zr(zr_arr)
        return

    def add_r_arr(self, r_arr):
        self.r_arr = r_arr

    def get_full_forward_p(self, deltaR=np.array([0.0])):
        """
        Given array of source positions, array of receiver positions
        and array of ranges, get the pressure field at all of them
        """
        zs_arr, zr_arr, r_arr = self.zs_arr, self.zr_arr, self.r_arr  
        modes = self.modes
        if modes is None:
            raise LinEnvError("Need to run full_forward_modes before getting pressure")
        krs = modes.krs
        phi_zs, phi_zr = self.phi_zs, self.phi_zr
        p_arr = pc.get_arr_pressure(phi_zs, phi_zr, krs, r_arr, deltaR) 
        self.p0_arr = p_arr
        return p_arr

    def linearize(self):
        """
        Get the zfg matrices necessary for linearization of pressure field
        """
        omega = 2*np.pi*self.freq
        ind_arr, z_arr, c_arr, rho_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr
        h_arr = self.h_arr
        attn_arr = self.attn_arr
        rho_hs, c_hs, attn_hs = self.rho_hs, self.c_hs, self.attn_hs
        cmin, cmax = self.cmin, self.cmax
        modes = self.modes
        M = modes.M
        krs, phi = modes.krs, modes.phi
        zfg = np.zeros((self.P, M, M), dtype=np.complex128)
        for k in range(self.P):
            ck_arr = self.pert_c_arr[:, k]
            zfg_k = adjoint.get_zfg_eta_ai(h_arr, ind_arr, ck_arr,z_arr, c_arr, rho_arr, krs, phi, omega)
            zfg[k, :, :] = zfg_k
        self.zfg = zfg
        return

    def linearize_kr(self):
        """
        Get the kr matrices necessary for linearization of pressure field
        """
        omega = 2*np.pi*self.freq
        ind_arr, z_arr, c_arr, rho_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr
        h_arr = self.h_arr
        attn_arr = self.attn_arr
        rho_hs, c_hs, attn_hs = self.rho_hs, self.c_hs, self.attn_hs
        cmin, cmax = self.cmin, self.cmax
        modes = self.modes
        M = modes.M
        krs, phi = modes.krs, modes.phi
        dkrda = np.zeros((M, self.P), dtype=np.complex128)
        k_sq_arr = omega**2 / c_arr**2
        integrator = get_layered_dkdc_integrator(h_arr, ind_arr, z_arr, k_sq_arr, c_arr, rho_arr)
        print(integrator.shape)
        print(phi.shape)
        integrator = integrator[:, np.newaxis]
        for k in range(self.P):
            ck_arr = self.pert_c_arr[:, k][:, np.newaxis]
            integrand = ck_arr * np.abs(phi)**2 / krs
            dkrda_k = np.sum(integrator * integrand, axis=0) #integrate over depth
            dkrda[:,k] = dkrda_k
        self.dkrda = dkrda
        return self.dkrda

    def linearize_forward_p(self):
        x0 = self.x0
        zfg_arr = self.zfg
        phi_zs, phi_zr = self.phi_zs, self.phi_zr
        r_arr = self.r_arr
        krs = self.modes.krs
        self.dpda = adjoint.get_dpda_arr(zfg_arr, krs, r_arr, phi_zr, phi_zs)
        return

    def get_linear_forward_p(self, x):
        p0 = self.p0_arr
        dpda = self.dpda
        x0 = self.x0
        p = p0 + dpda @ (x - x0)
        return p

    def get_numbaized_forward_p(self):
        #x0 = self.x0
        freq = self.freq
        ind_arr, z_arr, c_arr, rho_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr
        attn_arr = self.attn_arr
        attn_arr_npm = self.conv_factor * attn_arr
        omega = 2*np.pi*self.freq
        pert_c_arr = self.pert_c_arr
        attn_units = self.attn_units
        rho_hs, c_hs, attn_hs = self.rho_hs, self.c_hs, self.attn_hs

        attn_hs_npm = attn_hs*self.conv_factor # this is attenuation in nepers/meter
        c_hs_imag = get_c_imag_npm(c_hs, attn_hs_npm, omega)

        tmp_c_hs  = c_hs + 1j * c_hs_imag
        k_hs_sq = omega**2 / tmp_c_hs**2
        cmin, cmax = self.cmin, self.cmax
        h_arr = self.h_arr


        @njit
        def get_p(r_arr, zs_arr, zr_arr, tilt, x0):
            tmp_c_arr = c_arr + (pert_c_arr @ x0)[:,0]
            tmp_c_imag = get_c_imag_npm(tmp_c_arr, attn_arr_npm, omega)
            tmp_c_arr = tmp_c_arr + 1j * tmp_c_imag
            k_sq_arr = omega**2 / tmp_c_arr**2
            krs, phi, phi_z, ugs = get_modes(freq, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, cmin, cmax)
            M = krs.size
            phi_zs = np.zeros((zs_arr.size, M))
            phi_zr = np.zeros((zr_arr.size, M))
            deltaR = pc.get_delta_R(tilt, zr_arr)
            for i in range(M):
                phi_zs[:,i] = interp.vec_lin_int(zs_arr, phi_z, phi[:,i])
                phi_zr[:,i] = interp.vec_lin_int(zr_arr, phi_z, phi[:,i])
            p_arr = pc.get_arr_pressure(phi_zs, phi_zr, krs, r_arr, deltaR) 
            return p_arr
        return get_p

    def get_full_forward_p_func(self):
        #x0 = self.x0
        freq = self.freq
        ind_arr, z_arr, c_arr, rho_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr
        attn_arr = self.attn_arr
        attn_arr_npm = self.conv_factor * attn_arr
        omega = 2*np.pi*self.freq
        pert_c_arr = self.pert_c_arr
        attn_units = self.attn_units
        rho_hs, c_hs, attn_hs = self.rho_hs, self.c_hs, self.attn_hs

        attn_hs_npm = attn_hs*self.conv_factor # this is attenuation in nepers/meter
        c_hs_imag = get_c_imag_npm(c_hs, attn_hs_npm, omega)

        tmp_c_hs  = c_hs + 1j * c_hs_imag
        k_hs_sq = omega**2 / tmp_c_hs**2
        cmin, cmax = self.cmin, self.cmax
        h_arr = self.h_arr


        def get_p(r_arr, zs_arr, zr_arr, tilt, x0):
            dc = (pert_c_arr @ x0)
            print('dc hsape', dc.shape)
            tmp_c_arr = c_arr + dc
            tmp_c_imag = get_c_imag_npm(tmp_c_arr, attn_arr_npm, omega)
            tmp_c_arr = tmp_c_arr + 1j * tmp_c_imag
            k_sq_arr = omega**2 / tmp_c_arr**2
            krs, phi, phi_z, ugs = get_modes(freq, h_arr, ind_arr, z_arr, k_sq_arr, rho_arr, k_hs_sq, rho_hs, cmin, cmax)
            M = krs.size
            phi_zs = np.zeros((zs_arr.size, M))
            phi_zr = np.zeros((zr_arr.size, M))
            deltaR = pc.get_delta_R(tilt, zr_arr)
            for i in range(M):
                phi_zs[:,i] = interp.vec_lin_int(zs_arr, phi_z, phi[:,i])
                phi_zr[:,i] = interp.vec_lin_int(zr_arr, phi_z, phi[:,i])
            p_arr = pc.get_arr_pressure(phi_zs, phi_zr, krs, r_arr, deltaR) 
            return p_arr
        return get_p

def get_ml_noise_lh(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Use columns of model matrix to get new sound speeds
    d is data vector
    """
    @njit
    def lh(r, zs, zr, tilt, x):
        """
        """
        delta_c = model_matrix@x
        krs, phi, phi_z, ugs = get_modes(freq, h_arr, ind_arr, z_arr, c_arr+delta_c, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs = get_phi_zr(np.array([zs]), phi_z, phi)
        deltaR = get_delta_R(tilt, zr)
        p =get_pressure(phi_zr, phi_zs, krs, r,deltaR)
        alpha_sq = get_alpha_sq(d[:,0], p[:,0])
        log_lh = -.5*np.log(1- alpha_sq)
        return log_lh
    return lh

def test1():
    import time
    """ Arctic profile """
    from matplotlib import pyplot as plt
    Z_bott = 2000.
    R = 100*1e3 # 100 km range for updated profile

    plot=True

    env_list = []

    c_hs = 2500.
    rho_hs = 4.0
    attn_hs = 0.2
    attn_units ='dbpkmhz'
    c_layer1 = 2000.0
    rho_layer1 = 2.8
    attn_layer1 = 0.0
    h = 60. 
    z_list1 = [np.array([0, Z_bott]), np.array([Z_bott, Z_bott + h])]
    c_list1 = [np.array([1500., 1500]), np.array([c_layer1, c_layer1])]
    rho_list1 = [np.ones(2), np.array([rho_layer1, rho_layer1])]
    attn_list1 = [.000*np.ones(2), np.array([attn_layer1, attn_layer1])]
    freq = 35.0
    lam = 1500.0 / freq
    dz = lam / 40
    N_list = [int((x[-1] - x[0]) / dz) for x in z_list1]
    cmin = c_list1[0].min()
    cmax = c_list1[1].max()
    env = LinearizedEnv(freq, z_list1, c_list1, rho_list1, attn_list1, c_hs, rho_hs, attn_hs, attn_units, N_list, cmin, cmax)
    
    z_arr =env.z_arr
    pert_c_arr = np.zeros((z_arr.size, 2))
    pert_c_arr[:, 0] = 1e-2
    pert_c_arr[:, 1] = (z_arr - np.mean(z_arr)) / np.max(z_arr)

    env.add_c_pert_matrix(z_arr,pert_c_arr)
    env.add_x0(np.zeros(2))

    now = time.time()
    env.full_forward_modes()
    print('time to get modes', time.time() - now)
    now = time.time()
    env.linearize()
    print('time to get mode linearization matrix', time.time() - now)

    env_fig, env_ax = plt.subplots(1, 1, figsize=(5, 5))
    env_ax.plot(env.c_arr, z_arr)
    env_ax.set_ylabel('Depth (m)')
    env_ax.set_xlabel('c')
    plt.gca().invert_yaxis()

    zs_arr = np.array([100.0])
    zr_arr = np.linspace(10.0, Z_bott, 100)
    r_arr = np.array([R])

    now = time.time()
    env.add_zs_arr(zs_arr)
    env.add_zr_arr(zr_arr)
    env.add_r_arr(r_arr)
    env.linearize_forward_p()
    print('time to linearize pressure', time.time() - now)
    now = time.time()
    p_arr = env.get_full_forward_p()
    print('time to run full compute pressure', time.time() - now)

    fig, axes= plt.subplots(1, 2, figsize=(10, 5))
    #plt.pcolormesh(r_arr, zr_arr, 10*np.log10(np.abs(p_arr[0,...])))
    axes[0].plot(zr_arr, np.abs(p_arr[0,:,0]))
    axes[1].plot(zr_arr, np.angle(p_arr[0,:,0]))


    x = np.array([3.0, 4.0]) # add a linear profile with constant offset of 3m/s

    perturbed_c = env.c_arr + env.pert_c_arr @ x
    env_ax.plot(perturbed_c, env.z_arr)
    now = time.time()
    p_linearized = env.get_linear_forward_p(x)
    print('linearized forward model run time ', time.time() - now)


    env.update_x0(x)
    p_full = env.get_full_forward_p()
    rel_err = np.abs(p_linearized[0,:,0] - p_full[0,:,0]) / np.abs(p_full[0,:,0])
    fig, axes= plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(np.abs(p_full[0,:,0]))
    axes[0].plot(np.abs(p_linearized[0,:,0]))
    axes[1].plot(np.angle(p_full[0,:,0]))
    axes[1].plot(np.angle(p_linearized[0,:,0]))
    
    plt.figure()
    plt.suptitle('Relative error in linearized model')
    plt.plot(zr_arr, np.abs(rel_err))
    return

if __name__ == '__main__':
    test1()
    test1()
