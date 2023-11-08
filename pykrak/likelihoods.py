"""
Description:
    Likelihood functions using normal mode pressure field...

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak.raw_pykrak import get_modes
from pykrak.pressure_calc import get_pressure, get_delta_R, get_phi_zr
from pykrak.adjoint import get_phi_zr, get_zfg_eta_ai, get_rfg, get_dpda, get_dpdaidaj, get_zfg_eta_aiaj
from numba import njit
import numba  as nb

@njit
def ml_amplitude(d, p):
    """
    Measurement d,
    model vector p
    Get maximum likelihood amplitude of source
    """
    alpha = np.sum(p.conj()*d)/np.sum(p.conj()*p)
    return alpha

@njit#(nb.f8(nb.c16[:], nb.c16[:]))
def get_alpha_sq(d, p):
    """
    Data d, model vector p
    """
    numer = np.abs(np.sum(d.conj()*p))**2
    denom = (np.sum(p.conj()*p) * np.sum(d.conj()*d))
    alpha_sq = numer /denom
    alpha_sq = alpha_sq.real
    return alpha_sq


#@njit
def get_ml_noise_lh_pdv(alpha_sq, d, p, dpda):
    """
    partial derivative of noise max-likelihood likelihood function
    given data vec d, model vec p, and model deriv with respect to param
    a dpda
    """
    p_nm_sq = np.sum(np.conj(p)*p).real
    d_nm_sq = np.sum(np.conj(d)*d).real
    dp = np.sum(np.conj(d)*p)
    pd = dp.conj()

    #inn_prod1 = np.sum(np.conj(dpda)*d)
    #inn_prod2 = np.sum(np.conj(dpda)*p)

    vec = dp *d - alpha_sq*d_nm_sq*p
    numer = np.sum(np.conj(dpda)*vec).real
    denom = p_nm_sq*d_nm_sq

    grad = numer / denom
    grad /= (1-alpha_sq)
    grad = grad.real
    return grad

def get_ml_noise_lh_sec_deriv(x, d, p, pi, pj, pij):
    """
    x is corr. coeff squared
    d is data
    p is model pressure at current ssp state m
    pi is gradient of pressure w.r.t. ith parameter mi
    pij is Hessian of pressure w.r.t. i,j th parameter
    """

    gamma = np.abs(np.sum(d.conj()*p))**2
    delta = np.sum(p.conj()*p).real
    dp = np.sum(d.conj()*p)

    gamma_i = 2 * (np.sum(pi.conj()*d)*dp).real
    gamma_j = 2 * (np.sum(pj.conj()*d)*dp).real
    gamma_ij = 2 * (np.sum(pij.conj()*d)*dp + np.sum(pi.conj()*d) * np.sum(d*pj.conj())).real

    pip = np.sum(pi.conj() * p)
    pjp = np.sum(pj.conj() * p)

    delta_i = 2*pip.real
    delta_j = 2*pjp.real
    delta_ij = 2*(np.sum(pi.conj()* pj) + np.sum(pij.conj()* p)).real

    gamma_i = gamma_i / delta
    gamma_j = gamma_j / delta
    gamma_ij = gamma_ij / delta

    delta_i = delta_i / delta
    delta_j = delta_j / delta
    delta_ij = delta_ij / delta


    xij = gamma_ij 
    xij -= (gamma_i * delta_j + gamma_j * delta_i + gamma*delta_ij) 
    xij += 2 * gamma * delta_i *delta_j 

    xi = gamma_i - gamma * delta_i 
    xj = gamma_j  - gamma * delta_j 

    norm_d_sq = np.sum(d.conj()*d).real

    xij /= norm_d_sq
    xi /= norm_d_sq
    xj /= norm_d_sq

    
    lh_ij = .5* (xij) / (1-x) + .5*xi/(1-x) * xj / (1-x)
    if abs(lh_ij) > 70:
        print(lh_ij)
        print('xij, x, xi, xj', xij, x, xi, xj, (1-x)**2)
        print('gammai', gamma_i)
        print('gammaj', gamma_j)
        print('gammaij', gamma_ij)
        print('deltai', delta_i)
        print('deltaj', delta_j)
        print('deltaij', delta_ij)
        print(np.sum(pij.conj()*d)*dp)
        print(np.sum(pi.conj()*d) * np.sum(d*pj.conj()))
        sys.exit(0)

    return lh_ij

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
        krs, phi, phi_z = get_modes(freq, h_arr, ind_arr, z_arr, c_arr+delta_c, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs = get_phi_zr(np.array([zs]), phi_z, phi)
        deltaR = get_delta_R(tilt, zr)
        p =get_pressure(phi_zr, phi_zs, krs, r,deltaR)
        alpha_sq = get_alpha_sq(d[:,0], p[:,0])
        log_lh = -.5*np.log(1- alpha_sq)
        return log_lh
    return lh

def get_ml_noise_lh_with_gradient(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Use columns of model matrix to get new sound speeds
    d is data vector
    """
    omega = 2*np.pi*freq
    def lh(r, zs, zr, tilt, x):
        """
        """

        # change the environment and rerun mode code
        delta_c = model_matrix@x
        delta_c = delta_c.reshape(delta_c.size)
        tmp_c_arr = c_arr + delta_c
        krs, phi, phi_z = get_modes(freq, h_arr, ind_arr, z_arr, tmp_c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs = get_phi_zr(np.array([zs]), phi_z, phi)
        deltaR = get_delta_R(tilt, zr)

        # get pressure and likeliihood
        p =get_pressure(phi_zr, phi_zs, krs, r,deltaR)
        alpha_sq = get_alpha_sq(d[:,0], p[:,0])
        log_lh = -.5*np.log(1- alpha_sq)

           
        # compute likelihood gradient using adjoint
        num_params = x.size
        grad = np.zeros(num_params)
        for par_i in range(num_params):
            delta_c_arr = model_matrix[:,par_i]
            zfg = get_zfg_eta_ai(h_arr, ind_arr, delta_c_arr, z_arr, tmp_c_arr, rho_arr, krs, phi, omega)
            dpda = get_dpda(zfg, krs, r, phi_zr, phi_zs)
            dlhda = get_ml_noise_lh_pdv(alpha_sq, d[:,0], p[:,0], dpda)
            grad[par_i] = dlhda
        return log_lh, grad
    return lh


def get_ml_noise_lh_with_grad_hess(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Log likelihood function with ML estimate of noise conditioned on data d
    Also compute gradient of log-lh with respect to sound speed basis vectors in model_matrix
    columns
    Also copmute Hessian of log-lh ....
    """
    omega = 2*np.pi*freq
    def lh(r, zs, zr, tilt, x):
        """
        """

        # change the environment and rerun mode code
        delta_c = model_matrix@x
        delta_c = delta_c.reshape(delta_c.size)
        tmp_c_arr = c_arr + delta_c # this is the c_arr about which we compute gradient
        krs, phi, phi_z = get_modes(freq, h_arr, ind_arr, z_arr, tmp_c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs = get_phi_zr(np.array([zs]), phi_z, phi)
        deltaR = get_delta_R(tilt, zr)

        # get pressure and likeliihood
        p =get_pressure(phi_zr, phi_zs, krs, r,deltaR)
        alpha_sq = get_alpha_sq(d[:,0], p[:,0])
        log_lh = -.5*np.log(1- alpha_sq)

           
        # compute likelihood gradient using adjoint
        num_params = x.size
        grad = np.zeros(num_params)
        hess = np.zeros((num_params, num_params))
        for par_i in range(num_params):
            ci_arr = model_matrix[:,par_i]
            zfg_i = get_zfg_eta_ai(h_arr, ind_arr, ci_arr, z_arr, tmp_c_arr, rho_arr, krs, phi, omega)
            p_i = get_dpda(zfg_i, krs, r, phi_zr, phi_zs)
            lh_i = get_ml_noise_lh_pdv(alpha_sq, d[:,0], p[:,0], p_i)
            grad[par_i] = lh_i
            for par_j in range(par_i, num_params):
                cj_arr = model_matrix[:,par_j]
                if par_j > par_i:
                    zfg_j = get_zfg_eta_ai(h_arr, ind_arr, cj_arr, z_arr, tmp_c_arr, rho_arr, krs, phi, omega)
                    p_j = get_dpda(zfg_j, krs, r, phi_zr, phi_zs)
                else:
                    zfg_j = zfg_i
                    p_j = p_i
                zfg_ij = get_zfg_eta_aiaj(h_arr, ind_arr, ci_arr, cj_arr, z_arr, tmp_c_arr, rho_arr, krs, phi, omega)
                
                p_ij = get_dpdaidaj(zfg_i, zfg_j, zfg_ij, krs, r, phi_zr, phi_zs)
                lh_ij = get_ml_noise_lh_sec_deriv(alpha_sq, d[:,0], p[:,0], p_i, p_j, p_ij)
                hess[par_i, par_j] = lh_ij
                hess[par_j, par_i] = lh_ij
        return log_lh, grad, hess
    return lh

def get_ml_noise_lh_gridded(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Use columns of model matrix to get new sound speeds
    d is data vector
    """
    def lh(rgrid, zgrid, zr, tilt, x):
        """
        """
        delta_c = model_matrix@x
        krs, phi, phi_z = get_modes(freq, h_arr, ind_arr, z_arr, c_arr+delta_c, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)
        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs_grid = get_phi_zr(zgrid, phi_z, phi)
        deltaR = get_delta_R(tilt, zr)
        M = phi_zr.shape[1]

        p_grid = np.zeros((zr.size, zgrid.size, rgrid.size),dtype=np.complex128)
        log_lh_grid = np.zeros((zgrid.size, rgrid.size),dtype=np.float64)
        # fill out pgrid
        for i in range(zr.size):
            for j in range(rgrid.size):
                phi_weights = phi_zr[i,:].reshape((1, M))
                p =get_pressure(phi_zs_grid, phi_weights, krs, rgrid[j],np.array(deltaR[i]))
                p_grid[i,:,j] = p[:,0]

        # now comute likelihoods
        for i in range(zgrid.size):
            for j in range(rgrid.size):
                alpha_sq = get_alpha_sq(d[:,0], p_grid[:,i,j])
                log_lh = -.5*np.log(1- alpha_sq)
                log_lh_grid[i,j] = log_lh
        return log_lh_grid
    return lh
        
def get_ml_noise_lh_gridded_with_gradient(d, freq, h_arr, ind_arr, z_arr, c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax, model_matrix):
    """
    Use columns of model matrix to get new sound speeds
    d is data vector
    """
    omega = 2*np.pi*freq

    @njit
    def lh(rgrid, zgrid, zr, tilt, x):
        """
        """
        delta_c = model_matrix@x
        delta_c = delta_c.reshape((delta_c.size))
        tmp_c_arr = c_arr + delta_c # the SSP profile for this parameter point x
        krs, phi, phi_z = get_modes(freq, h_arr, ind_arr, z_arr, tmp_c_arr, rho_arr, attn_arr, c_hs, rho_hs, attn_hs, cmin, cmax)


        num_params = x.size
        for par_i in range(num_params):
            delta_c_arr = model_matrix[:,par_i]
            zfg = get_zfg_eta_ai(h_arr, ind_arr, delta_c_arr, z_arr, tmp_c_arr, rho_arr, krs, phi, omega)

        phi_zr = get_phi_zr(zr, phi_z, phi)
        phi_zs_grid = get_phi_zr(zgrid, phi_z, phi)
        deltaR = get_delta_R(tilt, zr)
        M = phi_zr.shape[1]

        p_grid = np.zeros((zr.size, zgrid.size, rgrid.size),dtype=np.complex128)
        log_lh_grid = np.zeros((zgrid.size, rgrid.size),dtype=np.float64)
        # fill out pgrid
        for i in range(zr.size):
            for j in range(rgrid.size):
                phi_weights = phi_zr[i,:].reshape((1, M))
                dR = np.array(deltaR[i,0])
                p =get_pressure(phi_zs_grid, phi_weights, krs, rgrid[j],dR)
                p_grid[i,:,j] = p[:,0]

        # now comute likelihoods
        for i in range(zgrid.size):
            for j in range(rgrid.size):
                alpha_sq = get_alpha_sq(d[:,0], p_grid[:,i,j])
                log_lh = -.5*np.log(1- alpha_sq)
                log_lh_grid[i,j] = log_lh
        return log_lh_grid
    return lh
