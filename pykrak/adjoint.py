"""
Description:
    Normal mode adjoint implementation for derivatives in Thode and Kim (2004). 

    Uses asymptotic expression for Hankel function
    1) H_\alpha^{1}(x) \sim \sqrt{2/\pi} \exp{i (x - \alpha \pi / 2  - \pi/4)} / \sqrt{x} 
    2) H_\alpha^{2}(x) \sim \sqrt{2/\pi} \exp{-i (x - \alpha \pi / 2  - \pi/4)} / \sqrt{x} 

    Thode and Kim use a Fourier transform convention where the forward transform kernel is \exp{i \omega t}, so that
    the Hankel function of the first kind represents outward going waves (radiating away from a source on the origin)

    I use a forward transform kernel of \exp{-i \omega t} throughout my code. Therefore, outward going waves are of the second kind. 
    This requires a bit of care in the approximations in equations 20b, 18b

    In particular, H_2^2 = e^{i \pi} H_0^2 (asymptotically) = - H_0^2, and H_1^2 = e^{i \pi /2} H_0^2 = i H_0^2 (not -i as in T and K) 

Date:
    5/14/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from pykrak.misc import get_simpsons_integrator, get_layer_N
from pykrak.pressure_calc import get_phi_zr
from numba import njit, prange
import numba as nb



@njit(cache=True)
def get_layered_eta_ai_integrator(h_arr, ind_arr, z_arr, c_arr, ci_arr, rho_arr):
    """
    Input
    h_arr - array of mesh spacings for each layer
    ind_arr - array of index of first value for each layer
    z_arr - depths of the layer meshes concatenated
    c_arr - sound speed of the layer meshes concatenated
    rho_arr - density of the layer meshes concatenated
    ci_arr - sound speed perturbation of the layer meshes concatenated

    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for i in range(len(layer_N)):
        if i < num_layers -1 :
            c_i = c_arr[ind_arr[i]:ind_arr[i+1]]
            rho_i = rho_arr[ind_arr[i]:ind_arr[i+1]]
            delta_c_i = ci_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            c_i = c_arr[ind_arr[i]:]
            rho_i = rho_arr[ind_arr[i]:]
            delta_c_i = ci_arr[ind_arr[i]:]
        integrator_i = get_simpsons_integrator(layer_N[i], h_arr[i])[0,:]
        integrator_i *= -2*delta_c_i / (c_i**3 * rho_i)
        if i == 0:
            integrator = integrator_i
        else:
            integrator[-1] += integrator_i[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_i[1:]))
    return integrator

@njit(cache=True)
def get_layered_eta_aiaj_integrator(h_arr, ind_arr, z_arr, c_arr, ci_arr, cj_arr, rho_arr):
    """
    eta is index of refraction squared...this is partial deriv...
    Input
    h_arr - array of mesh spacings for each layer
    ind_arr - array of index of first value for each layer
    z_arr - depths of the layer meshes concatenated
    c_arr - sound speed of the layer meshes concatenated (sound speed about which the Taylor expansion is being computed)
    rho_arr - density of the layer meshes concatenated
    ci_arr - array of ith basis vector of SSP perts.
    cj_arr - array of jth basis vector of SSP perts.

    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for k in range(len(layer_N)):
        if k < num_layers -1 :
            c_k = c_arr[ind_arr[k]:ind_arr[k+1]]
            rho_k = rho_arr[ind_arr[k]:ind_arr[k+1]]
            ci_k = ci_arr[ind_arr[k]:ind_arr[k+1]]
            cj_k = cj_arr[ind_arr[k]:ind_arr[k+1]]
        else:
            c_k = c_arr[ind_arr[k]:]
            rho_k = rho_arr[ind_arr[k]:]
            ci_k = ci_arr[ind_arr[k]:]
            cj_k = cj_arr[ind_arr[k]:]
        integrator_k = get_simpsons_integrator(layer_N[k], h_arr[k])[0,:]
        integrator_k *= 6*ci_k*cj_k / (c_k**4 * rho_k)
        if k == 0:
            integrator = integrator_k
        else:
            integrator[-1] += integrator_k[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_k[1:]))
    return integrator

# independent of source position index of refraction equations...
@njit(cache=True)
def get_zfg_eta_ai(h_arr, ind_arr, ci_arr, z_arr, c_arr, rho_arr, krs, phi, omega):
    """
    Equation 18b in Thode and Kim (2004):
    $$ Z_{fg}(\eta_a) = k_{ref}^2 \int_{0}^{\infty} \frac{\eta_{a}(z')}{\rho(z')} U_f(z') U_g(z') \dd z' $$

    $$ \eta_{a}(z) = -2 \frac{c_{ref}^2}{c^3(z)} \Delta c_{a}(z) $$
    """
    M = krs.size # number of modes
    integrator = get_layered_eta_ai_integrator(h_arr, ind_arr, z_arr, c_arr, ci_arr, rho_arr)
    zfg_arr = np.zeros((M,M))

    for f in range(M):
        uf = phi[:,f]
        for g in range(f, M):
            ug = phi[:,g]
            integrand = uf * ug
            zfg = np.sum(integrand*integrator)
            zfg_arr[f,g] = zfg
            zfg_arr[g,f] = zfg # symmetric
    zfg_arr *= omega**2
    return zfg_arr

@njit(cache=True)
def get_zfg_eta_aiaj(h_arr, ind_arr, ci_arr, cj_arr, z_arr, c_arr, rho_arr, krs, phi, omega):
    """
    Equation 18b in Thode and Kim (2004) evaluated for \pdv[2]{\eta}{a_i}{a_j}
    $$ Z_{fg}(\eta_{a_i a_j}) = k_{ref}^2 \int_{0}^{\infty} \frac{\eta_{a_i a_j}(z')}{\rho(z')} U_f(z') U_g(z') \dd z' $$

    $$ \eta_{a_i a_j}(z) = 6 \frac{c_{ref}^2}{c^4(z)} c_i(z)  c_j(z) $$
    """
    M = krs.size # number of modes
    integrator = get_layered_eta_aiaj_integrator(h_arr, ind_arr, z_arr, c_arr, ci_arr, cj_arr, rho_arr)
    zfg_arr = np.zeros((M,M))

    for f in range(M):
        uf = phi[:,f]
        for g in range(f, M):
            ug = phi[:,g]
            integrand = uf * ug
            zfg = np.sum(integrand*integrator)
            zfg_arr[f,g] = zfg
            zfg_arr[g,f] = zfg # symmetric
    zfg_arr *= omega**2
    return zfg_arr

@njit(cache=True)
def H0(kr, r):
    """
    Asymptotic form of zeroth outward going Hankel function (technically the second
    kind I believe, if Wikipedia asymptotic forms are a universal definition)
    Consistent with a Fourier transform convention
    P(\omega) = \int_{-\infty}^{\infty} p(t) e^{- i \omega t} \dd t
    """
    p = np.exp(-1j*kr*r)
    p *= np.sqrt(2)*np.exp(1j*np.pi/4)
    p /= np.sqrt(kr.real*r*np.pi)
    return p

@njit(cache=True)
def get_rfg(krs, r):
    """
    Equation 18c in Thode and Kim (2004)
    $$ R_{fg} = \begin{cases} \frac{H_0(k_{rf} r) - H_0(k_{rg} r)}{k_{rf}^2 - k_{rg}^2} \text{    if } f \neq g \\ 
        \frac{i r H_0(k_{rf}r)}{2 k _{rf}} \text {   if  } f = g \\
        \end{cases} \; . $$

    r is source range separation

    Depending on FFT convention, Hankel function of first or second kind represents the outgoing wave
    I use the second kind, consistent with a Fourier transform kernel of exp(-i \omega t)
    Then, H1(kr r) = 1j *r * H_0 (kr r)

    """
    M = krs.size
    rfg_arr = np.zeros((M,M), dtype=nb.c8)
    for f in range(M):
        krf = krs[f]
        H0f = H0(krf, r)
        for g in range(f, M):
            krg = krs[g]
            if f == g:
                rfg = -1j*r*H0f / (2*krf) # minus sign because of different FFT convention used compared to T and K
                rfg_arr[f,g] = rfg
            else:
                numer = (H0f - H0(krg, r))
                denom = (krf**2 - krg**2)
                rfg =  numer / denom 
                rfg_arr[f,g] = rfg
                rfg_arr[g,f] = rfg #symmetric
    return rfg_arr

@njit(cache=True)
def get_rfgh(krs, r):
    """
    Equation 20b in Thode and Kim (2004)
    Should refactor to avoid redundant calculations
    """
    M = krs.size
    #rfg_arr = np.zeros((M, M, M), dtype=nb.c8)
    rfg_arr = np.zeros((M, M, M), dtype=np.complex128)
    krs_sq = krs**2
    for f in range(M):
        krf = krs[f]
        H0f = H0(krf, r)
        kr_sqf = krs_sq[f]
        for g in range(M):
            krg = krs[g]
            H0g = H0(krg, r)
            kr_sqg = krs_sq[g]
            for h in range(M):
                krh = krs[h]
                H0h = H0(krh, r)
                kr_sqh = krs_sq[h]
                if (f != g) and (f != h) and (g != h): #f neq h neq g
                    term1 = H0f /( (kr_sqf - kr_sqg) * (kr_sqf - kr_sqh))
                    term2 = H0g /( (kr_sqg - kr_sqf) * (kr_sqg - kr_sqh))
                    term3 = H0h /( (kr_sqh - kr_sqf) * (kr_sqh - kr_sqg))
                    term =term1+term2+term3
                    rfg_arr[f,g,h] = term
                elif (f != g) and (f != h): # g = h
                    top = -1j * r *  H0g  
                    bott = (2*krg *(kr_sqg - kr_sqf))
                    term = top / bott
                    rfg_arr[f,g,h] = term
                elif (f != g): # f = h neq g
                    top = -1j * r *  H0h  
                    bott = (2*krh *(kr_sqh - kr_sqg))
                    term = top / bott
                    rfg_arr[f,g,h] = term
                elif (f != h): # f = g neq h
                    top = -1j * r *  H0f  
                    bott = (2*krf *(kr_sqf - kr_sqh))
                    term = top / bott
                    rfg_arr[f,g,h] = term
                else: # f = g = h
                    term = -r**2 * H0f / (8*kr_sqf) 
                    rfg_arr[f,g,h] = term
    return rfg_arr
                    
@njit(cache=True)
def get_dpda(zfg, krs, r, phi_zr, phi_zs):
    """
    Given the matrix Zfg for a given perturbation basis vector, 
    calculate the pressure gradient with respect to the perturbation strength
    for the pressure field due to source-receiver separation of r
    receiver depths zr, and source depth zs
    Using equation 18a in Thode and Kim (2004)
    """
    dpda = np.zeros(phi_zr.shape[0],dtype=np.complex128)
    M = krs.size

    rfg = get_rfg(krs, r)

    for f in range(M):
        uf = phi_zr[:,f]
        for g in range(M):
            ug = phi_zs[0,g]
            term = uf*ug*zfg[f,g]*rfg[f,g]
            dpda += term
    dpda *= .25*1j
    
    return dpda

@njit(cache=True)
def get_mode_dpda(zfg, krs, r, phi_zs):
    """
    Get the derivative of the modal amplitudes with respect to perturbation to
    the environment
    Integrate dpda against U_f(z) / rho(z) dz to obtain
    d A / d a = ... \sum
    """
    M = krs.size
    mode_dpda = np.zeros(M, dtype=np.complex128)
    M = krs.size
    rfg = get_rfg(krs, r)
    prod = zfg * rfg
    for f in range(M):
        zg = zfg[f,:]
        rg = rfg[f,:]
        for g in range(M):
            ug = phi_zs[0,g]
            term = ug*zfg[f,g]*rfg[f,g]
            mode_dpda[f] += term
    mode_dpda *= .25*1j
    return mode_dpda


@njit(cache=True)
def get_dpdaidaj(zfg_ai, zfg_aj, zfg_aiaj, krs, r, phi_zr, phi_zs):
    """
    Equation 20 a) in Thode and Kim
    """

    rfgh = get_rfgh(krs, r)
    rfg = get_rfg(krs, r)

    Nr = phi_zr.shape[0]
    dpdaidaj = np.zeros((Nr), dtype=np.complex128)
    M = krs.size
    for f in range(M):
        for g in range(M):
            # do the first sum contributions
            term = zfg_ai[f,g]*phi_zr[:,f] # independent of h
            for h in range(M):
                dpdaidaj += term * zfg_aj[g,h]*rfgh[f,g,h]*phi_zs[:,h]
            # now second sum contribution
            term = .5*zfg_aiaj[f,g]*rfg[f,g]*phi_zr[:,f]*phi_zs[:,g]
            dpdaidaj += term
    dpdaidaj *= .5*1j
    return dpdaidaj

@njit(cache=True)
def get_dpda_arr(zfg_arr, krs, r_arr, phi_zr, phi_zs):
    """
    Get the derivative of the pressure with respect to the environment
    for all of the parameters implicitly specificed in zfg_arr
    zfg_arr - np 3d array
        first axis is num parameters, second is mode number and third is mode number
    krs - np 1d array
        wavenumbers
    r_arr - np 1d array
        array of ranges
    phi_zr - np 2d array
        first index is receiver depth index, second is mode number
    phi_zs - np 2d array
        first index is source depth index, second is mode number
    """

    num_zs = phi_zs.shape[0]
    num_zr = phi_zr.shape[0]
    num_r = r_arr.size
    P = zfg_arr.shape[0]    
    dpda = np.zeros((num_zs, num_zr, num_r, P), dtype=np.complex128)
    for k in range(P):
        zfg_k = zfg_arr[k, :, :]
        for zs_i in range(num_zs):
            phi_zs_i = (phi_zs[zs_i,:])
            phi_zs_i = np.reshape(phi_zs_i, (1, phi_zs_i.size))
            for r_i in range(num_r):
                dpda[zs_i, :, r_i, k] = get_dpda(zfg_k, krs, r_arr[r_i], phi_zr, phi_zs_i)
    return dpda

@njit(cache=True)
def get_mode_dpda_arr(zfg_arr, krs, r_arr, phi_zs):
    """
    Get the derivative of the mode amplitude with respect to the environment
    for all of the parameters implicitly specificed in zfg_arr
    zfg_arr - np 3d array
        first axis is num parameters, second is mode number and third is mode number
    krs - np 1d array
        wavenumbers
    r_arr - np 1d array
        array of ranges
    phi_zs - np 2d array
        first index is source depth index, second is mode number
    """

    num_zs = phi_zs.shape[0]
    M = krs.size
    num_r = r_arr.size
    P = zfg_arr.shape[0]    
    mode_dpda = np.zeros((M, num_zs, num_r, P), dtype=np.complex128)
    for k in range(P):
        zfg_k = zfg_arr[k, :, :]
        for zs_i in range(num_zs):
            phi_zs_i = (phi_zs[zs_i,:])
            phi_zs_i = np.reshape(phi_zs_i, (1, phi_zs_i.size))
            for r_i in range(num_r):
                mode_dpda[:, zs_i, r_i, k] = get_mode_dpda(zfg_k, krs, r_arr[r_i], phi_zs_i)
    return mode_dpda



@njit(cache=True)
def get_kernel(h_arr, ind_arr, z_arr, c_arr, rho_arr, krs, phi_z, phi, omega, zr, zs, rgrid, stride):
    """
    Get VTSK
    Input
    h_arr - np 1d array of floats
        mesh spacings for each layer
    ind_arr - np 1d array of ints
        index of first value for each layer
    z_arr - np 1d array
        depths of the layer meshes concatenated
    c_arr - np 1d array
        sound speed of the layer meshes concatenated
    rho_arr - np 1d array
        density of the layer meshes concatenated
    krs -np 1d array
        wavenumbers to use in the kernel
    phi_z - np 1d array
        grid that mode depth function are evaluated on
    phi - np 2d array
        mode depth functions
        first axis is depth, second is mode number
    omega - float
        angular frequency
    zr - np 1d array
        receiver depths
    zs - np 1d array with single element (float)
        source depth
    rgrid - np 1d array
        grid of source ranges
    stride - int
        downsample the depths


    Output - 
    integrator - np array that contains the weights to apply to the mode product
        to get the simpsons rule integration of the mode product with eta (equation 18b)
    """
    layer_N = get_layer_N(ind_arr, z_arr)
    num_layers = len(layer_N)
    for i in range(len(layer_N)):
        if i < num_layers -1 :
            c_i = c_arr[ind_arr[i]:ind_arr[i+1]]
            rho_i = rho_arr[ind_arr[i]:ind_arr[i+1]]
        else:
            c_i = c_arr[ind_arr[i]:]
            rho_i = rho_arr[ind_arr[i]:]
        integrator_i = get_simpsons_integrator(layer_N[i], h_arr[i])[0,:]
        integrator_i *= 1 / (c_i**3 * rho_i)
        if i == 0:
            integrator = integrator_i
        else:
            integrator[-1] += integrator_i[0] # phi shares points with previous layer
            integrator = np.concatenate((integrator, integrator_i[1:]))
    integrator *= -2*omega**2

    phi_zr = get_phi_zr(zr, phi_z, phi)
    phi_zs = get_phi_zr(zs, phi_z, phi)



    integrator = integrator[::stride]
    M = krs.size
    kernel = np.zeros((zr.size, rgrid.size, integrator.size), dtype=np.complex128)
    for j in range(zr.size):
        kernel_j = np.zeros((rgrid.size, integrator.size), dtype=np.complex128)
        for f in range(M):
            uf = phi[::stride,f]
            krf = krs[f]
            kernel_j += np.outer(-1j*rgrid*H0(krf,rgrid), uf**2 * phi_zr[j,f] * phi_zs[0,f]/ (2*krf))
            for g in range(M):
                krg = krs[g]
                ug = phi[::stride,g]
                if g == f:
                    pass
                else:
                    kernel_j += np.outer(H0(krf,rgrid) - H0(krg,rgrid), uf * ug * phi_zr[j,f] * phi_zs[:,g] /(krf**2 - krg**2))
                #integrand = uf * ug
        kernel_j = kernel_j*integrator
        kernel[j,...] = kernel_j
    kernel *= 1j / 4
    return kernel

