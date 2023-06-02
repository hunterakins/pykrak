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
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from pykrak.misc import get_simpsons_integrator, get_layer_N
from pykrak.sturm_seq import get_arrs
from numba import njit
import numba as nb
from pykrak.pressure_calc import get_phi_zr
from pykrak import pressure_calc as pc



@njit
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
@njit
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
@njit
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

@njit
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

@njit
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

@njit
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

@njit
def get_rfgh(krs, r):
    """
    Equation 20b in Thode and Kim (2004)
    Should refactor to avoid redundant calculations
    """
    M = krs.size
    #rfg_arr = np.zeros((M, M, M), dtype=nb.c8)
    rfg_arr = np.zeros((M, M, M), dtype=np.complex_)
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
                    
@njit
def get_dpda(zfg, krs, r, phi_zr, phi_zs):
    """
    Given the matrix Zfg for a given perturbation basis vector, 
    calculate the pressure gradient with respect to the perturbation strength
    for the pressure field due to source-receiver separation of r
    receiver depths zr, and source depth zs
    Using equation 18a in Thode and Kim (2004)
    """
    dpda = np.zeros(phi_zr.shape[0],dtype=np.complex_)
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

@njit
def get_dpdaidaj(zfg_ai, zfg_aj, zfg_aiaj, krs, r, phi_zr, phi_zs):
    """
    Equation 20 a) in Thode and Kim
    """

    rfgh = get_rfgh(krs, r)
    rfg = get_rfg(krs, r)

    Nr = phi_zr.shape[0]
    dpdaidaj = np.zeros((Nr), dtype=np.complex_)
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

def test_pekeris():
    """
    Test adjoint model by comparison with dpda
    """

    from pykrak.env_pert import PertEnv
    z_list = [np.array([0, 100.]), np.array([100., 146.])]
    c_list = [np.array([1500., 1500.]), np.array([1645., 1645. + 46.])]
    rho_list = [np.ones(2), 1.3*np.ones(2)]
    attn_list = [np.zeros(2), 0.25 * np.ones(2)]
    c_hs = 1800.
    rho_hs = 2.0
    attn_hs = 0.0
    attn_units ='dbplam'
    freq = 100.
    #z_list = [np.array([0, 5000.])]
    #c_list = [np.array([1500., 1500.])]
    #rho_list = [np.ones(2)]
    #attn_list = [np.zeros(2)]
    #c_hs = 1800.
    #rho_hs = 2.0
    #attn_hs = 0.0
    #attn_units ='dbplam'
    #freq = 10.

    zr = np.linspace(10, 90., 20)
    zs = np.array([50.])
    r = 1*1e3

    env = PertEnv(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units)
    env.plot_env()
    env.add_freq(freq)
    N_list = env.get_N_list() # get lambda / 20 spacing
    N_list= [1*x for x in N_list] # double it..
    rmax = 1e6
    krs = env.get_krs(**{'N_list': N_list, 'rmax':rmax})
    phi = env.phi

    z_list, c_list, rho_list, attn_list = env.interp_env_vals(N_list)
    h_list = [x[1] - x[0] for x in z_list]
    h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(h_list, z_list, c_list, rho_list)


    # now introduce a perturbation vector
    #delta_c = np.ones(env.c_list[0].size) # constant offset
    delta_c = np.array([-.4, .4]) # linear offset
    from copy import deepcopy
    tmp_c_list = deepcopy(c_list)
    tmp_delta_c = np.interp(z_list[0], env.z_list[0], delta_c)
    tmp_c_list[0] += tmp_delta_c
    _, _, _, new_c_arr, _ = get_arrs(h_list, z_list, tmp_c_list, rho_list)
    ci_arr = new_c_arr - c_arr # get it interpolated onto the same grid

    # also add it as a perturbation matrix...
    env.add_model_matrix(delta_c[:,None], 0)




    cref = 1500.
    omega = 2*np.pi*freq
    kref = omega/cref
    zfg = get_zfg_eta_ai(h_arr, ind_arr, ci_arr, z_arr, c_arr, rho_arr, krs, phi, omega)

    plt.figure()
    plt.pcolormesh(zfg)
    plt.colorbar()

    phi_z = env.phi_z

    phi_zr = get_phi_zr(zr, phi_z, phi)
    phi_zs = get_phi_zr(zs, phi_z, phi)
    dpda = get_dpda(zfg, krs, r, phi_zr, phi_zs)

    fig, axes = plt.subplots(1,2)
    axes[0].grid()
    axes[1].grid()
    axes[0].plot(abs(dpda), 'k')
    axes[1].plot(180/np.pi*np.angle(dpda), 'k')

    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(zs)
    from pykrak import pressure_calc as pc
    p1 = pc.get_pressure(phi_zr, phi_zs, krs, r)

    M = krs.size

    delta_a_grid = [2., 1., .1, .01, .001]
    dpda_mat = np.zeros((zr.size, len(delta_a_grid)), dtype=np.complex_)
    for i in range(len(delta_a_grid)):
        delta_a = delta_a_grid[i]
        delta_pert = np.array(delta_a).reshape(1,1)
        env.perturb_env([delta_pert])
        krs = env.get_krs(**{'N_list': N_list, 'rmax':rmax})
        if krs.size != M:
            print('krs size changed!, linear theory no good?')
        phi_zr = env.get_phi_zr(zr)
        phi_zs = env.get_phi_zr(zs)
        p_new = pc.get_pressure(phi_zr, phi_zs, krs, r)
        dp = p_new - p1
        dpda_mat[:,i] = dp[:,0] / delta_a
        axes[0].plot(abs(dp/delta_a))
        axes[1].plot(180/np.pi*np.angle(dp/delta_a))
        env.unperturb_env([delta_pert])

        
    plt.show()

def calc_pert_p(env, delta_pert, zr, zs, r):
    """
    Compute numerical forward difference first derivative
    for pressure field with parameter step delta a
    """
    dz = 1500 / (100*env.freq) #
    env.perturb_env([delta_pert])
    N_list = [int((env.z_list[i][-1] - env.z_list[i][0])/ dz) + 1 for i in range(len(env.z_list))]
    krs = env.get_krs(**{'N_list': N_list, 'Nh':1})
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(zs)
    p1 = pc.get_pressure(phi_zr, phi_zs, krs, r)
    p1 = p1[:,0]
    env.unperturb_env([delta_pert])
    return p1

def test_swellex():
    """
    Test adjoint model by comparison with dpda
    """

    from pykrak.envs import factory
    freq = 100.
    env = factory.create('swellex')(**{'pert':True})
    env.add_freq(freq)
    conv_factor = env.add_attn_conv_factor()
    dz = 1500 / (100*freq) #
    N_list = [int((env.z_list[i][-1] - env.z_list[i][0])/ dz) + 1 for i in range(len(env.z_list))]
    z_list, c_list, rho_list, attn_list= env.interp_env_vals(N_list)
    c_hs, rho_hs, attn_hs = env.c_hs, env.rho_hs, env.attn_hs
    h_list = [x[1] - x[0] for x in z_list]
    h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(h_list, z_list, c_list, rho_list)

    zr = np.linspace(20, 190., 4)
    zs = np.array([50.])
    r = 5*1e3

    env.plot_env()
    krs = env.get_krs(**{'N_list': N_list, 'Nh':1})
    phi = env.phi

    # now introduce a perturbation vector
    #delta_c = np.ones(env.c_list[0].size) # constant offset
    delta_c = env.z_list[0]-  np.mean(z_list[0]) # linear offset
    delta_c /= 100
    from copy import deepcopy
    tmp_c_list = deepcopy(c_list)
    tmp_delta_c = np.interp(z_list[0], env.z_list[0], delta_c)
    tmp_c_list[0] += tmp_delta_c
    _, _, _, new_c_arr, _ = get_arrs(h_list, z_list, tmp_c_list, rho_list)
    ci_arr = new_c_arr - c_arr # get it interpolated onto the same grid

    plt.figure()
    plt.plot(env.z_list[0], delta_c, 'b')
    plt.plot(z_arr, ci_arr, 'k')

    # also add it as a perturbation matrix...
    env.add_model_matrix(delta_c[:,None], 0)

    omega = 2*np.pi*freq
    zfg = get_zfg_eta_ai(h_arr, ind_arr, ci_arr, z_arr, c_arr, rho_arr, krs, phi, omega)

    phi_z = env.phi_z

    phi_zr = get_phi_zr(zr, phi_z, phi)
    phi_zs = get_phi_zr(zs, phi_z, phi)
    dpda = get_dpda(zfg, krs, r, phi_zr, phi_zs)


    # plot adjoint dpda
    fig, axes = plt.subplots(1,2)
    fig.suptitle('Comparison of adjoint and numerical derivatives')
    axes[0].grid()
    axes[1].grid()
    axes[0].plot(abs(dpda), 'k')
    axes[1].plot(180/np.pi*np.angle(dpda), 'k')

    # calculate pressure for unperterubed env for numerical calcs.
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(zs)
    from pykrak import pressure_calc as pc
    p0 = pc.get_pressure(phi_zr, phi_zs, krs, r)
    p0 = p0[:,0]


    print('p0', p0)

    # gen random vector for inner prod testing
    np.random.seed(2)
    tmp = np.random.randn(zr.size) + 1j*np.random.randn(zr.size)
    tmp /= np.linalg.norm(tmp) 
    print('d', tmp)
   
    
    adj_df = np.sum(dpda.conj()*tmp)

    ip0 = np.sum(np.conj(p0)*tmp)

    M = krs.size

    #delta_a_grid = [.01, .001, .0001, .00001, .000001]
    delta_a_grid = np.logspace(-4, -1, 3)

    d_ip_mat = np.zeros(len(delta_a_grid), dtype=np.complex_)
    dpda_mat = np.zeros((zr.size, len(delta_a_grid)), dtype=np.complex_)

    dpda_adj = dpda.copy()

    # loop over step size of parameter pert. for num. calc. of derivs.
    for i in range(len(delta_a_grid)):
        delta_a = delta_a_grid[i]
        delta_pert = np.array(delta_a).reshape(1,1)
        p1 = calc_pert_p(env, delta_pert, zr, zs, r)

        dp = p1-p0
        dpda = dp / delta_a
        dpda_mat[:,i] = dpda

        # now test inner product and norm sq chain rules
        ip1 = np.sum((p1.conj()*tmp))
        dipda = np.sum(np.conj(dp) * tmp) / delta_a #

        d_ip_mat[i] = dipda

        axes[0].plot(abs(dpda))
        axes[1].plot(180/np.pi*np.angle(dpda))


    resid = dpda_mat - dpda_adj[:, None]
    p_resid = np.linalg.norm(p0)
    resid_norm = np.linalg.norm(resid, axis=0)
    plt.figure()
    plt.plot(delta_a_grid, resid_norm / p_resid, 'b')
    plt.grid()
    plt.xlabel('delta_a')
    plt.ylabel('residual norm')
    plt.xscale('log')

    fig, axes = plt.subplots(1,2, sharex='all')
    fig.suptitle('Coparison of magnitude and phase of inner product derivatve')
    axes[0].plot(delta_a_grid, abs(d_ip_mat), 'b')
    axes[1].plot(delta_a_grid, 180/np.pi*np.angle(d_ip_mat), 'b')
    axes[0].plot(delta_a_grid, [abs(adj_df)]*len(delta_a_grid), 'k', label='adjoint')
    axes[1].plot(delta_a_grid, [180/np.pi*np.angle(adj_df)]*len(delta_a_grid), 'k', label='adjoint')
    axes[0].set_xscale('log')
    plt.legend()

    print('adj df', adj_df)
        
    plt.show()

def test_swellex_second_deriv():
    """
    Test adjoint model by comparison with dpda
    """

    from pykrak.envs import factory
    freq = 10.
    env = factory.create('swellex')(**{'pert':True})
    env.add_freq(freq)
    conv_factor = env.add_attn_conv_factor()
    dz = 1500 / (100*freq) # fine for good 
    N_list = [int((env.z_list[i][-1] - env.z_list[i][0])/ dz) + 1 for i in range(len(env.z_list))]
    z_list, c_list, rho_list, attn_list= env.interp_env_vals(N_list)
    c_hs, rho_hs, attn_hs = env.c_hs, env.rho_hs, env.attn_hs
    h_list = [x[1] - x[0] for x in z_list]
    h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(h_list, z_list, c_list, rho_list)

    zr = np.linspace(20, 190., 4)
    zs = np.array([50.])
    r = 5*1e3


    env.plot_env()
    krs = env.get_krs(**{'N_list': N_list, 'Nh':1})
    phi = env.phi

    # now introduce a perturbation vector
    #delta_c = np.ones(env.c_list[0].size) # constant offset
    delta_c = env.z_list[0]-  np.mean(z_list[0]) # linear offset
    delta_c /= 100
    from copy import deepcopy
    tmp_c_list = deepcopy(c_list)
    tmp_delta_c = np.interp(z_list[0], env.z_list[0], delta_c)
    tmp_c_list[0] += tmp_delta_c
    _, _, _, new_c_arr, _ = get_arrs(h_list, z_list, tmp_c_list, rho_list)
    ci_arr = new_c_arr - c_arr # get it interpolated onto the same grid

    plt.figure()
    plt.plot(env.z_list[0], delta_c, 'b')
    plt.plot(z_arr, ci_arr, 'k')

    # also add it as a perturbation matrix...
    env.add_model_matrix(delta_c[:,None], 0)

    omega = 2*np.pi*freq
    zfg = get_zfg_eta_ai(h_arr, ind_arr, ci_arr, z_arr, c_arr, rho_arr, krs, phi, omega)
    zfg_aa = get_zfg_eta_aiaj(h_arr, ind_arr, ci_arr, ci_arr, z_arr, c_arr, rho_arr, krs, phi, omega)

    phi_z = env.phi_z

    phi_zr = get_phi_zr(zr, phi_z, phi)
    phi_zs = get_phi_zr(zs, phi_z, phi)
    #dpda = get_dpda(zfg, krs, r, phi_zr, phi_zs) # adjoint deriv.

    dpda2 = get_dpdaidaj(zfg, zfg, zfg_aa, krs, r, phi_zr, phi_zs)


    # plot adjoint dpda
    fig, axes = plt.subplots(1,2)
    axes[0].grid()
    axes[1].grid()
    axes[0].plot(abs(dpda2), 'k')
    axes[1].plot(180/np.pi*np.angle(dpda2), 'k')

    # calculate pressure for unperterubed env for numerical calcs.
    phi_zr = env.get_phi_zr(zr)
    phi_zs = env.get_phi_zr(zs)
    p0 = pc.get_pressure(phi_zr, phi_zs, krs, r)
    p0 = p0[:,0]

    M = krs.size

    delta_a_grid = np.logspace(-3, -1, 10)

    dpda2_adj = dpda2.copy()

    # loop over step size of parameter pert. for num. calc. of derivs.
    dpda_mat = np.zeros((zr.size, len(delta_a_grid)), dtype=np.complex_)
    for i in range(len(delta_a_grid)):
        delta_a = delta_a_grid[i]
        print('delta a', delta_a)
        delta_pert = np.array(delta_a).reshape(1,1)
        pr = calc_pert_p(env, delta_pert, zr, zs, r)

        pl = calc_pert_p(env, -delta_pert, zr, zs, r)

        d2p = pr+ pl - 2*p0
        dpda = d2p / delta_a**2
        dpda_mat[:,i] = dpda

        axes[0].plot(abs(dpda), label=str(delta_a))
        axes[1].plot(180/np.pi*np.angle(dpda), label=str(delta_a))

    resid = dpda_mat - dpda2_adj[:, None]
    plt.legend()
    p_resid = np.linalg.norm(p0)
    resid_norm = np.linalg.norm(resid, axis=0)
    plt.figure()
    plt.plot(delta_a_grid, resid_norm / p_resid, 'b')
    plt.grid()
    plt.xlabel('delta_a')
    plt.ylabel('residual norm')
    plt.xscale('log')

    #fig, axes = plt.subplots(1,2, sharex='all')
    #axes[0].plot(delta_a_grid, abs(d_ip_mat), 'b')
    #axes[1].plot(delta_a_grid, 180/np.pi*np.angle(d_ip_mat), 'b')
    #axes[0].plot(delta_a_grid, [abs(adj_df)]*len(delta_a_grid), 'k')
    #axes[1].plot(delta_a_grid, [180/np.pi*np.angle(adj_df)]*len(delta_a_grid), 'k')
    #axes[0].set_xscale('log')

        
    plt.show()
    
if __name__ == '__main__':
    #test_pekeris()
    #test_swellex()
    test_swellex_second_deriv()

