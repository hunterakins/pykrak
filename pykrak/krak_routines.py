"""
Description:

Date:

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from numba import njit

@njit
def initialize(h_arr, ind_arr, z_arr, omega2, cp_arr, cs_arr, rho_arr, cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, c_low, c_high):
    """
    Initializes arrays defining difference equations.

    Args:
    h_arr - mesh size for the different layers
    ind_arr - index of the start of each layer in z_arr
    z_arr - array of depths , contains doubled interface points
    omega2 - squared angular frequency
    cp_arr - compressional wave speed (complex)
    cs_arr - shear wave speed (complex)
    rho_arr - density

    cp_top - compressional wave speed in top hs
    cs_top - shear wave speed in top hs
    rho_top - density in top hs

    rho_top = 0 for Pressure Release, rho_top = 1e10 for Rigid
    Same for rho_bott

    cp_bott - compressional wave speed in bottom hs
    cs_bott - shear wave speed in bottom hs
    rho_bott - density in bottom hs

    c_low is min phase speed 
    c_high is max phase speed

    There should be the same number of points

    """
    elastic_flag = False # set to true if any media are elastic
    c_min = np.inf
    Nmedia = h_arr.size # number of layers
    n_points = z_arr.size # z_arr contains the doubled interface depths
    first_acoustic = -1
    last_acoustic = 0

    # Allocate arrays
    b1 = np.zeros(n_points, dtype=np.float64)
    b1c = np.zeros(n_points, dtype=np.complex128)
    b2 = np.zeros(n_points, dtype=np.float64)
    b3 = np.zeros(n_points, dtype=np.float64)
    b4 = np.zeros(n_points, dtype=np.float64)
    rho_arr = rho_arr.copy()

    # Process each medium
    for medium in range(Nmedia):
        ii = ind_arr[medium]
        if medium == Nmedia - 1:
            Nii = z_arr[ii:].size
        else:
            Nii = z_arr[ii : ind_arr[medium+1]].size

        # Load diagonals
        if np.real(cs_arr[ii]) == 0.0:  # Acoustic medium
            c_min = min(c_min, np.min(np.real(cp_arr[ii:ii + Nii])))
            if first_acoustic == -1:
                first_acoustic = medium
            last_acoustic= medium
            b1[ii:ii + Nii] = -2.0 + h_arr[medium]**2 * np.real(omega2 / (cp_arr[ii:ii + Nii])**2)
            b1c[ii:ii + Nii] = 1j * np.imag(omega2 / (cp_arr[ii:ii + Nii])**2)

        else:  # Elastic medium
            elastic_flag = True
            two_h = 2.0 * h_arr[medium]
            for j in range(ii, ii + Nii):
                c_min = min(np.real(cs_arr[j]), c_min)
                cp2 = np.real(cp_arr[j]**2)
                cs2 = np.real(cs_arr[j]**2)
                b1[j] = two_h / (rho_arr[j] * cs2)
                b2[j] = two_h / (rho_arr[j] * cp2)
                b3[j] = 4.0 * two_h * rho_arr[j] * cs2 * (cp2 - cs2) / cp2
                b4[j] = two_h * (cp2 - 2.0 * cs2) / cp2
                rho_arr[j] *= two_h * omega2

    if (rho_top == 0.0) or (rho_top == 1e10): # pressure or rigid, no need to overwrite c_high
        pass
    else:
        if cs_top != 0.0:
            elastic_flag = True
            c_min = min(c_min, np.real(cs_top))
            c_high = min(c_high, np.real(cs_top))
        else:
            c_min = min(c_min, np.real(cp_top))

    if (rho_bott == 0.0) or (rho_bott == 1e10): # pressure or rigid, no need to overwrite c_high
        pass
    else:
        if cs_bott != 0.0:
            elastic_flag = True
            c_min = min(c_min, np.real(cs_bott))
            c_high = min(c_high, np.real(cs_bott))
        else:
            c_min = min(c_min, np.real(cp_bott))
            c_high = min(c_high, np.real(cp_bott))



    if elastic_flag: # for Scholte wave
        c_min *= 0.85
    c_low = max(c_low, c_min)

    return b1, b1c, b2, b3, b4, rho_arr, c_low, c_high, elastic_flag, first_acoustic, last_acoustic

@njit
def get_f_g(cp, cs, rho, x, omega2, mode_count):
    if rho == 0.0: # Vacuum
        f = 1.0
        g = 0.0
        yV = np.array([f, g, 0.0, 0.0, 0.0])
    elif rho == 1e10: # Rigid
        f = 0.0
        g = 1.0
        yV = np.array([f, g, 0.0, 0.0, 0.0])
    else: # Acousto-elastic halfspace
        if cs.real > 0.0:
            gammaS2 = x- (omega2 / cs.real**2)
            gammaP2 = x - (omega2 / cp.real**2)
            gammaS = np.sqrt(gammaS2).real
            gammaP = np.sqrt(gammaP2).real
            mu = rho * cs.real**2

            yV = np.zeros(5)
            yV[0] = (gammaS * gammaP - x) / mu
            yV[1] = ((gammaS2 + x)**2 - 4.0 * gammaS * gammaP * x) * mu
            yV[2] = 2.0 * gammaS * gammaP - gammaS2 - x
            yV[3] = gammaP * (x - gammaS2)
            yV[4] = gammaS * (gammaS2 - x)

            f = omega2 * yV[3]
            g = yV[1]
            if g > 0.0:
                mode_count += 1

        else:
            gammap = np.sqrt(x - omega2 / cp**2)
            f = gammap
            g = rho
            yV = np.array([1e10, 1e10, 1e10, 1e10, 1e10])
    return f, g, yV

@njit
def elastic_up(x, yV, iPower, h, b1, b2, b3, b4, rho_arr, Floor, Roof, iPowerR, iPowerF):
    """
    Propagates up through a single elastic layer using compound matrix formulation.
    
    Parameters:
    x : float
        Trial eigenvalue, k2.
    yV : numpy.ndarray
        Solution of differential equation (initial values passed as input).
    iPower : int
        Power scaling factor, modified during computation.
    h : float 
        layer thickness for the medium
    b1, b2, b3, b4, rho : array of the discretized wave equation arrays freom initialize
    Floor, Roof : float
        Scaling thresholds.
    iPowerR, iPowerF : int
        Scaling adjustments for `iPower`.
    Returns:
    tuple
        Updated yV, iPower.
    """
    # Initialize variables
    two_x = 2.0 * x
    two_h = 2.0 * h
    four_h_x = 4.0 * h * x
    j = b1.size - 1
    xb3 = x * b3[j] - rho_arr[j]

    zV = np.zeros(5)
    zV[0] = yV[0] - 0.5 * (b1[j] * yV[3] - b2[j] * yV[4])
    zV[1] = yV[1] - 0.5 * (-rho_arr[j] * yV[3] - xb3 * yV[4])
    zV[2] = yV[2] - 0.5 * (two_h * yV[3] + b4[j] * yV[4])
    zV[3] = yV[3] - 0.5 * (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
    zV[4] = yV[4] - 0.5 * (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

    # Modified midpoint method
    N = b1.size
    for ii in range(N-1):
        j -= 1
        #print('EUP, ii, j, Yv', ii, j, yV)
        #print('b1[j], b2[j], b3[j], b4[j], rho_arr[j]', b1[j], b2[j], b3[j], b4[j], rho_arr[j])

        xV = yV.copy()
        yV = zV.copy()

        xb3 = x * b3[j] - rho_arr[j]

        zV[0] = xV[0] - (b1[j] * yV[3] - b2[j] * yV[4])
        zV[1] = xV[1] - (-rho_arr[j] * yV[3] - xb3 * yV[4])
        zV[2] = xV[2] - (two_h * yV[3] + b4[j] * yV[4])
        zV[3] = xV[3] - (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
        zV[4] = xV[4] - (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

        # Scale if necessary
        if ii != N-2:
            if abs(zV[1]) < Floor:
                zV *= Roof
                yV *= Roof
                iPower -= iPowerR
            elif abs(zV[1]) > Roof:
                zV *= Floor
                yV *= Floor
                iPower -= iPowerF

    # Apply the standard filter at the terminal point
    yV = (xV + 2.0 * yV + zV) / 4.0

    #print('Yv final', yV)

    return yV, iPower

@njit
def elastic_down(x, yV, iPower, h, b1, b2, b3, b4, rho_arr, Floor, Roof, iPowerR, iPowerF):
    """
    Propagates down through a single elastic layer using compound matrix formulation.
    
    Parameters:
    x : float
        Trial eigenvalue, k2.
    yV : numpy.ndarray
        Solution of differential equation (initial values passed as input).
    iPower : int
        Power scaling factor, modified during computation.
    h : float 
        layer thickness for the medium
    b1, b2, b3, b4, rho : array of the discretized wave equation arrays freom initialize
    Floor, Roof : float
        Scaling thresholds.
    iPowerR, iPowerF : int
        Scaling adjustments for `iPower`.
    Returns:
    tuple
        Updated yV, iPower.
    """
    # Initialize variables
    two_x = 2.0 * x
    two_h = 2.0 * h
    four_h_x = 4.0 * h * x
    j = 0
    xb3 = x * b3[j] - rho_arr[0]

    zV = np.zeros(5)
    #print(yV.dtype, b1.dtype, b2.dtype, b3.dtype, b4.dtype, rho_arr.dtype)
    zV[0] = yV[0] + 0.5 * (b1[j] * yV[3] - b2[j] * yV[4])
    zV[1] = yV[1] + 0.5 * (-rho_arr[j] * yV[3] - xb3 * yV[4])
    zV[2] = yV[2] + 0.5 * (two_h * yV[3] + b4[j] * yV[4])
    zV[3] = yV[3] + 0.5 * (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
    zV[4] = yV[4] + 0.5 * (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

    # Modified midpoint method
    N = b1.size
    for ii in range(N - 1):
        j += 1
        #print('EDOwn, ii, j, Yv', ii, j, yV)

        xV = yV.copy()
        yV = zV.copy()

        xb3 = x * b3[j] - rho_arr[j]

        zV[0] = xV[0] + (b1[j] * yV[3] - b2[j] * yV[4])
        zV[1] = xV[1] + (-rho_arr[j] * yV[3] - xb3 * yV[4])
        zV[2] = xV[2] + (two_h * yV[3] + b4[j] * yV[4])
        zV[3] = xV[3] + (xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2])
        zV[4] = xV[4] + (rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2])

        # Scale if necessary
        if ii != N-1:
            if abs(zV[1]) < Floor:
                zV *= Roof
                yV *= Roof
                iPower -= iPowerR
            elif abs(zV[1]) > Roof:
                zV *= Floor
                yV *= Floor
                iPower -= iPowerF

    # Apply the standard filter at the terminal point
    yV = (xV + 2.0 * yV + zV) / 4.0

    #print('Yv final', yV)

    return yV, iPower

@njit
def get_bc_impedance(x, omega2, top_flag, cp, cs, rho,
                        h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr,
                        first_acoustic, last_acoustic,
                        mode_count):
    """
    Compute the impedance functions for the top and bottom halfspaces
    top_flag - True if the top boundary
    """
    iPower = 0
    Floor = 1e-50
    Roof = 1e50
    iPowerR = 50
    iPowerF = -50

    f, g, Yv = get_f_g(cp, cs, rho, x, omega2, mode_count)
    if top_flag:
        g = -g 

    # Shoot through elastic layers if necessary
    if top_flag:
        if first_acoustic !=0 : # there are elastic layers on the surface
            for medium in range(first_acoustic):
                if medium == ind_arr.size - 1:
                    i0, i1 = ind_arr[medium], ind_arr[medium] + z_arr[ind_arr[medium]:].size
                else:
                    i0, i1 = ind_arr[medium], ind_arr[medium+1]
                Yv, iPower = elastic_down(x, Yv, iPower, h_arr[medium], 
                                                b1[i0:i1], b2[i0:i1], b3[i0:i1], b4[i0:i1], 
                                                rho_arr[i0:i1], Floor, Roof, iPowerR, iPowerF)
            f = omega2 * Yv[3]
            g = Yv[1]
    else:
        if last_acoustic != h_arr.size - 1: # there are elastic layers below
            if np.all(Yv == 1e10):
                raise ValueError('Yv is not initialized, need to use rigid halfspace when shooting up through elastic layers')
            for medium in range(h_arr.size-1, last_acoustic, -1):
                #print('medium', medium)
                if medium == ind_arr.size - 1:
                    i0, i1 = ind_arr[medium], ind_arr[medium] + z_arr[ind_arr[medium]:].size
                else:
                    i0, i1 = ind_arr[medium], ind_arr[medium+1]
                Yv, iPower = elastic_up(x, Yv, iPower, h_arr[medium], 
                                                b1[i0:i1], b2[i0:i1], b3[i0:i1], b4[i0:i1], 
                                                rho_arr[i0:i1], Floor, Roof, iPowerR, iPowerF)
            f = omega2 * Yv[3]
            g = Yv[1]
    return f, g, iPower, mode_count

@njit
def acoustic_layers(x, f, g, iPower, ind_arr, h_arr, z_arr, b1, rho_arr, 
                    CountModes, mode_count, first_acoustic, last_acoustic):
    """
    Shoot through acoustic layers
    from the bottom, where there is a boundary condition
    f p + g \dv{p}{z} / rho(z-) = 0
    x - float 64, eigenvalue is x = kr^2
    f, g are impedance functions at the bottom (float 64)
    iPower - int, power of 10 for scaling the shooting solutions
    h_arr - mesh array 

    """

    # Parameters
    iPowerF = -50
    Roof = 1.0e50
    Floor = 1.0e-50

    # Loop over successive acoustic media starting at the end and going up
    for Medium in range(last_acoustic, first_acoustic-1, -1):
        hMedium = h_arr[Medium]
        if Medium == ind_arr.size - 1:
            z_layer = z_arr[ind_arr[Medium]:]
        else:
            z_layer = z_arr[ind_arr[Medium]:ind_arr[Medium+1]]
        NMedium = z_layer.size # includes interface points
        h2k2 = hMedium**2 * x

        ii = ind_arr[Medium] + NMedium -1
        rhoMedium = rho_arr[ind_arr[Medium]]  # Density is homogeneous using value at the top of each medium

        p1 = -2.0 * g
        p2 = (b1[ii] - h2k2) * g - 2.0 * hMedium * f * rhoMedium

        # Shoot (towards surface) through a single medium
        for ii in range(ind_arr[Medium] + NMedium-1, ind_arr[Medium], -1):
            p0 = p1
            p1 = p2
            p2 = (h2k2 - b1[ii]) * p1 - p0

            if CountModes:
                if p0 * p1 <= 0.0:
                    mode_count += 1

            if abs(p2) > Roof:  # Scale if necessary
                p0 *= Floor
                p1 *= Floor
                p2 *= Floor
                iPower -= iPowerF

        # Update f and g
        rhoMedium = rho_arr[ind_arr[Medium]]  # Density at the top of the layer
        f = -(p2 - p0) / (2.0 * hMedium * rhoMedium)
        g = -p1
    return f, g, iPower

@njit
def funct(x, args):
    """
    funct(x) = 0 is the dispersion relation
    (funct is the difference between the impedance for the solution obtained 
    by shooting upwards from the bottom boundary
    with eigenvalue x and the impedance for the solution 
    obtained by shooting downwards from the top boundary through
    any surface elastic layers to the first acoustic medium interface)

    ev_mat[i,j] is the jth mode for the ith mesh used in the richardson extrap.
    iset is the current index of the mesh to use in ev_mat

    """
    omega2, ev_mat, iset, \
    h_arr, ind_arr, z_arr, \
    cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, \
    b1, b1c, b2, b3, b4, rho_arr,  c_low, c_high,\
    elastic_flag, first_acoustic, last_acoustic, mode, CountModes, mode_count = args
    iPowerR = 50
    iPowerF = -50
    Roof = 1.0e50
    Floor = 1.0e-50


    mode_count = 0

    # shoot up from the bottom
    f_bott, g_bott, iPower, mode_count = get_bc_impedance(x, omega2, False, cp_bott, cs_bott, rho_bott, 
                                        h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                        first_acoustic, last_acoustic, mode_count)


    f, g, iPower = acoustic_layers(x, f_bott, g_bott, iPower, ind_arr, h_arr, z_arr, b1, rho_arr, 
                                    CountModes, mode_count, first_acoustic, last_acoustic)
    print('eig x, f, g, iPower after AcousticLayers = ', x, f, g, iPower)
    f_top, g_top, iPower_top, mode_count = get_bc_impedance(x, omega2, True, cp_top, cs_top, rho_top, 
                                        h_arr, ind_arr, z_arr, b1, b2, b3, b4, rho_arr, 
                                        first_acoustic, last_acoustic, mode_count)

    Delta  =(f * g_top - g * f_top).real
    iPower = iPower + iPower_top

    if ( g * Delta > 0.0 ): 
        mode_count = mode_count + 1

    # Deflate previous roots
    # NOTE: Modes are indexed from 1!
    # so the loop below deflates the previously found roots

    if ( ( mode > 1 ) and (len(ind_arr) > last_acoustic - first_acoustic + 1 ) ):
       for j in range(mode-1):
          Delta = Delta / ( x - ev_mat[iset, j] )

          # Scale if necessary
          while ( np.abs(Delta ) < Floor and np.abs( Delta ) > 0.0):
             Delta  = Roof * Delta
             iPower = iPower - iPowerR

          while (np.abs( Delta ) > Roof ):
             Delta  = Floor * Delta
             iPower = iPower - iPowerF
    return Delta, iPower, mode_count

def bisection(x_min, x_max, M, args):
    """
    Returns isolating intervals (xL, xR) for each eigenvalue
    in the given interval [x_min, x_max].
    """
    max_bisections = 50

    # Initialize boundaries
    x_l = x_min*np.ones(M+1)
    x_r = x_max*np.ones(M+1)

    # Compute the initial number of modes at x_max
    delta, i_power, mode_count = funct(x_max, args)
    n_zeros_initial = mode_count

    if M == 1:
        return x_l, x_r

    # Loop over eigenvalues to refine intervals
    for mode in range(1, M):
        if x_l[mode] == x_min:
            x2 = x_r[mode]
            x1 = max(np.max(x_l[mode + 1:]), x_min)

            for _ in range(max_bisections):
                x = x1 + (x2 - x1) / 2
                delta, i_power = mode_count_func(x)
                n_zeros = mode_count_func(x) - n_zeros_initial

                if n_zeros < mode:
                    x2 = x
                    x_r[mode] = x
                else:
                    x1 = x
                    if x_r[n_zeros + 1] >= x:
                        x_r[n_zeros + 1] = x
                    if x_l[n_zeros] <= x:
                        x_l[n_zeros] = x

                if x_l[mode] != x_min:
                    break

    return x_l, x_r

def solve1(args, h_v):
    """
    Solve for eigenvalues using Sturm sequences and Brent's method.

    Parameters:
        omega2 : float
            Frequency squared.
        c_high : float
            Upper limit of phase speed.
        c_low : float
            Lower limit of phase speed.
        nfreq : int
            Number of frequencies.
        file_root : str
            Root name for output files.
        prt_file : file-like
            Output file for printing.
        title : str
            Title for output.
        i_set, ifreq, iprof : int
            Indices for current dataset.
        mode_count_func : callable
            Function to count modes.
        erro_out_func : callable
            Function to handle errors.
        bisection_func : callable
            Function to initialize bounds for eigenvalue refinement.
        zbrentx_func : callable
            Function to refine eigenvalues.
        evmat, extrap, k, vg : ndarray, optional
            Arrays to hold results. If provided, will be updated.
        n : ndarray
            Array of mode counts per acoustic layer.
        first_acoustic, last_acoustic : int
            Indices for the acoustic layers.
    """

    omega2, ev_mat, iset, \
    h_arr, ind_arr, z_arr, \
    cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, \
    b1, b1c, b2, b3, b4, rho_arr,  c_low, c_high,\
    elastic_flag, first_acoustic, last_acoustic = args

    # Initialization
    # Determine the number of modes
    x_min = 1.00001 * omega2 / c_high ** 2
    m = 0
    count_modes = True
    margs = args + (1, count_modes, 0)
    delta, i_power, m = funct(x_min)
    print("Number of modes = {m}")

    # Allocate memory for xL and xR
    x_l = np.zeros(m + 1)
    x_r = np.zeros(m + 1)

    # Allocate memory for eigenvalue matrix if needed
    if i_set == 1:
        n_sets = evmat.shape[0] if evmat is not None else 0
        if evmat is not None:
            evmat = None
            extrap = None
            k = None
            vg = None
        evmat = np.zeros((n_sets, m))
        extrap = np.zeros((n_sets, m))
        k = np.zeros(m)
        vg = np.zeros(m)

    # Check upper bound
    x_max = omega2 / c_low ** 2
    delta, i_power = mode_count_func(x_max)
    m -= mode_count_func(x_max)

    if m == 0:
        l_record_length = 32
        nz_tab = 0

        if ifreq == 1 and iprof == 1:
            with open(f"{file_root}.mod", "wb") as mod_file:
                mod_file.write(np.array([l_record_length, title, nfreq, 1, nz_tab, nz_tab], dtype='int32').tobytes())
                mod_file.write(np.array([m], dtype='int32').tobytes())

        erro_out_func("KRAKEN", "No modes for given phase speed interval")
        return

    n_total = np.sum(n[first_acoustic:last_acoustic + 1])
    if m > n_total / 5:
        print(f"Approximate number of modes = {m}", file=prt_file)
        print("Warning in KRAKEN - Solve1 : Mesh too coarse to sample the modes adequately", file=prt_file)

    # Initialize bounds for eigenvalue refinement
    bisection_func(x_min, x_max, x_l, x_r)

    # Refine each eigenvalue
    count_modes = False

    for mode in range(1, m + 1):
        x1 = x_l[mode]
        x2 = x_r[mode]
        eps = abs(x2) * 10.0 ** (2.0 - np.finfo(float).precision)

        x, error_message = zbrentx_func(x1, x2, eps)

        if error_message:
            print(f"ISet, mode = {i_set}, {mode}", file=prt_file)
            print(f"Warning in KRAKEN-ZBRENTX : {error_message}", file=prt_file)

        evmat[i_set, mode - 1] = x

    # Clean up
    del x_l, x_r

@njit
def solve2(args, h_v):
    """
    h_v is array of mesh sizes
    """

    max_iteration = 2000
    

    omega2, ev_mat, iset, \
    h_arr, ind_arr, z_arr, \
    cp_top, cs_top, rho_top, cp_bott, cs_bott, rho_bott, \
    b1, b1c, b2, b3, b4, rho_arr,  c_low, c_high,\
    elastic_flag, first_acoustic, last_acoustic = args
    CountModes=False
    mode_count = 0 # doesn't matter

    M_max = ev_mat.shape[1]


    # inital guess
    x = omega2 / c_low**2
    print('x0', x)

    for mode in range(1, M_max+1):
        # Initial guess for x
        imode = mode-1
        x *= 1.00001
        
        # use extrapolation to produce initial guess if possible
        if iset >= 1:
            p = ev_mat[:iset, imode] # load previous mesh estimates

            if iset >= 2: # extrapolation
                for ii in range(iset-1):
                    for j in range(iset - ii-1):
                        x1 = h_v[j]**2
                        x2 = h_v[j + ii+1]**2

                        p[j] = (
                            ((h_v[iset - 1]**2 - x2) * p[j] - (h_v[iset - 1]**2 - x1) * p[j + 1]) /
                            (x1 - x2)
                        )
                x = p[0]

        # Calculate tolerance for root finder
        print('b1 size', b1.size)
        #tolerance = np.abs(x) * b1.size * 10.0**(1.0 - 15) # 15 is precision for float 64
        tolerance = np.abs(x) * b1.size * 10.0**(1.0 - np.finfo(np.float64).precision)


        # Use secant method to refine eigenvalue
        margs = args + (mode, CountModes, mode_count)

        print('tolerance', tolerance)
        print('x', x)
        
        x, iteration, error_message = root_finder_secant_real(x, tolerance, max_iteration, funct, margs)

        if error_message != '':
            print(f"Warning in Solve2 - RootFinderSecant: {error_message}")
            print(f"iset, mode = {iset}, {mode}")
            x = np.finfo(x).tiny

        ev_mat[iset, imode] = x

        # Discard modes outside user-specified spectrum
        if omega2 / c_high**2 > x:
            ev_mat = ev_mat[:, :imode]
            break

    return ev_mat

@njit
def root_finder_secant_real(x2, tolerance, max_iterations, func, args):
    """
    Secant method for finding roots of a real-valued function.
    Used for funct, so the function returns three things

    Parameters:
        x2 (float): Initial guess for the root (updated in-place).
        tolerance (float): Error bound for convergence.
        max_iterations (int): Maximum number of allowable iterations.
        func (callable): Function that computes the value of the function and an integer power.

    Returns:
        x2 (float): Estimated root.
        iteration (int): Number of iterations performed.
        error_message (str): Empty unless there was a failure to converge.
    """
    error_message = ""
    if tolerance <= 0.0:
        return x2, 0, "Non-positive tolerance specified"

    x1 = x2 + 10.0 * tolerance
    f1, i_power1, _ = func(x1, args)

    for iteration in range(1, max_iterations + 1):
        x0, f0, i_power0 = x1, f1, i_power1
        x1 = x2
        f1, i_power1, _ = func(x1, args)

        c_num = f1 * (x1 - x0)
        c_den = f1 - f0 * 10.0 ** (i_power0 - i_power1)

        if abs(c_num) >= abs(c_den * x1):
            shift = 0.1 * tolerance
        else:
            shift = c_num / c_den

        x2 = x1 - shift

        if abs(x2 - x1) + abs(x2 - x0) < tolerance:
            return x2, iteration, ""

    return x2, max_iterations, "Failure to converge in RootFinderSecant"

@njit
def root_finder_secant_complex(x2, tolerance, max_iterations, func, args):
    """
    Secant method for finding roots of a complex-valued function.

    Parameters:
        x2 (complex): Initial guess for the root (updated in-place).
        tolerance (float): Error bound for convergence.
        max_iterations (int): Maximum number of allowable iterations.
        func (callable): Function that computes the value of the function and an integer power.

    Returns:
        x2 (complex): Estimated root.
        iteration (int): Number of iterations performed.
        error_message (str): Empty unless there was a failure to converge.
    """
    error_message = ""

    if tolerance <= 0.0:
        return x2, 0, "Non-positive tolerance specified"

    x1 = x2 + 100.0 * tolerance
    f1, i_power1, _ = func(x1, args)

    for iteration in range(1, max_iterations + 1):
        x0, f0, i_power0 = x1, f1, i_power1
        x1 = x2
        f1, i_power1, _ = func(x1, args)

        c_num = f1 * (x1 - x0)
        c_den = f1 - f0 * 10.0 ** (i_power0 - i_power1)

        if abs(c_num) >= abs(c_den * x1):
            shift = 0.1 * tolerance
        else:
            shift = c_num / c_den

        x2 = x1 - shift

        if abs(x2 - x1) + abs(x2 - x0) < tolerance:
            return x2, iteration, ""

    return x2, max_iterations, "Failure to converge in RootFinderSecant"

