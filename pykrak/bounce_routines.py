import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from pykrak import attn_pert as ap
from pykrak import krak_routines as kr
import time

import numpy as np
IPOWER_R = 50
IPOWER_F = -50
ROOF = 1.0e50
FLOOR = 1.0e-50

@njit(cache=True)
def _elastic_propagate(
    x, yV, iPower, h, b1, b2, b3, b4, rho_arr, direction
):
    """
    Python equivalent of Fortran ElasticUP / ElasticDN.

    Parameters
    ----------
    x : float or complex
        Trial eigenvalue, k^2.
    yV : array_like, shape (5,)
        Initial compound-matrix state.
    iPower : int
        Dynamic-range exponent.
    h : float
        Thickness of this layer.
    b1, b2, b3, b4, rho_arr : array_like
        Layer samples, ordered from top to bottom.
    direction : {"up", "down"}
        Propagation direction.

    Returns
    -------
    yV : ndarray, shape (5,)
    iPower : int
    """
    b1 = np.asarray(b1)
    b2 = np.asarray(b2)
    b3 = np.asarray(b3)
    b4 = np.asarray(b4)
    rho_arr = np.asarray(rho_arr)

    n = b1.size
    if n < 2:
        raise ValueError("Each elastic layer needs at least two samples.")
    #if any(a.size != n for a in (b2, b3, b4, rho_arr)):
    #    raise ValueError("All layer coefficient arrays must have equal lengths.")

    # Preserve complex arithmetic, including complex trial eigenvalues.
    #dtype = np.result_type(
    #    np.asarray(yV).dtype, np.asarray(x).dtype,
    #    b1.dtype, b2.dtype, b3.dtype, b4.dtype, rho_arr.dtype,
    #    np.complex128,
    #)
    #yV = np.asarray(yV, dtype=dtype).copy()

    two_x = 2.0 * x
    two_h = 2.0 * h
    four_h_x = 4.0 * h * x

    if direction == "up":
        j = n - 1
        sign = -1.0
    elif direction == "down":
        j = 0
        sign = 1.0
    else:
        raise ValueError("direction must be 'up' or 'down'")

    xb3 = x * b3[j] - rho_arr[j]
    zV = np.empty(5, np.complex128)

    # Euler's method for the first half-step.
    zV[0] = yV[0] + sign * 0.5 * (
        b1[j] * yV[3] - b2[j] * yV[4]
    )
    zV[1] = yV[1] + sign * 0.5 * (
        -rho_arr[j] * yV[3] - xb3 * yV[4]
    )
    zV[2] = yV[2] + sign * 0.5 * (
        two_h * yV[3] + b4[j] * yV[4]
    )
    zV[3] = yV[3] + sign * 0.5 * (
        xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2]
    )
    zV[4] = yV[4] + sign * 0.5 * (
        rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2]
    )

    # Modified midpoint method.
    xV = np.empty(5, dtype=np.complex128)

    for ii in range(n - 1):
        j += -1 if direction == "up" else 1

        xV[:] = yV
        yV = zV.copy()

        xb3 = x * b3[j] - rho_arr[j]

        zV[0] = xV[0] + sign * (
            b1[j] * yV[3] - b2[j] * yV[4]
        )
        zV[1] = xV[1] + sign * (
            -rho_arr[j] * yV[3] - xb3 * yV[4]
        )
        zV[2] = xV[2] + sign * (
            two_h * yV[3] + b4[j] * yV[4]
        )
        zV[3] = xV[3] + sign * (
            xb3 * yV[0] + b2[j] * yV[1] - two_x * b4[j] * yV[2]
        )
        zV[4] = xV[4] + sign * (
            rho_arr[j] * yV[0] - b1[j] * yV[1] - four_h_x * yV[2]
        )

        # Fortran tests ABS(DBLE(zV(2))), not the complex modulus.
        # Do not scale at the terminal midpoint iteration.
        if ii != n - 2:
            z2_real = abs(zV[1].real)

            if z2_real < FLOOR:
                zV *= ROOF
                yV *= ROOF
                iPower -= IPOWER_R
            elif z2_real > ROOF:
                zV *= FLOOR
                yV *= FLOOR
                iPower -= IPOWER_F

    # Standard filter at the terminal point.
    yV = (xV + 2.0 * yV + zV) / 4.0

    return yV, iPower


@njit(cache=True)
def elastic_up(
    x, yV, iPower, h, b1, b2, b3, b4, rho_arr,
    Floor=FLOOR, Roof=ROOF, iPowerR=IPOWER_R, iPowerF=IPOWER_F,
):
    """Propagate upward through one elastic layer."""
    return _elastic_propagate(
        x, yV, iPower, h, b1, b2, b3, b4, rho_arr, "up"
    )

@njit(cache=True)
def elastic_down(
    x, yV, iPower, h, b1, b2, b3, b4, rho_arr,
    Floor=FLOOR, Roof=ROOF, iPowerR=IPOWER_R, iPowerF=IPOWER_F,
):
    """Propagate downward through one elastic layer."""
    return _elastic_propagate(
        x, yV, iPower, h, b1, b2, b3, b4, rho_arr, "down"
    )


@njit(cache=True)
def get_f_g(
    cp, cs, rho, x, omega2, mode_count, complex_flag
):
    """
    Python counterpart of the BC selection and impedance calculation
    in Fortran BCImpedance.

    Parameters
    ----------
    cp, cs, rho : scalar
        Half-space compressional speed, shear speed, and density.
    x : complex
        Squared horizontal wavenumber.
    omega2 : float or complex
        Angular frequency squared.
    mode_count : int
        Retained for API compatibility; BCImpedance does not update it
        in the V, R, or A branches.
    complex_flag : bool
        Retained for API compatibility. Fortran keeps complex values.
    Returns
    -------
    f, g, yV
        Complex impedance functions and the five-element state vector.
    """
    if rho == 0.0:  # Vacuum
        f = 1.0 + 0.0j
        g = 0.0
        yV = np.array([f, g, 0.0, 0.0, 0.0])
    elif rho == 1e10:  # Rigid
        f = 0.0
        g = 1.0+0.0j
        yV = np.array([f, g, 0.0, 0.0, 0.0])
    else:
        x = complex(x)
        omega2 = complex(omega2)
        cp = complex(cp)
        cs = complex(cs)
        rho = complex(rho)

        zero = 0.0 + 0.0j
        yV = np.zeros(5, dtype=np.complex128)

        if cs.real > 0.0:
            gammaS2 = x - omega2 / cs**2
            gammaP2 = x - omega2 / cp**2

            gammaS = complex(pekeris_root(gammaS2))
            gammaP = complex(pekeris_root(gammaP2))

            mu = rho * cs**2

            yV[0] = (gammaS * gammaP - x) / mu
            yV[1] = (
                (gammaS2 + x)**2
                - 4.0 * gammaS * gammaP * x
            ) * mu
            yV[2] = 2.0 * gammaS * gammaP - gammaS2 - x
            yV[3] = gammaP * (x - gammaS2)
            yV[4] = gammaS * (gammaS2 - x)

            f = omega2 * yV[3]
            g = yV[1]

        else:
            gammaP = complex(
                pekeris_root(x - omega2 / cp**2)
            )
            f = gammaP
            g = rho

            # The supplied Fortran does not assign yV in this branch.
            # Use placeholders only if downstream code requires a vector;
            # these are not a Fortran-defined physical state.
            yV[:] = 1e10 + 0.0j

        # Do not take real parts here: Fortran f and g are COMPLEX(KIND=8).
        return f, g, yV

@njit(cache=True)
def pekeris_root(z):
    z = complex(z)

    if z.real >= 0.0:
        return np.sqrt(z + 0j)
    else:
        return 1j * np.sqrt(-z + 0j)


@njit(cache=True)
def get_bc_impedance(
    x,
    omega2,
    top_flag,
    cp,
    cs,
    rho,
    h_arr,
    ind_arr,
    z_arr,
    b1,
    b2,
    b3,
    b4,
    rho_arr,
    first_acoustic,
    last_acoustic,
    mode_count,
    complex_flag,
):
    """
    Compute boundary-condition impedance and propagate through
    elastic layers, following BCImpedancecMod.

    Assumptions
    -----------
    * get_f_g() initializes the boundary state for the requested
      boundary condition.
    * ind_arr and z_arr use the indexing convention of the
      existing Python implementation.
    * Each layer's coefficient samples are contiguous.
    """
    iPower = 0

    # Retain complex x for complex boundary-condition calculations.
    if complex_flag:
        x = complex(x)
        omega2 = complex(omega2)

    f, g, Yv = get_f_g(
        cp, cs, rho, x, omega2, mode_count, complex_flag
    )

    Yv = np.asarray(Yv, dtype=np.complex128).copy()

    if top_flag:
        g = -g

    # Fortran: TOP -> Medium = 1, ..., FirstAcoustic - 1.
    # Python arrays are assumed to use zero-based medium indices.
    if top_flag and first_acoustic > 1:
        for medium in range(first_acoustic - 1):
            i0 = int(ind_arr[medium])

            if medium + 1 < len(ind_arr):
                i1 = int(ind_arr[medium + 1])
            else:
                i1 = i0 + len(z_arr) - i0

            Yv, iPower = elastic_down(
                x, Yv, iPower, h_arr[medium],
                b1[i0:i1], b2[i0:i1],
                b3[i0:i1], b4[i0:i1], rho_arr[i0:i1],
            )

        f = omega2 * Yv[3]
        g = Yv[1]

    # Fortran: BOT -> Medium = NMedia, ..., LastAcoustic + 1.
    elif not top_flag and last_acoustic != len(h_arr) - 1:
        if np.all(Yv == 1e10):
            raise ValueError(
                "Yv is not initialized; a rigid halfspace is required "
                "when shooting upward through elastic layers."
            )

        for medium in range(len(h_arr) - 1, last_acoustic, -1):
            i0 = int(ind_arr[medium])

            if medium + 1 < len(ind_arr):
                i1 = int(ind_arr[medium + 1])
            else:
                i1 = len(b1)

            Yv, iPower = elastic_up(
                x, Yv, iPower, h_arr[medium],
                b1[i0:i1], b2[i0:i1],
                b3[i0:i1], b4[i0:i1], rho_arr[i0:i1],
            )

        f = omega2 * Yv[3]
        g = Yv[1]

    if not complex_flag:
        f = np.real(f)
        g = np.real(g)

    return f, g, iPower, mode_count



@njit(cache=True)
def compute_reflection_coefficient(
    k_min,
    k_max,
    nk_tab,
    omega,
    rad_deg,
    omega2,
    cp_top,
    cs_top,
    rho_top,
    cp_bott,
    cs_bott,
    rho_bott,
    h_arr,
    ind_arr,
    z_arr,
    b1,
    b2,
    b3,
    b4,
    rho_arr,
    first_acoustic,
    last_acoustic,
    complex_flag=True,
    count_modes=False,
    mode_count=0,
    ipower_initial=0,
):
    """
    Compute the BOUNCE reflection coefficient over a wavenumber grid.

    Parameters
    ----------
    k_min, k_max : float
        Minimum and maximum horizontal wavenumbers.

    nk_tab : int
        Number of wavenumber samples. Must be at least 2.

    omega : float
        Angular frequency.

    rad_deg : float
        Radians-to-degrees conversion factor, normally 180 / pi.

    omega2 : float
        Squared angular frequency.

    cp_top, cs_top, rho_top : scalar
        Top half-space properties.

    cp_bott, cs_bott, rho_bott : scalar
        Bottom half-space properties.

    h_arr, ind_arr, z_arr : ndarray
        Mesh and indexing arrays.

    b1, b2, b3, b4, rho_arr : ndarray
        Arrays returned or prepared by initialize().

    first_acoustic, last_acoustic : int
        Zero-based acoustic-medium indices.

    complex_flag : bool
        Passed to get_bc_impedance. For BOUNCE this will generally be True.

    count_modes : bool
        Passed to acoustic_layers.

    mode_count : int
        Initial mode count.

    ipower_initial : int
        Initial scaling exponent.

    Returns
    -------
    result : dict
        Dictionary containing the computed wavenumber, impedance,
        angle, reflection magnitude, phase, and complex reflection
        coefficient arrays.
    """

    if nk_tab < 2:
        raise ValueError("nk_tab must be at least 2")

    # Equivalent to:
    #
    # Deltak = (kMax - kMin) / (NkTab - 1)
    delta_k = (k_max - k_min) / (nk_tab - 1)

    # Storage corresponding to xTab, fTab, gTab, and ITab.
    x_tab = np.zeros(nk_tab, dtype=np.float64)
    f_tab = np.zeros(nk_tab, dtype=np.complex128)
    g_tab = np.zeros(nk_tab, dtype=np.complex128)
    i_tab = np.zeros(nk_tab, dtype=np.int64)

    # This stores the mode count after each wavenumber calculation.
    mode_count_tab = np.zeros(nk_tab, dtype=np.int64)

    # Equivalent to the Fortran Wavenumber loop.
    for ik in range(nk_tab):
        k1 = k_min + ik * delta_k

        # In BOUNCE, x is complex.
        x = k1**2

        # Reset the per-wavenumber scaling exponent and mode count.
        i_power = ipower_initial
        current_mode_count = mode_count

        # Bottom impedance.
        f_bott, g_bott, i_power, current_mode_count = (
            get_bc_impedance(
                x,
                omega2,
                False,
                cp_bott,
                cs_bott,
                rho_bott,
                h_arr,
                ind_arr,
                z_arr,
                b1,
                b2,
                b3,
                b4,
                rho_arr,
                first_acoustic,
                last_acoustic,
                current_mode_count,
                complex_flag,
            )
        )

        # Propagate through the acoustic layers.
        f, g, i_power, current_mode_count = kr.acoustic_layers_no_count(
            x,
            f_bott,
            g_bott,
            i_power,
            ind_arr,
            h_arr,
            z_arr,
            b1,
            rho_arr,
            False,
            current_mode_count,
            first_acoustic,
            last_acoustic,
        )

        # Store the quantities produced by the bottom-to-top calculation.
        x_tab[ik] = np.real(x)
        f_tab[ik] = f
        g_tab[ik] = g
        i_tab[ik] = i_power
        mode_count_tab[ik] = current_mode_count

    # Select the reference sound speed.
    #
    # Fortran:
    # IF (HSTop%BC(1:1) == 'A') THEN
    #     c0 = DBLE(HSTop%cP)
    # ELSE
    #     c0 = 1500
    # END IF
    #
    # Here this assumes that the caller supplies a separate indication
    # of whether the top boundary is a half-space. The density sentinel
    # convention follows your existing Python code.
    top_is_halfspace = (
        rho_top != 0.0 and rho_top != 1.0e10
    )

    if top_is_halfspace:
        c0 = float(np.real(cp_top))
    else:
        c0 = 1500.0

    if c0 <= 0.0:
        raise ValueError("Reference sound speed c0 must be positive")
    print('c0', c0)
    # Free-space wavenumber.
    k0 = omega / c0

    # Horizontal wavenumber.
    kx = np.sqrt(x_tab)

    # Allocate output arrays.
    kz = np.zeros(nk_tab, dtype=np.float64)
    theta = np.zeros(nk_tab, dtype=np.float64)
    reflection_coefficient = np.ones(
        nk_tab,
        dtype=np.complex128,
    )
    reflection_magnitude = np.ones(nk_tab, dtype=np.float64)
    phase = np.full(nk_tab, 180.0, dtype=np.float64)

    # Equivalent to:
    #
    # WHERE (k0 > kx)
    #     ...
    # ELSEWHERE
    #     ...
    # END WHERE
    propagating = k0 > kx

    kz[propagating] = np.sqrt(
        k0**2 - kx[propagating]**2
    )

    theta[propagating] = rad_deg * np.arctan2(
        kz[propagating],
        kx[propagating],
    )


    numerator = (
        f_tab[propagating]
        - 1j * kz[propagating] * g_tab[propagating]
    )

    denominator = (
        f_tab[propagating]
        + 1j * kz[propagating] * g_tab[propagating]
    )

    reflection_coefficient[propagating] = (
        -numerator / denominator
    )

    reflection_magnitude[propagating] = np.abs(
        reflection_coefficient[propagating]
    )

    phase[propagating] = rad_deg * np.arctan2(
        np.imag(reflection_coefficient[propagating]),
        np.real(reflection_coefficient[propagating]),
    )

    # The evanescent-wave values already have the desired values:
    #
    # kz = 0
    # theta = 0
    # RCmplx = 1
    # R = 1
    # phase = 180

    # Phase unwrapping.
    #
    # The first Fortran condition is logically impossible:
    #
    # AIMAG(R1) > 0 .AND. AIMAG(R1) < 0
    #
    # Consequently, this implementation retains only the meaningful
    # second crossing condition.
    loops = 0

    for itheta in range(nk_tab - 2, -1, -1):
        r1 = reflection_coefficient[itheta]
        r2 = reflection_coefficient[itheta + 1]

        if (
            np.imag(r1) < 0.0
            and np.imag(r2) > 0.0
            and np.real(r2) < 0.0
        ):
            loops -= 1

        phase[itheta] -= loops * 360.0

    return x_tab, f_tab, g_tab, i_tab, kx, kz, theta, reflection_coefficient, reflection_magnitude, phase, c0, delta_k


def list_input_bounce(
    freq,
    nk_tab,
    z_list,
    cp_list,
    cs_list,
    rho_list,
    attnp_list,
    attns_list,
    cp_top,
    cs_top,
    rho_top,
    attnp_top,
    attns_top,
    cp_bott,
    cs_bott,
    rho_bott,
    attnp_bott,
    attns_bott,
    attn_units,
    Ng_list,
    rmax,
    c_low,
    c_high,
    interp_type
    ):
    omega = 2*np.pi*freq
    omega2 = (2 * np.pi * freq) ** 2
    Ng_arr0 = np.array(Ng_list, dtype=np.int32)

    h_arr, ind_arr, z_arr, cp_arr, cs_arr, rho_arr = kr.mesh_list_inputs(
        z_list,
        cp_list,
        cs_list,
        rho_list,
        attnp_list,
        attns_list,
        Ng_arr0,
        attn_units,
        omega,
        interp_type
    )


    (
        b1,
        b1c,
        b2,
        b3,
        b4,
        rho_arr,
        c_low,
        _, # don't overwrite high speed
        elastic_flag,
        first_acoustic,
        last_acoustic,
    ) = kr.initialize(
        h_arr,
        ind_arr,
        z_arr,
        omega2,
        cp_arr,
        cs_arr,
        rho_arr,
        cp_top,
        cs_top,
        rho_top,
        cp_bott,
        cs_bott,
        rho_bott,
        c_low,
        c_high,
    )

    k_min = omega / c_high
    k_max = omega / c_low


    x_tab, f_tab, g_tab, i_tab, kx, kz, theta, reflection_coefficient, reflection_magnitude, phase, c0, delta_k = compute_reflection_coefficient(
        k_min=k_min,
        k_max=k_max,
        nk_tab=nk_tab,
        omega=2*np.pi*freq,
        rad_deg=180.0 / np.pi,
        omega2=omega2,
        cp_top=cp_top,
        cs_top=cs_top,
        rho_top=rho_top,
        cp_bott=cp_bott,
        cs_bott=cs_bott,
        rho_bott=rho_bott,
        h_arr=h_arr,
        ind_arr=ind_arr,
        z_arr=z_arr,
        b1=b1,
        b2=b2,
        b3=b3,
        b4=b4,
        rho_arr=rho_arr,
        first_acoustic=first_acoustic,
        last_acoustic=last_acoustic,
        complex_flag=True,
        count_modes=False,
        mode_count=0,
        ipower_initial=0,
    )




def test():
    z_list = [np.array([0.0, 10.0]), np.array([10.0, 20.0])]
    cp_list = [np.array([1550.0, 1550.0]), np.array([1600.0, 1600.0])]
    cs_list = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    attnp_list = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    attns_list = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    rho_list = [np.array([1.5, 1.5]), np.array([2.0, 2.0])]
    cp_top = 1480.0
    cs_top = 0.0
    rho_top = 1.0
    attnp_top = 0.0
    attns_top = 0.0

    cp_bott = 2000.0
    cs_bott = 0.0
    rho_bott = 2.5
    attnp_bott = 0.00
    attns_bott = 0.00

    attn_units = "dbplam"
    rmax = 0.0
    freq = 1000.0
    c_low = 1400.0
    c_high = 1e9
    nk_tab = 901
    lam = 1500.0 / freq
    dz = lam / 20
    Ng_list = [int((z_list[i][1] - z_list[i][0]) / dz) +  1 for i in range(len(z_list))]

    theta_crit = 180/np.pi*np.arccos(cp_top / cp_list[0])
    print('theta_crit', theta_crit)

    for freq in [1000, 1001, 1002, 1003, 1004]:
        now =time.time()
        list_input_bounce(
            freq,
            nk_tab,
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            cp_top,
            cs_top,
            rho_top,
            attnp_top,
            attns_top,
            cp_bott,
            cs_bott,
            rho_bott,
            attnp_bott,
            attns_bott,
            attn_units,
            Ng_list,
            rmax,
            c_low,
            c_high,
            'lin',
            )
        print('runtime', time.time() - now)
test()

