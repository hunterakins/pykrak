"""
Description:
    Interpolation routines numba-ized

Date:
    7/13

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from numba import jit_module


def find_indices(x,y, xgrid, ygrid):
    """
    Given floats x and y
    and MONOTONICALLY INCREASING grids xgrid and ygrid, 
    find the indices that bracket x and y
    """
    x_l_ind, x_u_ind = get_klo_khi(x, xgrid)
    y_l_ind, y_u_ind = get_klo_khi(y, ygrid)

    dx = xgrid[x_u_ind] - xgrid[x_l_ind]
    dy = ygrid[y_u_ind] - ygrid[y_l_ind]
    return x_l_ind, x_u_ind, y_l_ind, y_u_ind, dx, dy

def get_klo_khi(x, xgrid):
    """ 
    Nonuniformly spaced xgrid
    but sorted monitonically from small to big
    Get the bracketing indices
    """

    n = xgrid.size
    klo = 0 
    khi = n-1 
    while khi-klo > 1:
        k = (khi+klo) >> 1
        if xgrid[k] > x:
            khi = k 
        else: klo = k 
    return klo, khi 

def get_bilin_interp(z, r, zgrid, rgrid, cgrid):
    """
    Given a depth z and range r
    and SSP sampled at grid points in zgrid and rgrid
    Bi-linearly interpolate to find SSP at z,r
    """
    # find the indices of the grid points that are closest to z, r
    z_l_ind, z_u_ind, r_l_ind, r_u_ind, dz, dr = find_indices(z, r, zgrid, rgrid)

    # find the fractional distance between the grid points
    z_l_frac = (zgrid[z_u_ind] - z) / dz
    z_u_frac = (z - zgrid[z_l_ind]) / dz
    r_l_frac = (rgrid[r_u_ind] - r) / dr
    r_u_frac = (r - rgrid[r_l_ind]) / dr

    # get interpolation weights
    w11 = z_l_frac * r_l_frac
    w12 = z_l_frac * r_u_frac
    w21 = z_u_frac * r_l_frac
    w22 = z_u_frac * r_u_frac

    # interpolate c, dcdz, dcdr
    c = w11 * cgrid[z_l_ind, r_l_ind] + w12 * cgrid[z_l_ind, r_u_ind] + w21 * cgrid[z_u_ind, r_l_ind] + w22 * cgrid[z_u_ind, r_u_ind]
    return c

def get_bilin_interp_grads(z, r, zgrid, rgrid, cgrid, dcdz_grid, dcdr_grid):
    """
    Given a depth z and range r
    and SSP sampled at grid points in zgrid and rgrid
    Bi-linearly interpolate to find SSP at z,r
    """
    # find the indices of the grid points that are closest to z, r
    z_l_ind, z_u_ind, r_l_ind, r_u_ind, dz, dr = find_indices(z, r, zgrid, rgrid)

    # find the fractional distance between the grid points
    z_l_frac = (zgrid[z_u_ind] - z) / dz
    z_u_frac = (z - zgrid[z_l_ind]) / dz
    r_l_frac = (rgrid[r_u_ind] - r) / dr
    r_u_frac = (r - rgrid[r_l_ind]) / dr

    # get interpolation weights
    w11 = z_l_frac * r_l_frac
    w12 = z_l_frac * r_u_frac
    w21 = z_u_frac * r_l_frac
    w22 = z_u_frac * r_u_frac

    # interpolate c, dcdz, dcdr
    c = w11 * cgrid[z_l_ind, r_l_ind] + w12 * cgrid[z_l_ind, r_u_ind] + w21 * cgrid[z_u_ind, r_l_ind] + w22 * cgrid[z_u_ind, r_u_ind]
    dcdz = w11 * dcdz_grid[z_l_ind, r_l_ind] + w12 * dcdz_grid[z_l_ind, r_u_ind] + w21 * dcdz_grid[z_u_ind, r_l_ind] + w22 * dcdz_grid[z_u_ind, r_u_ind]
    dcdr = w11 * dcdr_grid[z_l_ind, r_l_ind] + w12 * dcdr_grid[z_l_ind, r_u_ind] + w21 * dcdr_grid[z_u_ind, r_l_ind] + w22 * dcdr_grid[z_u_ind, r_u_ind]
    return c, dcdz, dcdr

def get_bicu_coeff(cpts, dcdzpts, dcdrpts, d2cdzdrpts, dz, dr):
    """ Get coefficients for bicubic interpolation as in Numerical Recipes
    (Press 1992, pg. 126)
    Input
    cpts - np 1d array
        values of the sound speed on the four corners that surround
        the point
    dcdzpts - np 1d array
        values of the partial derivative of sound speed with respect to z
        on the four corners that surround the point
    dcdrpts - np 1d array
        values of the partial derivative of sound speed with respect to r
        on the four corners that surround the point
    d2cdzdrpts - np 1d array
        values of the partial derivative of sound speed with respect to z and r
        on the four corners that surround the point
    dz - float
        grid spacing in z
    dr - float
        grid spacing in r
    """
    w = np.zeros((16, 16))
    w[0,:] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    w[1,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    w[2,:] = np.array([-3,0, 0, 3, 0, 0, 0, 0,-2, 0, 0, -1, 0, 0, 0, 0])
    w[3,:] = np.array([2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    w[4,:] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    w[5,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    w[6,:] = np.array([0, 0, 0, 0,-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1])
    w[7,:] = np.array([0, 0, 0, 0, 2, 0, 0,-2, 0, 0, 0, 0, 1, 0, 0, 1])
    w[8,:] = np.array([-3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    w[9,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0])
    w[10,:] = np.array([9,-9, 9,-9, 6, 3,-3,-6, 6,-6,-3, 3, 4, 2, 1, 2])
    w[11,:] = np.array([-6, 6,-6, 6,-4,-2, 2, 4,-3, 3, 3,-3,-2,-1,-1,-2])
    w[12,:] = np.array([2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    w[13,:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0])
    w[14,:] = np.array([-6, 6,-6, 6,-3,-3, 3, 3,-4, 4, 2,-2,-2,-2,-1,-1])
    w[15,:] = np.array([4,-4, 4,-4, 2, 2,-2,-2, 2,-2,-2, 2, 1, 1, 1, 1])


    x = np.zeros(16)
    dzdr = dz*dr
    for i in range(4):
        x[i] = cpts[i]
        x[i+4] = dcdzpts[i]*dz
        x[i+8] = dcdrpts[i]*dr
        x[i+12] = d2cdzdrpts[i]*dzdr

    c_flat = np.zeros(16) # matrix multiplication wx
    for i in range(16):
        xx = 0.
        for j in range(16):
            xx += w[i,j]*x[j]
        c_flat[i] = xx


    coeffs = np.zeros((4,4))
    l = 0
    for i in range(4):
        for j in range(4):
            coeffs[i,j] = c_flat[l]
            l += 1
    return coeffs

def get_bicubic_interp(z, r, zgrid, rgrid, cgrid, dcdz_grid, dcdr_grid, d2cdzdr_grid, coeff_arr):
    """
    Implement bicubic interpolation of SSP and derivatives
    """

    """
    find corners of the grid cell that contains the point z, r (potentially on an edge)
    """
    z_l_ind, z_u_ind, r_l_ind, r_u_ind, dz, dr = find_indices(z, r, zgrid, rgrid)
    zl,zu = zgrid[z_l_ind], zgrid[z_u_ind]
    rl,ru = rgrid[r_l_ind], rgrid[r_u_ind]

    t = (z - zl) / dz
    u = (r - rl) / dr

    coeffs = coeff_arr[z_l_ind, r_l_ind,...]

    """
    Load values on grid corners into arrays
    """
    #c1 = cgrid[z_l_ind, r_l_ind]
    #c2 = cgrid[z_u_ind, r_l_ind]
    #c3 = cgrid[z_u_ind, r_u_ind]
    #c4 = cgrid[z_l_ind, r_u_ind]
    #carr = np.array([c1, c2, c3, c4])

    #dcdz1 = dcdz_grid[z_l_ind, r_l_ind]
    #dcdz2 = dcdz_grid[z_u_ind, r_l_ind]
    #dcdz3 = dcdz_grid[z_u_ind, r_u_ind]
    #dcdz4 = dcdz_grid[z_l_ind, r_u_ind]
    #dcdzarr = np.array([dcdz1, dcdz2, dcdz3, dcdz4])

    #dcdr1 = dcdr_grid[z_l_ind, r_l_ind]
    #dcdr2 = dcdr_grid[z_u_ind, r_l_ind]
    #dcdr3 = dcdr_grid[z_u_ind, r_u_ind]
    #dcdr4 = dcdr_grid[z_l_ind, r_u_ind]
    #dcdrarr = np.array([dcdr1, dcdr2, dcdr3, dcdr4])

    #d2cdzdr1 = d2cdzdr_grid[z_l_ind, r_l_ind]
    #d2cdzdr2 = d2cdzdr_grid[z_u_ind, r_l_ind]
    #d2cdzdr3 = d2cdzdr_grid[z_u_ind, r_u_ind]
    #d2cdzdr4 = d2cdzdr_grid[z_l_ind, r_u_ind]
    #d2cdzdrarr = np.array([d2cdzdr1, d2cdzdr2, d2cdzdr3, d2cdzdr4])


    """ 
    Get the coefficients 
    """
    #coeffs = get_bicu_coeff(carr, dcdzarr, dcdrarr, d2cdzdrarr, dz, dr)


    """

    Explicit calculation for check
    c_val, dcdz_val, dcdr_val = 0., 0., 0. # return values
    for i in range(4):
        for j in range(4):
            c_val += coeffs[i,j]* t**i * u**j
            if i > 0:
                dcdz_val += i*coeffs[i,j]* t**(i-1) * u**j
            if j > 0 :
                dcdr_val += j*coeffs[i,j]* t**i * u**(j-1)
    dcdz_val /= dz
    dcdr_val /= dr
    """

    c_val, dcdz_val, dcdr_val = 0.0, 0.0, 0.0

    for i in range(3, -1, -1):
        c_val = t*c_val + ((coeffs[i][3]*u + coeffs[i,2])*u + coeffs[i][1])*u + coeffs[i][0]
        dcdr_val = t*dcdr_val + (3.0*coeffs[i][3]*u + 2.0*coeffs[i][2])*u + coeffs[i][1]
        dcdz_val = u*dcdz_val + (3.0*coeffs[3][i]*t + 2.0*coeffs[2][i])*t + coeffs[1][i]
    dcdz_val /= dz
    dcdr_val /= dr
    return c_val, dcdz_val, dcdr_val

def get_spline(x, y, yp1, ypn):
    """
    Given grid of points x for funytion evals y
    and gradient values yp1 and ypn at the first and last grid points,

    It returns the second derivatives (partial y partial x)
    at the grid depth points 
    Input 
    x - np 1d array 
        array of grid points 
    y - np 1d array
        array of tabulated values 
    yp1 - np 1d array
        array of derivative values at first depth
    ypn - np 1d array
        array of derivative values at final depth

    Output 
    y2 - np 1d array
        the second partial derivative of y with respect to x
    """
    n = y.shape[0]
    u = np.zeros(n, dtype=y.dtype)
    y2 = np.zeros(n, dtype=y.dtype)

    if yp1 > .99*1e30: # this denotes that the lower b.c. is ``natural'' (y''(0) = 0)
        y2[0] = 0.
        u[0] = 0.
    else:
        y2[0] = -0.5
        u[0] = (3./(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1)
       
    for i in range(1,n-1):
        sig = (x[i]-x[i-1])/(x[i+1]-x[i-1])
        p = sig*y2[i-1]+2.
        y2[i] = (sig-1.)/p
        u[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
        u[i] = (6.*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p
    if ypn > .99*1e30:
        qn = 0.
        un = 0.
    else:
        qn = 0.5
        un = (3./(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
    y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2]+1.)

    for k in range(n-2, -1, -1):
        y2[k] = y2[k]*y2[k+1]+u[k]
    return y2

def splint(x, xgrid, ygrid, y2grid):
    """
    SPLine INTegration
    Nonunifromly spaced xgrid
    but sorted monitonically from small to big
    values ygrid and y2grid (from spline)
    Get the value of y and dydx at depth x
    """
    n = xgrid.size
    klo, khi = get_klo_khi(x, xgrid)

    # now klo and khi bracket the input value of x
    dx = xgrid[khi] - xgrid[klo]
    if dx == 0:
        raise ValueError('bad input to nonuniform_splint')
    a = (xgrid[khi]-x)/dx
    b = (x-xgrid[klo])/dx
    y = a*ygrid[klo] + b*ygrid[khi] + ((a**3-a)*y2grid[klo] + (b**3-b)*y2grid[khi])*(dx**2)/6.
    dydx = -ygrid[klo] / dx + ygrid[khi] / dx -1/6*(3*a**2 - 1)*y2grid[klo]*dx + 1/6*(3*b**2 - 1)*y2grid[khi]*dx
    return y, dydx

def vec_splint(xpts, xgrid, ygrid, y2grid):
    ypts, dydx_pts = np.zeros(xpts.size, ygrid.dtype), np.zeros(xpts.size, dtype=ygrid.dtype)
    for i in range(xpts.size):
        ypts[i], dydx_pts[i] = splint(xpts[i], xgrid, ygrid, y2grid)
    return ypts, dydx_pts

def lin_int(x, xgrid, ygrid):
    """ 
    Linear interpolation
    """
    klo, khi = get_klo_khi(x, xgrid)
    alpha = (x-xgrid[klo])/(xgrid[khi]-xgrid[klo])
    return (1-alpha)*ygrid[klo] + alpha*ygrid[khi]

def vec_lin_int(xpts, xgrid, ygrid):
    """
    Linear interpolation for a vector of inputs
    """
    yout = np.zeros(xpts.size, dtype=ygrid.dtype)
    for i in range(xpts.size):
        yout[i] = lin_int(xpts[i], xgrid, ygrid)
    return yout

def get_depth_spline(z, c, cp1, cpn):
    """
    Given grid of depths z for which ssp is evaluated c
    and depth graident values cp1 and cpn at the first and last grid points,
    c is assumed to be given as a range-dependent profile, with the second axis denoting range
    Thus, this simply gives the depth splines

    It returns the second derivatives (partial c partial z)
    at the grid depth points 
    Input 
    z - np 1d array 
        array of grid points (first axis of c)
    c - np 2d array
        array of tabulated values (second axis is range)
    cp1 - np 1d array
        array of derivative values at first depth
    cpn - np 1d array

    Output 
    c2 - np 2d array
        first axis is depth, second is range
        gives the second partial derivative of c with respect to z
        at each tabulated range
    """
    nz = c.shape[0]
    nr = c.shape[1]
    u = np.zeros((nz,nr))
    c2 = np.zeros((nz,nr))

    if np.all(cp1 > .99*1e30): # this denotes that the lower b.c. is ``natural'' (y''(0) = 0)
        c2[0,:] = 0.
        u[0,:] = 0.
    else:
        c2[0,:] = -0.5
        u[0,:] = (3./(z[1]-z[0]))*((c[1,:]-c[0,:])/(z[1]-z[0])-cp1)
       
    for i in range(1,nz-1):
        sig = (z[i]-z[i-1])/(z[i+1]-z[i-1])
        p = sig*c2[i-1,:]+2.
        c2[i,:] = (sig-1.)/p
        u[i,:] = (c[i+1,:]-c[i,:])/(z[i+1]-z[i]) - (c[i,:]-c[i-1,:])/(z[i]-z[i-1])
        u[i,:] = (6.*u[i,:]/(z[i+1]-z[i-1])-sig*u[i-1,:])/p
    if np.all(cpn > .99*1e30):
        qn = 0.
        un = np.zeros(nr)
    else:
        qn = 0.5
        un = (3./(z[nz-1]-z[nz-2]))*(cpn-(c[nz-1,:]-c[nz-2,:])/(z[nz-1]-z[nz-2]))
    c2[nz-1,:] = (un-qn*u[nz-2,:])/(qn*c2[nz-2,:]+1.)

    for k in range(nz-2, -1, -1):
        c2[k,:] = c2[k,:]*c2[k+1,:]+u[k,:]
    return c2

def splint_lin_interp(z, r, zgrid, rgrid, cgrid, c2grid):
    """
    Does the spline interpolation on axis 1 and linear interpolation on axis 2
    Given UNIFORMLY SPACED zgrid with corresponding
    range-dependent values cgrid and second (DEPTH) derivatives 
    c2grid
    And a point (z,r) at which you would like to know the sound speed c,
    and the sound speed gradient (partial c partial z, partial c partial r)
    Return those numbers...

    To do so, it uses the spline information to get c and dcdz at the depth
    z and two bracketing ranges rl <= r <= ru
    Then it does a linear interpolation in range ...
    """

    # get grid corners
    z_l_ind, z_u_ind, r_l_ind, r_u_ind, dz, dr = find_indices(z, r, zgrid, rgrid)
    # use range inds to get c2grid values at bracketing ranges
    spl1 = c2grid[:,r_l_ind]
    spl2 = c2grid[:,r_u_ind]
    # and cgrid values
    c1 = cgrid[:,r_l_ind]
    c2 = cgrid[:,r_u_ind]

    """ 
    use splint to get the value of c and dcdz at depth z and the bracketing ranges
    """
    cl, dcdzl = splint(z, zgrid, c1, spl1)
    cu, dcdzu = splint(z, zgrid, c2, spl2)

    
    """
    Linearly interpolate these in range
    """
    w1 = (rgrid[r_u_ind] - r)/dr
    w2 = (r - rgrid[r_l_ind])/dr
    c = w1*cl + w2*cu
    dcdz = w1*dcdzl + w2*dcdzu
    
    return c, dcdz

def vec_splint_lin_interp(zpts, r, zgrid, rgrid, cgrid, c2grid):
    cpts, dcdzpts = np.zeros(zpts.size), np.zeros(zpts.size)
    for i in range(zpts.size):
        z = zpts[i]
        ci, dcdzi = splint_lin_interp(z, r, zgrid, rgrid, cgrid, c2grid)
        cpts[i] = ci
        dcdzpts[i] = dcdzi
    return cpts, dcdzpts

def splint_arr(z_arr, zgrid, cgrid, c2grid):
    """
    SPLine INTegration
    Nonunifromly spaced zgrid
    but sorted monitonically from small to big
    values cgrid and c2grid (from spline)
    Get the value of c and dcdz at depth z
    """
    n = zgrid.size
    n_arr = z_arr.size
    c_arr, dcdz_arr = np.zeros(n_arr), np.zeros(n_arr)

    for z_i in range(n_arr):
        z = z_arr[z_i]
        klo = 0
        khi = n-1
        while khi-klo > 1:
            k = (khi+klo) >> 1
            if zgrid[k] > z:
                khi = k
            else: klo = k

        # now klo and khi bracket the input value of z
        dz = zgrid[khi] - zgrid[klo]
        if dz == 0:
            raise ValueError('bad input to nonuniform_splint')
        a = (zgrid[khi]-z)/dz
        b = (z-zgrid[klo])/dz
        c = a*cgrid[klo] + b*cgrid[khi] + ((a**3-a)*c2grid[klo] + (b**3-b)*c2grid[khi])*(dz**2)/6.
        dcdz = -cgrid[klo] / dz + cgrid[khi] / dz -1/6*(3*a**2 - 1)*c2grid[klo]*dz + 1/6*(3*b**2 - 1)*c2grid[khi]*dz
        c_arr[z_i] = c
        dcdz_arr[z_i] = dcdz
    return c_arr, dcdz_arr

def find_range_indices(r, rgrid):
    """
    Given float r
    and REGULAR grid rgrid, 
    find the indices that bracket r
    """
    dr = rgrid[1]-rgrid[0]

    r_l_ind = int((r-rgrid[0])/dr) # index of r to the left
    r_u_ind = r_l_ind + 1 # index of r to the right

    # handle end case 
    if r == rgrid[-1]:
        r_l_ind = len(rgrid)-2
        r_u_ind = len(rgrid)-1

    if (r < rgrid[r_l_ind]) or (r > rgrid[r_u_ind]):
        print('Failed to find bracketing interval in r')
        print('r, rgrid[r_l_ind], rgrid[r_u_ind]')
        print(r, rgrid[r_l_ind], rgrid[r_u_ind])
    return r_l_ind, r_u_ind, dr

def get_range_lin_interp(r, rgrid, c_zr):
    """
    r is desired range
    rgrid is all ranges c_zr is defined (second axis)
    c_zr is 2d array of depth and range profiles
    """
    r_l_ind, r_u_ind, dr = find_range_indices(r, rgrid)

    # find the fractional distance between the grid points
    r_l_frac = (rgrid[r_u_ind] - r) / dr
    r_u_frac = (r - rgrid[r_l_ind]) / dr

    # get interpolation weights
    w1 = r_l_frac
    w2 = r_u_frac

    # interpolate 
    c = w1 * c_zr[:, r_l_ind] +  w2*c_zr[:,r_u_ind]
    return c

jit_module(nopython=True,cache=True)
