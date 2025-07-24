"""
Description:
    Check interpolation against numpy/scipy

Date:
    10/29/2023


Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from pykrak import interp
from scipy.interpolate import CubicSpline

def spline_test1():
    """
    Compare spline from this library and scipy 
    for sin function
    """
    f1 = lambda x: np.sin(x)
    x1 = np.linspace(0, 10, 10)
    x_interp = np.linspace(0, 10, 100)
    y1 = f1(x1)

    d2ydx2 = interp.get_spline(x1, y1, 1e30, 1e30)
    py_spline = CubicSpline(x1, y1, bc_type='natural')
    py_out = py_spline(x_interp)
    y, dydx = interp.vec_splint(x_interp, x1, y1, d2ydx2)

    plt.figure()
    plt.plot(x_interp, y, 'bo', label='this implementation')
    plt.plot(x_interp, py_out, 'ro', label='Scipy implementation')
    plt.plot(x1, y1, 'ko', label='data')

    plt.figure()
    plt.suptitle('Difference between python cubic spline and this implementation')
    plt.plot(x_interp, y-py_out, 'k')
    plt.grid()
    plt.show()

def spline_test2():
    """
    See performance on complex data
    """
    f1 = lambda x: np.exp(1j*x)
    x1 = np.linspace(0, 10, 10)
    x_interp = np.linspace(0, 10, 100)
    y1 = f1(x1)

    d2ydx2 = interp.get_spline(x1, y1, 1e30, 1e30)
    py_spline = CubicSpline(x1, y1, bc_type='natural')
    py_out = py_spline(x_interp)
    y, dydx = interp.vec_splint(x_interp, x1, y1, d2ydx2)

    fig, axes = plt.subplots(2,1)
    axes[0].plot(x_interp, np.abs(y), 'bo', label='this implementation')
    axes[0].plot(x_interp, np.abs(py_out), 'ro', label='Scipy implementation')
    axes[1].plot(x_interp, np.angle(y), 'bo', label='this implementation')
    axes[1].plot(x_interp, np.angle(py_out), 'ro', label='Scipy implementation')
    axes[0].plot(x1, np.abs(y1), 'ko', label='data')
    axes[1].plot(x1, np.angle(y1), 'ko', label='data')

    plt.figure()
    plt.suptitle('Difference between python cubic spline and this implementation')
    plt.plot(x_interp, np.abs(y-py_out), 'k')
    plt.grid()
    plt.show()

def lin_int_test1():
    """
    Compare spline from this library and scipy 
    """
    f1 = lambda x: np.sin(x)
    x1 = np.linspace(0, 10, 20)
    x_interp = np.linspace(0, 10, 100)
    y1 = f1(x1)

    y_interp = interp.vec_lin_int(x_interp, x1, y1)
    py_interp = np.interp(x_interp, x1, y1)

    plt.figure()
    plt.plot(x_interp, y_interp, 'bo', label='this implementation')
    plt.plot(x_interp, py_interp, 'ro', label='Scipy implementation')
    plt.plot(x1, y1, 'ko', label='data')

    plt.figure()
    plt.suptitle('Difference between numpy and this')
    plt.plot(x_interp, y_interp-py_interp, 'k')
    plt.grid()
    plt.show()


lin_int_test1()
spline_test1()
spline_test2()
