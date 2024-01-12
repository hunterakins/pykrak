"""
Description:
    An environment where the SSP is specified by values at nodal depths
    Each node has c_p, rho, and attn

Date:
    1/11/2023

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

class NodeEnvError(Exception):
    pass

class NodeEnv(Env):
    def __init__(self, freq, z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units, cmin, cmax):
        super().__init__(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units)
        self.freq = freq
        self.z_arr, self.ind_arr = cat_list_to_arr(z_list)
        self.c_arr, _ = cat_list_to_arr(c_list)
        self.rho_arr, _ = cat_list_to_arr(rho_list)
        self.attn_arr, _ = cat_list_to_arr(attn_list)
        self.add_freq(freq)
        self.cmin = cmin
        self.cmax = cmax
        self.zs_arr = None
        self.zr_arr = None
        self.modes = None
        self.phi_zs = None
        self.phi_zr = None
        self.x0 = None
        self.add_attn_conv_factor()

    def set_node_val(self, layer_ind, node_ind, z_val, c_val, rho_val, attn_val):
        self.z_list[layer_ind][node_ind] = z_val
        self.c_list[layer_ind][node_ind] = c_val
        self.rho_list[layer_ind][node_ind] = rho_val
        self.attn_list[layer_ind][node_ind] = attn_val
        return

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

    def get_x0(self):
        """
        Get the node val matrix x
        that matches the original passed environment
        """
        N = self.z_arr.size 
        x = np.zeros((N+1, 5))
        x[:-1, 0] = np.linspace(0, N-1, N, dtype=int)
        x[:-1, 1] = self.z_arr
        x[:-1, 2] = self.c_arr
        x[:-1, 3] = self.rho_arr
        x[:-1, 4] = self.attn_arr
        x[-1, 0] = self.c_hs
        x[-1, 1] = self.rho_hs
        x[-1, 1] = self.attn_hs
        return x

    def get_forward_p(self, h_mesh):
        """
        Forward model p is a function 
        That takes in
        r_arr: array of receiver ranges
        zs_arr: array of source depths
        zr_arr: array of receiver depths
        tilt: float (tilt of array) 
        x0 : array of perturbations to sound speed
             [node_ind, z, c_p, rho, attn]
             last row is [c_hs, rho_hs, attn_hs, 0, 0]
             past in -1 for values if you don't want them to change
        """

        freq = self.freq

        ind_arr, z_arr, c_arr, rho_arr, attn_arr = self.ind_arr, self.z_arr, self.c_arr, self.rho_arr, self.attn_arr
        conv_factor = self.conv_factor
        cmin = self.cmin
        cmax = self.cmax

        def get_p(r_arr, zs_arr, zr_arr, tilt, x0):
            """
            First insert the values in x0 in the nodes 
            Note that I don't update the halfspace values here
            """
            num_perts = x0.shape[0]-1
            tmp_ind_arr = ind_arr.copy()
            tmp_c_arr = c_arr.copy()
            tmp_rho_arr = rho_arr.copy()
            tmp_attn_arr = attn_arr.copy()
            tmp_z_arr = z_arr.copy()

            for i in range(num_perts):
                node_ind, z, c, rho, attn = x0[i,:]
                node_ind = int(node_ind)

                # if it's a layer point make sure you adjust the depth of the shared node depth
                if node_ind < z_arr.size-1:
                    if z_arr[node_ind + 1] == z_arr[node_ind]:
                        z_arr[node_ind + 1] = z
                if node_ind > 0:
                    if z_arr[node_ind - 1] == z_arr[node_ind]:
                        z_arr[node_ind - 1] = z
                z_arr[node_ind] = z
                c_arr[node_ind] = c
                rho_arr[node_ind] = rho
                attn_arr[node_ind] = attn

            """
            Now put the node values onto a mesh with linear interpolation
            """
            ind_mesh = np.zeros(ind_arr.size, dtype=int)
            h_mesh_arr = np.zeros(ind_arr.size)
            for i in range(ind_arr.size): # num layers
                if i != ind_arr.size - 1:
                    z0 = z_arr[ind_arr[i]]
                    z1 = z_arr[ind_arr[i+1]]
                    z_layer = z_arr[ind_arr[i]:ind_arr[i+1]]
                    c_layer = c_arr[ind_arr[i]:ind_arr[i+1]]
                    rho_layer = rho_arr[ind_arr[i]:ind_arr[i+1]]
                    attn_layer = attn_arr[ind_arr[i]:ind_arr[i+1]]
                else:
                    z0 = z_arr[ind_arr[i]]
                    z1 = z_arr[-1]
                    z_layer = z_arr[ind_arr[i]:-1]
                    c_layer = c_arr[ind_arr[i]:-1]
                    rho_layer = rho_arr[ind_arr[i]:-1]
                    attn_layer = attn_arr[ind_arr[i]:-1]
                Z = z1 - z0
                N = max(10, int(Z/h_mesh)+1)
                if i == 0:
                    z_mesh = np.linspace(z0, z1, N)
                    c_mesh = np.interp(z_mesh, z_layer, c_layer)
                    rho_mesh = np.interp(z_mesh, z_layer, rho_layer)
                    attn_mesh = np.interp(z_mesh, z_layer, attn_layer)
                else:
                    layer_mesh = np.linspace(z0, z1, N)
                    c_mesh = np.concatenate((c_mesh, np.interp(layer_mesh, z_layer, c_layer)))
                    rho_mesh = np.concatenate((rho_mesh, np.interp(layer_mesh, z_layer, rho_layer)))
                    attn_mesh = np.concatenate((attn_mesh, np.interp(layer_mesh, z_layer, attn_layer)))
                    z_mesh = np.concatenate((z_mesh, layer_mesh))

                h_mesh_arr[i] = z_mesh[1] - z_mesh[0]
                if i < ind_arr.size - 1:
                    ind_mesh[i+1] = ind_mesh[i] + N

            """
            Now add in attenuation and convert to k_sq vals
            """
            attn_mesh_npm = conv_factor * attn_mesh
            c_imag = get_c_imag_npm(c_mesh, attn_mesh_npm, 2*np.pi*freq)
            c_mesh = c_mesh + 1j*c_imag
            k_sq_mesh = (2*np.pi*freq/c_mesh)**2

            """
            Deal with halfspace
            """
            c_hs, rho_hs, attn_hs = x0[-1,0], x0[-1,1], x0[-1,2]
            attn_hs_npm = conv_factor * attn_hs
            c_hs = c_hs + 1j*get_c_imag_npm(c_hs, attn_hs_npm, 2*np.pi*freq)
            k_hs_sq = (2*np.pi*freq/c_hs)**2
                
            krs, phi, phi_z = get_modes(freq, h_mesh_arr, ind_mesh, z_mesh, k_sq_mesh, rho_mesh, k_hs_sq, rho_hs, cmin, cmax)
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

def test1():
    freq = 100.0
    z_list = [np.array([0, 25.0, 50.0, 150.0, 200.0]), np.array([200.0, 220.0])]
    c_list = [1500.0*np.ones(z_list[0].size), 1600.0*np.ones(z_list[1].size)]
    rho_list = [1.0*np.ones(z_list[0].size), 1.8*np.ones(z_list[1].size)]
    attn_list = [0.0*np.ones(z_list[0].size), 0.2*np.ones(z_list[1].size)]
    attn_units = 'dbplam'
    c_hs = 1700.0
    rho_hs = 2.5
    attn_hs = 0.6
    cmin = 1300.0
    cmax = c_hs-1e-10
    ne = NodeEnv(freq, z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs, attn_units, cmin,cmax)
    #ne.set_node_val(0, 0, 0.0, 1520.0, 1.0, 0.0)
    #ne.set_node_val(1,1, 220.0, 1620.0, 1.8, 0.2)
    #ne.plot_env()

    x = ne.get_x0()
    lam = 1500.0 / freq
    h_mesh = lam / 10
    pf = ne.get_forward_p(h_mesh)

    r_arr = np.linspace(100.0, 10*1e3, 100)
    zs_arr = np.array([55.0])
    zr_arr = np.linspace(10.0, 190.0, 200)
    tilt = 0
    out = pf(r_arr, zs_arr, zr_arr, tilt, x)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.pcolormesh(r_arr, zr_arr, 20*np.log10(np.abs(out[0,:,:])))
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test1()
    

