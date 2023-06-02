import numpy as np
from matplotlib import pyplot as plt
from pykrak.sturm_seq import get_krs, get_arrs
from pykrak.shooting_routines import shoot_first_layer, shoot_from_bottom
from pykrak.inverse_iteration import get_phi
from pykrak.attn_pert import add_attn, get_attn_conv_factor
from pykrak.group_pert import get_ugs
import numba as nb
"""
Description:
    This module contains the class Env, which is used to store the model parameters
    and manage the normal mode calculation
    It also contains a Modes object to store the output of a single frequency run

Date:
    4/18/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego

Copyright (C) 2023 F. Hunter Akins

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

class Modes:
    def __init__(self, freq, krs, phi, M, z):
        """ Mode output from a single frequency run """
        self.freq = freq
        self.krs = krs  
        self.phi = phi
        self.M = M
        self.z = z

    def get_phi_zr(self, zr):
        """
        Interpolate modes over array depths zr
        """
        phi = self.phi
        if np.all(phi==0):
            phi = self.get_phi()
        M = self.M
        phi_zr = np.zeros((zr.size, M))
        phi_z = self.z
        for i in range(M):
            phi_zr[:,i] = np.interp(zr, phi_z, phi[:,i])
        return phi_zr

class Env:
    """
    Model parameters for running normal mode model
    """
    def __init__(self, z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units):
        self.omega = None
        self.freq = None
        self.z_list = z_list
        self.c_list = c_list
        self.rho_list = rho_list
        self.attn_list = attn_list
        self.c_hs = c_hs
        self.rho_hs = rho_hs
        self.attn_hs = attn_hs
        self.krs = np.zeros(10)
        self.N_list =  []
        self.phi = np.zeros(10)
        self.M = None
        self.attn_units = attn_units
        self.conv_factor = None
        self.mode_dict = {}

    def filter_layers(self):
        """
        For a given frequency, merge layers that are thinner than lambda / 10
        """
        Z_min = (1500. / self.freq) / 10
        bad_layers = []
        i = 1
        z_list = self.z_list
        c_list = self.c_list
        rho_list = self.rho_list
        attn_list = self.attn_list
        while i < len(self.z_list) - 2:
            z_layer = self.z_list[i]
            Z = z_layer[-1] - z_layer[0]
            if Z < Z_min: # include it in the upper layer
                print('merging layers ', i, i+1)
                print(z_list[i-1], z_list[i])
                z_list[i-1] = np.concatenate((z_list[i-1], z_list[i]))
                c_list[i-1] = np.concatenate((c_list[i-1], c_list[i]))
                rho_list[i-1] = np.concatenate((rho_list[i-1], rho_list[i]))
                attn_list[i-1] = np.concatenate((attn_list[i-1], attn_list[i]))
                z_list.pop(i)
                c_list.pop(i)
                rho_list.pop(i)
                attn_list.pop(i)
            else:
                i += 1
        self.z_list = z_list
        self.c_list = c_list
        self.rho_list = rho_list
        self.attn_list = attn_list
        return
            
    def add_attn_conv_factor(self):
        if self.attn_units in ['npm','dbpm']:
            conv_factor = get_attn_conv_factor(self.attn_units)
        elif self.attn_units in ['dbplam','q']:
            lam = self.c_list[0][0] / self.freq
            args = [lam]
            conv_factor = get_attn_conv_factor(self.attn_units, *args)
        elif self.attn_units in ['dbpkmhz']:
            args = [self.freq]
            conv_factor = get_attn_conv_factor(self.attn_units, *args)
        else:
            raise ValueError('Unsupported attenuation unit')
        return conv_factor

    def add_freq(self,freq):
        self.freq = freq
        self.omega = 2*np.pi*freq

    def get_N_list(self):
        """
        Get grid of depths for sturm shooting
        """
        omega = self.omega
        if omega == None:
            raise ValueError('Need to add frequency first (env.add_freq(freq))')
        f= omega/2/np.pi
        lam = np.min(self.c_list[0]) / f
        dz = lam /20 # default lambda / 20 sampling
        N_list = []
        for i in range(len(self.z_list)):
            z_layer = self.z_list[i]
            Z = z_layer[-1] - z_layer[0] 
            N = int(Z / dz) 
            N += 1
            N = max(N, 10)
            N_list.append(N)
        self.N_list = N_list
        return N_list

    def get_h_list(self, N_list=None):
        if type(N_list) == type(None):# use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        h_list = []
        for i in range(len(N_list)):
            N = N_list[i]
            z = self.z_list[i]
            h = (z[-1] - z[0]) / (N-1)
            h_list.append(h)
        self.h_list = h_list
        return h_list

    def get_cmin(self):
        cmin = min([np.min(x) for x in self.c_list])
        return cmin
    
    def get_cmax(self):
        cmax = .99*self.c_hs  
        return cmax

    def interp_env_vals(self, N_list):
        """
        For specified mesh step size, interpolate the env ssp and density     
        onto the mesh
        """
        z_list = []
        c_list = []
        rho_list = []
        attn_list = []
        for i in range(len(N_list)):
            N = N_list[i]
            #print('N', N)
            z = self.z_list[i]
            c = self.c_list[i]
            attn = self.attn_list[i]
            rho = self.rho_list[i]
            new_z = np.linspace(z[0], z[-1], N)
            new_c = np.interp(new_z, z, c)            
            new_rho = np.interp(new_z, z, rho)            
            new_attn = np.interp(new_z, z, attn)            
            z_list.append(new_z)
            c_list.append(new_c)
            attn_list.append(new_attn)
            rho_list.append(new_rho)
        return z_list, c_list, rho_list, attn_list

    def get_krs(self, verbose=False, **kwargs):
        """
        Get the wavenumbers for the environment
        keyword args:
        cmin - minimum phase speed for retained wavenumbers
        cmax - maximum phase speed for retained wavenumbers
        Nh - manual selection of number of mesh grids to use
            if rmax is also specified, then this number becomes the 
            maximum number of mesh grids used
            otherwise this is the number of grids used
        rmax - maximum range to compute field to
        N_list - if you care to specificy the initial grid size
            useful for model comparison
            otherwise it will use the default lambda / 20
        """
        if 'cmin' in kwargs.keys():
            cmin = kwargs['cmin']
        else:
            cmin = self.get_cmin()
        if 'cmax' in kwargs.keys():
            cmax = kwargs['cmax']
        else:
            cmax = self.get_cmax()
        if 'Nh' in kwargs.keys():
            Nh = kwargs['Nh']
        else:
            Nh = 5
        if 'rmax' in kwargs.keys():
            rmax = kwargs['rmax']
        else:
            rmax = 1e10 # this should force the manual mesh
        if 'N_list' in kwargs.keys():
            N_list = kwargs['N_list']
        else:
            N_list  = self.get_N_list()  



        # set bounds
        kr_min = self.omega / cmax
        kr_max = self.omega / cmin
        self.N_list = N_list
  
        H = np.zeros((Nh, Nh))
        kr_meshes = [] # store wavenumbers from each mesh
        M = 1e10 # number of modes

        # loop over meshes
        for i in range(Nh):  # (max) num Richardson mesh refinements

            # refine mesh
            factor = int(np.power(2.0, i))
            curr_N_list = [(x-1)*factor+1 for x in N_list]
            #print('curr_N_list', curr_N_list)
            tmp_z_list, tmp_c_list, tmp_rho_list, tmp_attn_list = self.interp_env_vals(curr_N_list)
            curr_h_list = [x[1] - x[0] for x in tmp_z_list]
            h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(curr_h_list, tmp_z_list, tmp_c_list, tmp_rho_list)
            h0 = h_arr[0]

            # get wavenumbers for this mesh
            lam_min = np.square(h0*kr_min) 
            lam_max = np.square(h0*kr_max) 
            krs = get_krs(self.omega, h_arr, ind_arr, z_arr, c_arr, rho_arr,\
                         self.c_hs, self.rho_hs, lam_min, lam_max)
            krs = krs[::-1] # largest to smallest
            krs = np.array(krs) 
            kr_meshes.append(krs)

            M = min(M,krs.size) # keep track of number of modes

            if i == 0: # initialize matrices for extrapolations
                kr_sq_mat = np.zeros((Nh, M))
                extrap_mat = np.zeros((Nh, M))
                extrap_mat[0,:] = krs

            # truncate matrices if new mesh decreased num modes
            krs = krs[:M]
            kr_sq_mat = kr_sq_mat[:,:M] 
            kr_sq_mat[i,:] = np.square(krs)
            extrap_mat = extrap_mat[:,:M]

            # Filling up richardson extrapolation matrix for current mesh
            for k in range(Nh):
                H[i,k] = np.power(curr_h_list[0], 2*k)
                # In matrix form [kr_i^2(h_1) \\ kr_i^2(h_2) \\ \vdots \\ kr_i^2(h_Nh)] = H [kr_0^2 \\ b_2 \\ b_4 \\ \vdots \\ b_{2N_h} ]  
             
            # eventually make this recursive (don't think it's a bottleneck)
            if i >= 1: # do richardson for all meshes up to now
                # Now do Richardson... kr_{i}^{2}(h) = kr_{0}^{2} + b_{2} h^{2} + b_{4} h^{4}\;....
                tmp_H = H[:i+1,:i+1] # just use current meshes...
                rich_krs = np.zeros(M)
                for k in range(M):
                    y = np.linalg.solve(tmp_H, kr_sq_mat[:i+1,k])
                    rich_krs[k] = np.sqrt(y[0])
                extrap_mat[i,:] = rich_krs
                errs = abs(extrap_mat[i,:] - extrap_mat[i-1,:])
                err = errs[int(2*M/3)]  # consistent with kraken 
                if rmax*err < 1: # if error is small then break
                    if verbose==True:
                        print('rmax attained, mesh num. ', i+1)
                    break
        if i > 0:
            krs = rich_krs            
        else:
            krs = krs            

        krs = extrap_mat[i,:] # return best extrapolation vals
        coarse_krs = extrap_mat[0,:] # keep krs from first mesh for computing phi
        self.coarse_krs = coarse_krs

        """ Factor in attenuation with perturbation theory"""
        attn = False
        for layer_attn in self.attn_list:
            if np.any(layer_attn):
                attn=True
        if not attn and (self.attn_hs):
            attn = True
        if attn==True: 
            conv_factor = self.add_attn_conv_factor()
            self.krs = krs
            phi = self.get_phi(N_list)  # need phi to compute attenuation
            tmp_z_list, tmp_c_list, tmp_rho_list, tmp_attn_list = self.interp_env_vals(N_list)
            h_list = [x[1] - x[0] for x in tmp_z_list]
            tmp_attn_list, tmp_attn_hs = [conv_factor*x for x in tmp_attn_list], conv_factor*self.attn_hs
            krs = add_attn(self.omega, krs, phi, h_list, tmp_z_list, tmp_c_list, tmp_rho_list, tmp_attn_list, self.c_hs, self.rho_hs, tmp_attn_hs)
            
        else:
            self.krs = krs
            phi = self.get_phi(N_list)  # get unperturbed phi (no attenuation)
        self.krs = krs
        self.M = M
        self.phi = phi
        return krs

    def get_phi(self, N_list=None):
        """
        Calculate the mode functions
        Use initial mesh for trapezoid rule integration
        Use wavenumbers from initial mesh for inverse iteration
        """
        if np.all(self.coarse_krs==0):
            raise ValueError('Must run get_krs first')
        else:
            krs = self.coarse_krs.real
            if type(N_list) == type(None):# use lambda / 20 
                N_list  = self.get_N_list()  
            else: # manual grid number
                pass
            z_list, c_list, rho_list, attn_list = self.interp_env_vals(N_list)
            h_list = [x[1] - x[0] for x in z_list]
            h_arr, ind_arr, z_arr, c_arr, rho_arr = get_arrs(h_list, z_list, c_list, rho_list)
            phi = get_phi(krs, self.omega, h_arr, ind_arr, z_arr, c_arr, rho_arr, self.c_hs, self.rho_hs)
        self.phi = phi
        self.get_phi_z(N_list)
        return phi

    def get_phi_z(self, N_list=None):
        """
        Get grid of depths at which the mode functions are evaluated
        """
        # N_list is an optional argument that allows you to specify the number of points in each layer
        if type(N_list) == type(None): # use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        for i in range(len(N_list)):
            N = N_list[i]
            z = self.z_list[i]
            mesh_z = np.linspace(z[0], z[-1], N)
            if i == 0:
                phi_z = mesh_z
            else:
                """layers share end interface dpeths"""
                phi_z = np.concatenate((phi_z, mesh_z[1:])) 
        self.phi_z = phi_z
        return phi_z

    def get_rho_grid(self, N_list=None):
        """
        Get rho at values of mesh used
        N_list is an optional argument that allows you to specify the number of points in each layer
        """
        rho_grid = np.array([])
        if type(N_list) == type(None):# use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        z_list, c_list, rho_list, attn_list = self.interp_env_vals(N_list)
        first = True
        for rho_arr in rho_list:
            if first == True:
                rho_grid = np.hstack((rho_grid, rho_arr))
                first=False
            else:
                rho_grid = np.hstack((rho_grid, rho_arr[1:]))
        return rho_grid

    def get_phi_zr(self, zr):
        """ Get modes evaluated at depths zr"""
        phi = self.phi
        if np.all(phi==0):
            phi = self.get_phi()
        M = phi.shape[-1]
        phi_zr = np.zeros((zr.size, M))
        phi_z = self.phi_z
        for i in range(M):
            phi_zr[:,i] = np.interp(zr, phi_z, phi[:,i])
        return phi_zr

    def get_ugs(self, N_list=None):
        """ Get group speeds of modes """
        if type(N_list) == type(None):# use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        z_list, c_list, rho_list, attn_list = self.interp_env_vals(N_list)
        h_list = [x[1] - x[0] for x in z_list]
        ugs = get_ugs(self.omega, self.krs.real, self.phi, h_list, z_list, c_list, rho_list, self.c_hs, self.rho_hs)
        self.ugs = ugs
        return ugs

    def add_to_mode_dict(self, freq, cmin=None, cmax=None, Nh=5):
        self.add_freq(freq)
        kwargs = {}
        if cmin is not None:
            kwargs['cmin'] = cmin
        if cmax is not None:
            kwargs['cmax'] = cmax
        if Nh is not None:
            kwargs['Nh'] = Nh
        krs = self.get_krs(**kwargs)
        M = self.krs.size
        phi = self.phi
        z = self.get_phi_z()
        modes = Modes(freq, krs, phi, M, z)
        self.mode_dict[freq] = modes
        return krs
        
    def plot_env(self, ax=None, color=None):
        """
        Very basic plotting
        """
        if ax is None:
            fig, ax = plt.subplots(1,1)
        for i in range(len(self.z_list)):
            if color is None:
                ax.plot(self.c_list[i], self.z_list[i])
            else:
                ax.plot(self.c_list[i], self.z_list[i], color)
        ax.set_ylim([self.z_list[-1][-1] + 50, 0])
        for i in range(len(self.z_list)):
            ax.hlines(self.z_list[i][-1], min([x.min() for x in self.c_list]), max([x.max() for x in self.c_list]), 'k', alpha=0.5)
        ax.hlines(self.z_list[-1][-1], min([x.min() for x in self.c_list]), max([x.max() for x in self.c_list]), 'k')
        mean_layer_c = sum([x.mean() for x in self.c_list]) / len(self.c_list)
        ax.text(mean_layer_c , self.z_list[-1][-1] +30 ,  '$c_b$:{0}, \n$\\rho_b$:{1}, \n$\\alpha_b$:{2}'.format(self.c_hs, self.rho_hs, self.attn_hs))
        print(self.c_list[-1][-1] , self.z_list[-1][-1] + 40)
        return

    def shoot_mode(self, kr, zr=None, normalize=False, N_list=None):
        """ For a guess kr, shoot through top water layer"""
        if type(N_list) == type(None):# use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        z_list, c_list, rho_list, attn_list = self.interp_env_vals(N_list)
        h_list = [x[1] - x[0] for x in z_list]
        h0 = h_list[0]
        lam = np.square(h0*kr)
        z, c =z_list[0], c_list[0]
        mode = shoot_first_layer(self.omega, h0, z, c, None, lam)
        """ Only approximate """
        if normalize == True:
            dz = z[1]- z[0]
            om_sq = np.square(2*np.pi*self.freq)
            mode *= 1.0 / np.sqrt(dz * np.sum(np.square(mode)))

        if  type(zr) != type(None):
            mode = np.interp(zr, z, mode)
            z = zr
        return z, mode 

    def shoot_mode_up(self, kr, zr=None, normalize=False, N_list=None):
        """ For a guess kr, shoot from the halfspace up to the surface """
        if type(N_list) == type(None):# use lambda / 20 
            N_list  = self.get_N_list()  
        else: # manual grid number
            pass
        z_list, c_list, rho_list, attn_list = self.interp_env_vals(N_list)
        h_list = [x[1] - x[0] for x in z_list]
        h0 = h_list[0]
        lam = np.square(h0*kr)
        mode, z = shoot_from_bottom(self.omega, nb.typed.List(h_list), nb.typed.List(z_list), nb.typed.List(c_list), nb.typed.List(rho_list), self.c_hs, self.rho_hs, lam)
        """ Only approximate """
        if normalize == True:
            dz = z[1]- z[0]
            om_sq = np.square(2*np.pi*self.freq)
            mode *= 1.0 / np.sqrt(dz * np.sum(np.square(mode)))

        if  type(zr) != type(None):
            mode = np.interp(zr, z, mode)
            z = zr
        return z, mode 

