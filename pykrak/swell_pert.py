import numpy as np
from matplotlib import pyplot as plt
from pykrak.pykrak_env import Env
from numba import njit
from pykrak.misc import get_simpsons_integrator, test_simpsons_integrator

"""
Description:
Create an inherited env object that add methods for mode sensitivity
required for linear inversion of modal data

Date:
6/10/2022

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

class SwellPertEnv(Env):
    """ Environment with extra methods for linearizing """

    def __init__(self, z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units):
        super().__init__(z_list, c_list, rho_list, attn_list, c_hs, rho_hs, attn_hs,attn_units)
        self.M_list = [None]*len(c_list) # 
        self.H_dict = {} # dictionary of linearized forward models keyed on frequency
        self.A_dict = {} # dictionary of linearized mode shape keyed on frequency
        self.dz_list = [x[1] - x[0] for x in self.z_list]

    def add_model_matrix(self, layer_M, layer_ind):
        """ 
        Model matrix give a representation for 
        SSP perturbations in layer_index layer_ind in terms of basis given
        by columns of model matrix
        For example, if a layer has three depth points, then M should 
        have dimensions 3 x num_params 
        """
        self.M_list[layer_ind] = layer_M
        return

    def perturb_env(self, m_list):
        """
        Given list of model parameters for each layter,
        apply model matrices to add perturbations to c in each 
        layer
        """
        for layer_ind in range(len(self.c_list)):
            layer_M = self.M_list[layer_ind]
            if layer_M is not None and (layer_ind < len(m_list)):
                delta_c = (layer_M@(m_list[layer_ind]))[:,0]
                self.c_list[layer_ind] += delta_c
        return

    def unperturb_env(self, m_list):
        """ Undo the perturbtion from perturb_env """
        for layer_ind in range(len(self.c_list)):
            layer_M = self.M_list[layer_ind]
            if layer_M is not None and (layer_ind < len(m_list)):
                delta_c = (layer_M@(m_list[layer_ind]))[:,0]
                self.c_list[layer_ind] -= delta_c
        return

    def update_bottom_layers(self, sed_x, clay_x):
        """ Really should create a separate object for swellex for this..."""
        h1, cb1, s1, rhob1, alpha1 = sed_x
        h2, cb2, s2, rhob2, alpha2 = clay_x
        self.z_list[1][1] = self.z_list[1][0] + h1
        self.c_list[1][0] = cb1
        self.c_list[1][1] = cb1 + s1 * h1
        self.rho_list[1][:] = rhob1
        self.attn_list[1][:] = alpha1

        self.z_list[2][0] = self.z_list[1][1]
        self.z_list[2][1] = self.z_list[2][0] + h2
        self.c_list[2][0] = cb2
        self.c_list[2][1] = cb2 + s2 * h2
        self.rho_list[2][:] = rhob2
        self.attn_list[2][:] = alpha2
        return

    def update_single_layer_bott(self, sed_x):    
        """ Really should create a separate object for swellex for this..."""
        h1, cb1, s1, rhob1, alpha1 = sed_x
        self.z_list[1][1] = self.z_list[1][0] + h1
        self.c_list[1][0] = cb1
        self.c_list[1][1] = cb1 + s1 * h1
        self.rho_list[1][:] = rhob1
        self.attn_list[1][:] = alpha1
        return

    def update_fine_single_layer_bott(self, sed_x):
        h1, cb1, s1, rhob1, alpha1 = sed_x
        dz = self.dz_list[1]
        print('dz', dz)
        new_N = int(h1 / dz)
        new_z = np.linspace(self.z_list[1][0], self.z_list[1][0] + h1, new_N)
        new_c = cb1 + (new_z - new_z[0]) * s1
        self.z_list[1] = new_z
        self.c_list[1] = new_c
        self.rho_list[1] = rhob1 * np.ones(new_N) 
        self.attn_list[1] = alpha1 * np.ones(new_N) 
        return

    def update_fine_double_layer_bott(self, x):
        h1, cb1, s1, rhob1, alpha1 = x[:5]
        h2, cb2, s2, rhob2, alpha2 = x[5:]
        dz = self.dz_list[1]
        new_N = int(h1 / dz)
        new_z = np.linspace(self.z_list[1][0], self.z_list[1][0] + h1, new_N)
        new_c = cb1 + (new_z - new_z[0]) * s1
        self.z_list[1] = new_z
        self.c_list[1] = new_c
        self.rho_list[1] = rhob1 * np.ones(new_N) 
        self.attn_list[1] = alpha1 * np.ones(new_N) 

        new_N = int(h2 / dz)
        new_z = np.linspace(self.z_list[1][-1], self.z_list[1][-1] + h2, new_N)
        new_c = cb2 + (new_z - new_z[0]) * s2
        self.z_list[2] = new_z
        self.c_list[2] = new_c
        self.rho_list[2] = rhob2 * np.ones(new_N) 
        self.attn_list[2] = alpha2 * np.ones(new_N) 
        
        return

    def update_fine_triple_layer_bott(self, x):
        h1, cb1, s1, rhob1, alpha1 = x[:5]
        h2, cb2, s2, rhob2, alpha2 = x[5:10]
        h3, cb3, s3, rhob3, alpha3 = x[10:]
        dz = self.dz_list[1]
        new_N = int(h1 / dz)
        new_z = np.linspace(self.z_list[1][0], self.z_list[1][0] + h1, new_N)
        new_c = cb1 + (new_z - new_z[0]) * s1
        self.z_list[1] = new_z
        self.c_list[1] = new_c
        self.rho_list[1] = rhob1 * np.ones(new_N) 
        self.attn_list[1] = alpha1 * np.ones(new_N) 

        new_N = int(h2 / dz)
        new_z = np.linspace(self.z_list[1][-1], self.z_list[1][-1] + h2, new_N)
        new_c = cb2 + (new_z - new_z[0]) * s2
        self.z_list[2] = new_z
        self.c_list[2] = new_c
        self.rho_list[2] = rhob2 * np.ones(new_N) 
        self.attn_list[2] = alpha2 * np.ones(new_N) 

        new_N = int(h3 / dz)
        new_z = np.linspace(self.z_list[2][-1], self.z_list[2][-1] + h3, new_N)
        new_c = cb3 + (new_z - new_z[0]) * s3
        self.z_list[3] = new_z
        self.c_list[3] = new_c
        self.rho_list[3] = rhob3 * np.ones(new_N) 
        self.attn_list[3] = alpha3 * np.ones(new_N) 
        return

    def update_hs(self, clay_x):
        self.c_hs = clay_x[0]
        self.rho_hs = clay_x[1]
        self.attn_hs = clay_x[2]
        return

    def get_H(self, freq, cmax):
        """ 
        H is a layer-by-layer linearization of the forward model (computing kr)
        about the default parameters for the environment
        Within each layer, a linear model H 
        maps a vector m to a vector delta_k where the vector m
        is related to a sound speed perturbation according to the 
        linear model of that layer ($\Delta c = \bm{M}_{i} \bm{m}$), where
        $\bm{M}_{i}$ is the ith layer model matrix
        Then $\Delta k_{n} = \int_{z_{up}}^{z_{down}} \Delta c(z) S_{n}(z) \dd z$, 
        where the sensitivity of the $n$th mode is 
        $S_{n}(z) = \frac{-\omega^{2}}{k_{bn} c_{b}(z)^{3} \rho(z)} \abs{\phi_{bn}(z)}^{2}  $
        """
        if freq not in self.mode_dict.keys(): # haven't run get_krs
            self.add_to_mode_dict(freq, cmax=cmax)
        modes = self.mode_dict[freq]


        H_list = [] # one integral for each layer
        for layer_ind in range(len(self.c_list)):
            M = self.M_list[layer_ind] #model matrix for layer
            if M is None:
                H_layer = None
            else:
                z_layer = self.z_list[layer_ind]
                c_layer = self.c_list[layer_ind]
                rho_layer = self.rho_list[layer_ind]
                dz_layer = z_layer[1] - z_layer[0]
                N_pts = z_layer.size
                H_layer = np.zeros((modes.M, M.shape[1])) # map perturbation coeffs to delta kr
            
                base_integrator = get_simpsons_integrator(N_pts, dz_layer)
                for kr_ind in range(modes.M):
                    phi_n = np.interp(z_layer, modes.z, modes.phi[:,kr_ind])
                    Sn = -np.square(2*np.pi*freq)*np.square(abs(phi_n))*np.power(c_layer, -3) / (modes.krs[kr_ind].real * rho_layer) 
                    H_layer[kr_ind,:] = (Sn*base_integrator)@M
            H_list.append(H_layer)
        self.H_dict[freq] = H_list
        return H_list

    def get_A(self, freq, cmax):
        """ 
        A is a layer-by-layer linearization of the forward model for the mode shapes
        The perturbation for each mode shape is given as a linear combination of 
        all the other modes
        \Delta \Phi_{n} = \sum_{m \neq n} a_{mn} \Phi_{m} , where $\Phi_{m} are the unperturbed modes and
        a_{mn} =  -2 \frac{\omega^{2}}{k_{n} - k_{m}} \int_{0}^{\infty} \Delta c(z') \Phi_{n} \{Phi_{m} / c_{0}^{3}(z') \dd z' \; .
        """
        if freq not in self.mode_dict.keys(): # haven't run get_krs
            self.add_to_mode_dict(freq, cmax=cmax)
        modes = self.mode_dict[freq]


        A_list = [] # one integral for each layer
        for layer_ind in range(len(self.c_list)):
            M = self.M_list[layer_ind] #model matrix for layer
            if M is None:
                A_layer = None
            else:
                z_layer = self.z_list[layer_ind]
                c_layer = self.c_list[layer_ind]
                rho_layer = self.rho_list[layer_ind]
                dz_layer = z_layer[1] - z_layer[0]
                N_pts = z_layer.size
                A_layer = np.zeros((modes.M, modes.M, M.shape[1])) # map mode matrix to mode perturbation matrix
            
                base_integrator = get_simpsons_integrator(N_pts, dz_layer)
            
                for i in range(modes.M):
                    phi_i = np.interp(z_layer, modes.z, modes.phi[:,i])
                    for j in range(i+1, modes.M):
                        phi_j = np.interp(z_layer, modes.z, modes.phi[:,i])
                        Si = -np.square(2*np.pi*freq)*phi_i*phi_j*np.power(c_layer, -3) / ((modes.krs[i] - modes.krs[j]).real* rho_layer) 
                        A_layer[i, j,:] = (Si*base_integrator)@M
                        A_layer[j, i,:] = - A_layer[i, j, :]
            A_list.append(A_layer)
        self.A_dict[freq] = A_list
        return A_list

    def get_lin_dkr(self, freq, cmax, m_list):
        """ At given frequency, for perturbation parameter layer
        list m_list, get linearized delta kr 
        """
        if freq in self.H_dict.keys():
            H_list = self.H_dict[freq]
        else:
            H_list = self.get_H(freq, cmax)
        modes = self.mode_dict[freq]
        dkr = np.zeros((modes.M))
        for layer_ind in range(len(m_list)):
            H = H_list[layer_ind]
            m = m_list[layer_ind]
            if H is not None:
                layer_dkr = H@m
                dkr += layer_dkr[:,0]
        return dkr

    def get_lin_dphi(self, freq, cmax, m_list):
        """ At given frequency, for perturbation parameter layer
        list m_list, get linearized delta kr 
        """
        if freq in self.A_dict.keys():
            A_list = self.A_dict[freq]
        else:
            A_list = self.get_A(freq, cmax)
        modes = self.mode_dict[freq]
        dkr = np.zeros((modes.M))
        z, phi = modes.z, modes.phi
        full_A = np.zeros((modes.M, modes.M))
        for layer_ind in range(len(m_list)):
            A = A_list[layer_ind]
            m = m_list[layer_ind]
            layer_A = A@m
            print(full_A.shape, layer_A.shape)
            full_A += np.squeeze(layer_A)
        delta_phi = phi@full_A
        return delta_phi

if __name__ == '__main__':
    test_simpsons_integrator()
            
