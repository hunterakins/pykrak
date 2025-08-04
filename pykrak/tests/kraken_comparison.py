"""
Description:
    Compare pykrak wavenumber estimates to kraken using same mesh

Date:
    10/29/2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt

from pykrak import test_helpers as th
from pyat.pyat import readwrite as rw
from pykrak import krak_routines as kr
from pykrak import field as field
import os


def kr_comp():
    env_files = [
        "ice_env.env",
        "solve2test.env",
        "pekeris_layer_attn.env",
        "pekeris.env",
        "pekeris_attn.env",
        "pekeris_rough_bdry.env"
    ]
    for env in env_files:
        # os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env(
            "at_files/{}.env".format(env[:-4]), "kraken"
        )
        c_low, c_high = cint.Low, cint.High
        attn_units = "dbplam"

        modes = rw.read_modes(
            **{"fname": "at_files/{}.mod".format(env[:-4]), "freq": freq}
        )
        krak_k = modes.k
        krak_modes = modes.phi
        z = modes.z
        krak_prt_k, mode_nums, vps, vgs = th.read_krs_from_prt_file(
            "at_files/{}.prt".format(env[:-4]), verbose=True
        )
        print("krak M", modes.M)

        RMax = 0.0

        (
            pykrak_env,
            N_list,
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            cp_hs,
            cs_hs,
            rho_hs,
            attnp_hs,
            attns_hs,
            pos,
            beam,
            cint,
            RMax,
        ) = th.init_pykrak_env(ssp, bdry, pos, beam, cint, RMax)
        pk_krs, phi_z, phi, ugs = pykrak_env.get_modes(
            freq, N_list, rmax=RMax, c_low=c_low, c_high=c_high
        )
        print("pykrak M", pk_krs.size)
        for i in range(len(krak_prt_k)):
            mode_num = mode_nums[i]
            krak_k = str(krak_prt_k[i].real)
            pk_k = str(pk_krs[mode_num - 1].real)

            krak_k_imag = str(krak_prt_k[i].imag)
            pk_k_imag = str(pk_krs[mode_num - 1].imag)
            #print(krak_prt_k[i], pk_krs[mode_num - 1])
            print(
                "krak, pykrak, krak vg, pykrak_vg",
                krak_k + "   j" + krak_k_imag,
                pk_k + "   j" + pk_k_imag,
                vgs[i],
                ugs[mode_num - 1],
            )
            num_dig = 0
            count = 0
            while (
                count < (min(len(krak_k), len(pk_k))) and krak_k[count] == pk_k[count]
            ):
                count += 1
            plt.plot(mode_num, count - 1, "ko")  # exclude decimal

            while (
                count < (min(len(krak_k_imag), len(pk_k_imag))) and krak_k_imag[count] == pk_k_imag[count]
            ):
                count += 1
            plt.plot(mode_num, count - 1, "b+")  # exclude decimal
        plt.xlabel("Mode num")
        plt.ylabel("Number of digits in agreement \n(Includes leading zero")
        plt.grid()
        plt.suptitle("Comparison of pykrak and kraken wavenumbers for {}".format(env))
        plt.show()


def phi_comp():
    env_files = [
        "solve2test.env",
        "pekeris_layer_attn.env",
        "pekeris_attn.env",
        "pekeris.env",
    ]
    for env in env_files:
        # os.system('cd at_files/ && kraken.exe {}'.format(env[:-4]))
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env(
            "at_files/{}.env".format(env[:-4]), "kraken"
        )

        c_low, c_high = cint.Low, cint.High

        modes = rw.read_modes(
            **{"fname": "at_files/{}.mod".format(env[:-4]), "freq": freq}
        )
        krak_k = modes.k
        krak_modes = modes.phi
        rho = modes.rho

        z = modes.z
        print("krak M", modes.M)

        (
            pykrak_env,
            N_list,
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            cp_hs,
            cs_hs,
            rho_hs,
            attnp_hs,
            attns_hs,
            pos,
            beam,
            cint,
            RMax,
        ) = th.init_pykrak_env(ssp, bdry, pos, beam, cint, RMax)

        print("N_list", N_list)

        pk_krs, phi_z, phi, ugs = pykrak_env.get_modes(
            freq, N_list, rmax=0.0, c_low=c_low, c_high=c_high
        )
        phi_new = np.zeros((z.size, phi.shape[1]))
        for i in range(phi.shape[1]):
            phi_new[:, i] = np.interp(z, phi_z, phi[:, i])

        phi = phi_new
        phi_z = z

        for i in range(3):
            plt.figure()
            plt.suptitle('Mode {}'.format(i+1))
            plt.plot(phi[:, 1], phi_z, label="PyKrak")
            plt.plot(krak_modes[:, 1].real, z, label="KRAKEN")
            plt.legend()
            plt.gca().invert_yaxis()
            plt.xlabel('Phi')
            plt.ylabel('z (m)')


            plt.figure()
            plt.plot(phi[:, i] - krak_modes[:, i], phi_z)
            plt.suptitle("Difference in mode shape from PyKrak and KRAKEN\nMode {}".format(i+1))
            plt.ylabel('z (m)')
            plt.xlabel('Difference in mode shape')
            plt.gca().invert_yaxis()
        plt.show()

def field_comp():
    env_files = [
        "pekeris.env",
    ]
    for env in env_files:
        TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env(
            "at_files/{}.env".format(env[:-4]), "kraken"
        )
        c_low, c_high = cint.Low, cint.High
        attn_units = "dbplam"


        RMax = 0.0

        (
            pykrak_env,
            N_list,
            z_list,
            cp_list,
            cs_list,
            rho_list,
            attnp_list,
            attns_list,
            cp_hs,
            cs_hs,
            rho_hs,
            attnp_hs,
            attns_hs,
            pos,
            beam,
            cint,
            RMax,
        ) = th.init_pykrak_env(ssp, bdry, pos, beam, cint, RMax)
        pk_krs, phi_z, phi, ugs = pykrak_env.get_modes(
            freq, N_list, rmax=RMax, c_low=c_low, c_high=c_high
        )

        zs = 500.0
        zs_arr = np.array([zs])
        zr = 2500.0
        zr_arr = np.array([zr])
        rr_arr = np.linspace(200.0, 220, 1001)*1e3
        rr_offset = 0.0
        print('freq', freq)
        rr_offset_arr = np.array([rr_offset])
        c_ref = 1500.0
        #beam_pattern = np.empty((0, 2), dtype=np.float64)

        cp = field.get_pressure(pk_krs, phi_z, phi, zs_arr, zr_arr, rr_arr, rr_offset_arr, freq)
        cp_src = field.get_pressure(pk_krs, phi_z, phi, zs_arr, zr_arr, np.array([1.0]), rr_offset_arr, freq)
        plt.figure()
        P_DB = 20*np.log10(np.abs(cp[0,0,:])) # get field along the line
        P_DB_src = 20*np.log10(np.abs(cp_src[0,0,0])) # get field at 1 m from the source
        print('P_DB_src', P_DB_src)
        TL = P_DB_src - P_DB
        plt.suptitle("Comparison of PyKRAK field for {}".format(env))
        plt.plot(rr_arr/1e3, TL, label="PyKrak")
        #plt.colorbar(label="TL (dB)")
        plt.gca().invert_yaxis()
        plt.xlabel('Range (km)')
        plt.ylabel('TL (dB)')
        plt.show()


field_comp()
kr_comp()
phi_comp()
