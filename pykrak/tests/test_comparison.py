"""
Description:
    Compare pykrak and to kraken wavenumber and modal depth function estimates
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pyat import readwrite as rw
from pykrak import field as field
from pykrak import krak_routines as kr
from pykrak import test_helpers as th

env_test_files = list(
    map(
        lambda x: Path("pykrak", "tests", "at_files", x),
        [
            "ice",
            "atten",
            "elsed",
            "flused",
            "double",
            "ice_env",
            "solve2test",
            "pekeris_layer_attn",
            "pekeris",
            "pekeris_attn",
            "pekeris_rough_bdry",
        ],
    )
)


@pytest.mark.parametrize("env", env_test_files, ids=lambda path: path.stem)
def test_kr_comp(env: Path):
    TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env(
        str(env.with_suffix(".env")), "kraken"
    )
    c_low, c_high = cint.Low, cint.High

    modes = rw.read_modes(**{"fname": str(env.with_suffix(".mod")), "freq": freq})
    z = modes.z
    krak_prt_k, mode_nums, vps, vgs = th.read_krs_from_prt_file(
        str(env.with_suffix(".prt")), verbose=False
    )
    krak_prt_k = np.array(krak_prt_k)

    RMax = RMax * 1e3  # Convert to meters

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

    max_mode_idx = None
    if modes.M != pk_krs.size:
        warnings.warn(
            f"Warning: Number of modes in kraken and pykrak do not match! {modes.M} != {pk_krs.size} for envs {env}"
        )
        max_mode_idx = min(modes.M, pk_krs.size)

    failures = {}
    for i_m, mode_num in enumerate(mode_nums):
        if max_mode_idx is not None:
            if (i_m < max_mode_idx) or (mode_num < max_mode_idx):
                break
        try:
            assert np.isclose(
                krak_prt_k[i_m],
                pk_krs[mode_num - 1],
                atol=2 * np.pi * freq / c_high * 1e-18,
            ), f"""Larger error in real part of horizontal wavenumber for env {env.stem} at mode {mode_num}.
            Found kraken : {krak_prt_k[i_m]} | pykrak {pk_krs[mode_num - 1]}
            """
        except AssertionError as e:
            failures[mode_num] = (krak_prt_k[i_m], pk_krs[mode_num - 1])

    if failures:
        failures_table_header = f"\n| mode  | {'kraken':12} | {'pykrak':12} |"
        failures_table_content = "\n".join(
            f"| {k:5} | {v[0]:12} | {v[1]:12} |" for k, v in failures.items()
        )
        raise AssertionError(
            f"""Larger error in horizontal wavenumber for env {env.stem} at modes {list(failures.keys())}.
            Found: {failures_table_header}\n{failures_table_content}
            """
        )


@pytest.mark.parametrize("env", env_test_files, ids=lambda path: path.stem)
def test_phi_comp(env: Path, plots_enabled):
    TitleEnv, freq, ssp, bdry, pos, beam, cint, RMax = rw.read_env(
        str(env.with_suffix(".env")), "kraken"
    )

    c_low, c_high = cint.Low, cint.High

    modes = rw.read_modes(**{"fname": str(env.with_suffix(".mod")), "freq": freq})
    krak_phi = modes.phi

    krak_prt_k, mode_nums, vps, vgs = th.read_krs_from_prt_file(
        str(env.with_suffix(".prt")), verbose=False
    )
    z = modes.z

    RMax = RMax * 1e3

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

    pk_krs, phi_z, pk_phi, ugs = pykrak_env.get_modes(
        freq, N_list, rmax=RMax, c_low=c_low, c_high=c_high
    )
    phi_new = np.zeros((z.size, pk_phi.shape[1]))
    for i in range(pk_phi.shape[1]):
        phi_new[:, i] = np.interp(z, phi_z, pk_phi[:, i])

    pk_phi = phi_new
    phi_z = z

    max_mode_idx = None
    if modes.M != pk_krs.size:
        warnings.warn(
            f"Warning: Number of modes in kraken and pykrak do not match! {modes.M} != {pk_krs.size} for envs {env}"
        )
        max_mode_idx = min(modes.M, pk_krs.size)

    failures = {}
    for i_m, (krak_phi_m, pk_phi_m) in enumerate(zip(krak_phi.real.T, pk_phi.T)):
        if max_mode_idx is not None:
            if i_m < max_mode_idx:
                break
        try:
            assert np.allclose(
                pk_phi_m, krak_phi_m, atol=1e-6
            ), f"""Larger error on modal depth function for env {env.stem} at mode {i_m + 1}.
            """
            # Found kraken : {krak_prt_k[i_m]} | pykrak {pk_krs.real[mode_num - 1]}
        except AssertionError as e:
            # pass
            failures[i_m + 1] = (krak_phi_m, pk_phi_m)

    if failures:
        if plots_enabled:
            custom_lines = [
                Line2D([0], [0], color="k", label="pykrak"),
                Line2D([0], [0], color="k", marker="x", lw=0, label="kraken"),
            ]
            plt.figure()
            plt.title(f"Env {env.stem}")
            u = 0
            for m, (krak_phi_m, pk_phi_m) in failures.items():
                u += 1
                line = plt.plot(pk_phi_m, z, label=f"mode {m + 1}")
                plt.plot(krak_phi_m, z, "x", color=line[0].get_color())
                if u % 10 == 0:  # split into different figure for 10 failures
                    if len(failures) > u:
                        plt.legend(loc=2, handles=custom_lines)
                        plt.gca().invert_yaxis()
                        plt.figure()
                        plt.title(f"Env {env.stem}")

            print("1")
            plt.legend(handles=custom_lines)
            plt.gca().invert_yaxis()
            plt.show()
        raise AssertionError(
            f"""Larger error on modal depth function for env {env.stem} at modes {list(failures.keys())}.
            """
        )


if __name__ == ("__main__"):
    # test_kr_comp(env_test_files[4])
    test_phi_comp(env_test_files[5], True)
