from pathlib import Path
from typing import Any

import amical
from astropy.io import fits

from vampires_dpp.specphot.filters import determine_primary_filter


def get_amical_parameters(header: fits.Header) -> dict[str, Any]:
    params = {}
    match header["U_MASK"]:
        case "SAM-18" | "SAM-18-Nudged":
            params["maskname"] = "g18"
            params["fw_splodge"] = 0.7
            params["hole_diam"] = 0.162
            params["cutoff"] = 8e-2
            params["n_wl"] = 3

    params["filtname"] = determine_primary_filter(header)
    params["fliplr"] = True
    params["instrum"] = "VAMPIRES"
    params["peakmethod"] = "square"
    return params


def check_mask_align(cube, params: dict[str, Any], save_path: str, theta: float, uv: float) -> None:
    """
    inputs are:
    cube - data, cropped, pre-processed
    params - dictionary of parameters for amical
    save_mask_align - path to save mask alignment pictures to
    theta - adjustable parameter
    uv - adjustable parameter
    """
    params["filtname"] = "750-50"
    infos = amical.mf_pipeline.bispect._check_input_infos(
        {}, targetname=None, filtname=params["filtname"], instrum=params["instrum"], verbose=False
    )

    infos["instrument"] = "VAMPIRES"
    ft_arr, _, npix = amical.mf_pipeline.bispect._construct_ft_arr(cube)

    try:
        n_holes = len(amical.mf_pipeline.bispect.get_mask(infos.instrument, params["maskname"]))
    except TypeError:
        msg = "Could not get mask from AMICAL"
        print(msg)
        return

    index_mask = amical.mf_pipeline.bispect.compute_index_mask(n_holes)
    n_baselines = index_mask.n_baselines

    mf = amical.mf_pipeline.bispect.make_mf(
        params["maskname"],
        infos.instrument,
        infos.filtname,
        npix,
        peakmethod=params["peakmethod"],
        fw_splodge=params["fw_splodge"],
        n_wl=params["n_wl"],
        cutoff=params["cutoff"],
        hole_diam=params["hole_diam"],
        scaling=uv,
        theta_detector=theta,
        i_wl=False,
        fliplr=params["fliplr"],
        display=False,
    )

    if mf is None:
        msg = "Could not make mf in AMICAL"
        print(msg)
        return

    ft_arr = np.fft.fftshift(ft_arr)

    amical.mf_pipeline.bispect._show_ft_arr_peak(
        ft_arr,
        n_baselines,
        mf,
        params["maskname"],
        params["peakmethod"],
        0,
        aver=False,
        centred=True,
        size=20,
        norm=None,
        alpha=1,
        vmin=None,
        vmax=None,
        log_stretch=True,
        savepath=save_path,
    )


if __name__ == "__main__":
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    save_path = Path("nrm_output")
    save_path.mkdir(exist_ok=True)

    hdul = fits.open(args.filename)
    data_cube = hdul[0].data
    for idx in range(len(hdul) - 2):
        cube = data_cube[idx]
        header = hdul[idx + 2].header
        data = np.nan_to_num(cube)
        params = get_amical_parameters(header)
        save_name = str(save_path / params["filtname"]) + "_"
        theta = 97
        uv = 1.06
        check_mask_align(data, params, save_path=save_name, uv=uv, theta=theta)
