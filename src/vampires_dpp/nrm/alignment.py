from pathlib import Path
from typing import Any, Literal

import amical
from astropy.io import fits


def get_amical_parameters(header: fits.Header) -> dict[str, Any]:
    params = {}
    match header["U_MASK"]:
        case "SAM-18" | "SAM-18-Nudged":
            params["maskname"] = "g18"
            params["fw_splodge"] = 0.7
            params["hole_diam"] = 0.162
            params["cutoff"] = 8e-2
            params["n_wl"] = 3

    # filter is always 750-50 because uv scaling is done manually
    params["filtname"] = "750-50"
    # data is already flipped in calibration
    params["fliplr"] = False
    params["instrum"] = "VAMPIRES"
    return params


def check_mask_align(
    cube,
    params: dict[str, Any],
    save_path: str,
    method: Literal["square", "gauss"] = "square",
    theta: float = 0,
    uv: float = 1,
) -> None:
    """
    inputs are:
    cube - data, cropped, pre-processed
    params - dictionary of parameters for amical
    save_mask_align - path to save mask alignment pictures to
    theta - adjustable parameter
    uv - adjustable parameter
    """
    infos = amical.mf_pipeline.bispect._check_input_infos(
        {}, targetname=None, filtname=params["filtname"], instrum=params["instrum"], verbose=False
    )

    infos["instrument"] = params["instrum"]
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
        peakmethod=method,
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
        method,
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


def remove_padding(cube):
    ...


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
        params = get_amical_parameters(header)
        save_name = str(save_path / header["FIELD"]) + "_"
        theta = 97
        uv = 1.06
        check_mask_align(cube, params, save_path=save_name, uv=uv, theta=theta)
