from typing import Any, Literal

import amical
import numpy as np
import tomli


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


def get_uv_theta(basepath, name, logger):
    uv_thetas = {}
    for key in ("cam1", "cam2"):
        path = basepath / f"{name}_uv_theta_{key}.toml"
        if not path.exists():
            logger.warning(
                f"Could not locate mask parameter file for {key}, expected it to be at {path}. Using center of image as default."
            )
            continue
        with path.open("rb") as fh:
            payload = tomli.load(fh)
        uv_thetas[key] = payload

        logger.debug(f"{key} uv = {uv_thetas[key]['uv']}")
        logger.debug(f"{key} theta = {uv_thetas[key]['theta']}")
    return uv_thetas
