from pathlib import Path
from typing import Final, Literal, Optional

import numpy as np
import scipy.stats as st
from astropy.io import fits
from numpy.typing import ArrayLike

from .image_processing import collapse_cube, pad_cube, shift_cube
from .indexing import frame_center

FRAME_SELECT_MAP: Final[dict[str, str]] = {
    "peak": "max",
    "strehl": "strel",
    "normvar": "nvar",
    "var": "var",
    "model": "mod_a",
    "fwhm": "mod_w",
}


def get_frame_select_mask(metrics, input_key, quantile=0):
    frame_select_key = FRAME_SELECT_MAP[input_key]
    # create masked metrics
    values = metrics[frame_select_key]
    cutoff = np.nanquantile(values, quantile, axis=1, keepdims=True)
    if input_key == "fwhm":
        return np.all(values < cutoff, axis=0)
    else:
        return np.all(values > cutoff, axis=0)


REGISTER_MAP: Final[dict[str, str]] = {
    "com": "com",
    "peak": "pk",
    "model": "mod",
}


def get_centroids_from(metrics, input_key):
    register_key = REGISTER_MAP[input_key]
    cx = metrics[f"{register_key}_x"]
    cy = metrics[f"{register_key}_y"]
    # if there are values from multiple PSFs (e.g. satspots)
    # take the mean centroid of each PSF
    if cx.ndim == 2:
        cx = np.mean(cx, axis=0)
    if cy.ndim == 2:
        cy = np.mean(cy, axis=0)
    centroids = np.column_stack((cy, cx))
    return centroids


def lucky_image_file(
    hdu,
    outpath,
    method: str = "median",
    register: Optional[Literal["com", "peak", "model"]] = "com",
    frame_select: Optional[Literal["peak", "normvar", "var", "strehl", "model", "fwhm"]] = None,
    metric_file: Optional[Path] = None,
    force: bool = False,
    select_cutoff: float = 0,
    **kwargs,
) -> Path:
    if (
        not force
        and outpath.is_file()
        and metric_file
        and Path(metric_file).stat().st_mtime < outpath.stat().st_mtime
    ):
        with fits.open(outpath) as hdul:
            return hdul[0]

    cube = hdu.data
    header = hdu.header

    # if no metric file assume straight collapsing
    if not metric_file:
        frame, header = collapse_cube(cube, method=method, header=header, **kwargs)
        hdu = fits.PrimaryHDU(frame, header=header)
        hdu.writeto(outpath, overwrite=True)
        return hdu

    metrics = np.load(metric_file)
    # mask metrics
    masked_metrics = metrics
    if frame_select and select_cutoff > 0:
        mask = get_frame_select_mask(metrics, input_key=frame_select, quantile=select_cutoff)
        masked_metrics = {key: metric[..., mask] for key, metric in metrics.items()}
        # frame select this cube
        cube = cube[mask]

    if register:
        centroids = get_centroids_from(masked_metrics, input_key=register)
        # print(centroids)
        offsets = centroids - frame_center(cube)
        # determine maximum padding, with sqrt(2)
        # for radial coverage
        rad_factor = (np.sqrt(2) - 1) * np.max(cube.shape[-2:]) / 2
        npad = np.ceil(rad_factor + 1).astype(int)
        cube_padded, header = pad_cube(cube, npad, header=header)
        cube = shift_cube(cube_padded, -offsets, **kwargs)

    ## Step 3: Collapse
    frame, header = collapse_cube(cube, method=method, header=header, **kwargs)
    header = add_metrics_to_header(header, masked_metrics)
    hdu = fits.PrimaryHDU(frame, header=header)
    hdu.writeto(outpath, overwrite=True)
    return hdu


COMMENT_FSTRS: Final[dict[str, str]] = {
    "max": "[adu] Peak signal{}measured in window {}",
    "sum": "[adu] Total signal{}measured in window {}",
    "mean": "[adu] Mean signal{}measured in window {}",
    "med": "[adu] Median signal{}measured in window {}",
    "var": "[adu^2] Signal variance{}measured in window {}",
    "nvar": "[adu] Normalized variance{}(var / mean) measured in window {}",
    "photr": "[px] Radius of photometric aperture",
    "photf": "[adu] Photometric flux{}in window {}",
    # "phote": "[adu] Photometric flux error in window {}",
    "com_x": "[px] Center-of-mass{}along x axis in window {}",
    "com_y": "[px] Center-of-mass{}along y axis in window {}",
    "pk_x": "[px] Peak signal index{}along x axis in window {}",
    "pk_y": "[px] Peak signal index{}along y axis in window {}",
    "psfmod": "Model used for PSF fitting",
    "mod_x": "[px] Model position fit{}along x axis in window {}",
    "mod_y": "[px] Model position fit{}along y axis in window {}",
    "mod_w": "[px] Model FWHM fit{}in window {}",
    "mod_a": "[adu] Model amplitude fit{}in window {}",
}


def add_metrics_to_header(hdr: fits.Header, metrics: dict) -> fits.Header:
    for key, arr in metrics.items():
        if key not in COMMENT_FSTRS:
            continue
        key_up = key.upper()
        if key_up in ("PSFMOD", "PHOTR"):
            hdr[key_up] = arr[0][0], COMMENT_FSTRS[key]
            continue
        for i, psf in enumerate(arr):
            comment = COMMENT_FSTRS[key].format(" ", i)
            hdr[f"{key_up}{i}"] = psf.mean(), comment
            err_comment = COMMENT_FSTRS[key].format(" error ", i)
            hdr[f"{key_up[:5]}ER{i}"] = st.sem(psf, nan_policy="omit"), err_comment
    return hdr
