import functools
import warnings
from pathlib import Path
from typing import Literal, TypeAlias

import bottleneck as bn
import numpy as np
import tqdm.auto as tqdm
from astropy.io import fits
from astropy.stats import biweight_location
from numpy.typing import NDArray

from vampires_dpp.headers import fix_header, sort_header
from vampires_dpp.util import load_fits

from .combine_frames import combine_frames
from .image_processing import derotate_cube
from .paths import any_file_newer, get_paths

CoaddMethod: TypeAlias = Literal["median", "mean", "varmean", "biweight"]


def coadd_hdul(hdul: fits.HDUList, *, method: CoaddMethod = "median") -> fits.HDUList:
    # collapse data along time axis
    ncoadd = hdul[0].shape[0]
    coll_data, _ = collapse_cube(hdul[0].data, method=method)
    coll_err = np.sqrt(bn.nansum(hdul["ERR"].data ** 2, axis=0)) / ncoadd
    # coll_err = collapse_cube(hdul["ERR"].data, method=method)
    output_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(coll_data, header=hdul[0].header),
            fits.ImageHDU(coll_err, header=hdul["ERR"].header),
        ]
    )
    output_hdul.extend(hdul[2:])

    # header info
    info = fits.Header()
    info["hierarch DPP COADD METHOD"] = method, "Coadd method"
    info["hierarch DPP COADD NCOADD"] = ncoadd, "Number of coadded frames"
    info["hierarch DPP COADD TINT"] = (
        hdul[0].header["EXPTIME"] * ncoadd,
        "[s] total integrated exposure time",
    )

    for hdu in output_hdul:
        hdu.header.update(info)

    return output_hdul


def weighted_collapse(data: NDArray, angles: NDArray, **kwargs) -> NDArray:
    """Do a variance-weighted simultaneous derotation and collapse of ADI data based on the algorithm presented in `Bottom 2017 <https://ui.adsabs.harvard.edu/abs/2017RNAAS...1...30B>`_.

    Parameters
    ----------
    data : NDArray
        3D cube
    angles : NDArray
        Vector of angles to derotate data cube by

    Returns
    -------
    NDArray
        2D derotated and collapsed frame
    """
    variance_frame = np.var(data, axis=0, keepdims=True)

    # if the variance is zero, return the mean
    if np.allclose(variance_frame, 0):
        derotated = derotate_cube(data, angles, **kwargs)
        return bn.nanmean(derotated, 0)

    # expand the variance frame into a cube
    variance_cube = np.repeat(variance_frame, data.shape[0], axis=0)
    # derotate both signal and variance
    derotated_data = derotate_cube(data, angles, **kwargs)
    derotated_variance = derotate_cube(variance_cube, angles, **kwargs)
    derotated_variance[derotated_variance == 0] = np.inf
    # calculate weighted sum
    numer = bn.nansum(derotated_data / derotated_variance, axis=0)
    denom = bn.nansum(1 / derotated_variance, axis=0)
    weighted_frame = numer / denom
    return weighted_frame


def varmean(cube):
    weights = 1 / bn.nanvar(cube, axis=(1, 2), keepdims=True)
    return bn.nansum(cube * weights, axis=0) / bn.nansum(weights)


def collapse_cube(
    cube: NDArray, method: str = "median", header: fits.Header | None = None, **kwargs
) -> tuple[NDArray, fits.Header | None]:
    """Collapse a cube along its time axis

    Parameters
    ----------
    cube : NDArray
        3D cube
    method : str, optional
        One of "median", "mean", "varmean", "biweight", by default "median"
    header : fits.Header, optional
        FITS header, which will be updated with metadata if provided. By default None

    Returns
    -------
    Tuple[NDArray, Optional[fits.Header]]
        Tuple of collapsed frame and updated header. If header is not provided, will be None.
    """
    # clean inputs
    match method.strip().lower():
        case "median":
            collapse_func = functools.partial(bn.nanmedian, axis=0)
        case "mean":
            collapse_func = functools.partial(bn.nanmean, axis=0)
        case "varmean":
            collapse_func = varmean
        case "biweight":
            collapse_func = functools.partial(biweight_location, axis=0, c=6, ignore_nan=True)

    # suppress all-nan axis warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = collapse_func(cube)

    if header is not None:
        header["hierarch DPP COL METH"] = method, "DPP cube collapse method"
        header["hierarch DPP NCOADD"] = cube.shape[0], "Num. of coadded frames in cube"

    return frame, header


def collapse_cube_file(filename, force: bool = False, **kwargs) -> Path:
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = load_fits(path, header=True)
    frame, header = collapse_cube(cube, header=header, **kwargs)

    fits.writeto(outpath, frame, header=sort_header(header), overwrite=True)
    return outpath


def collapse_frames(frames, headers=None, method="median", **kwargs):
    cube, header = combine_frames(frames, headers=headers, **kwargs)
    return collapse_cube(cube, method=method, header=header)


def collapse_frames_files(
    filenames, output, force=False, cubes=False, quiet=True, fix=False, **kwargs
):
    path = Path(output)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path

    frames = []
    headers = []
    _iter = tqdm.tqdm(filenames, "Collecting files") if not quiet else filenames
    for filename in _iter:
        # use memmap=False to avoid "too many files open" effects
        # another way would be to set ulimit -n <MAX_FILES>
        frame, header = load_fits(filename, header=True, memmap=False)
        if fix:
            header = fix_header(header)
        if cubes:
            rand_idx = np.random.default_rng().integers(low=0, high=len(frame))
            frame = frame[rand_idx]
        frames.append(frame)
        headers.append(header)

    frame, header = collapse_frames(frames, headers=headers, **kwargs)
    fits.writeto(path, frame, header=sort_header(header), overwrite=True)
    return path
