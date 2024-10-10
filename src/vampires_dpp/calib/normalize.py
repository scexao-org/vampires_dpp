# library functions for common calibration tasks like
# background subtraction, collapsing cubes

import warnings

import numpy as np
from astropy.io import fits
from loguru import logger

from vampires_dpp.headers import fix_header, sort_header
from vampires_dpp.paths import get_paths
from vampires_dpp.util import load_fits, load_fits_header

__all__ = ("deinterleave_cube", "filter_empty_frames", "normalize_file")


def deinterleave_cube(
    data: np.ndarray, header: fits.Header, discard_empty: bool = True
) -> tuple[fits.PrimaryHDU | None, fits.PrimaryHDU | None]:
    flc1_filt = data[::2]
    if discard_empty:
        flc1_filt = filter_empty_frames(flc1_filt)
    hdu1 = None
    if flc1_filt is not None:
        header = fix_header(header.copy())
        header["NAXIS3"] = len(flc1_filt)
        header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        header["U_FLC"] = "A", "VAMPIRES FLC polarization state"
        header = sort_header(header)
        hdu1 = fits.PrimaryHDU(flc1_filt, header=header)

    flc2_filt = data[1::2]
    if discard_empty:
        flc2_filt = filter_empty_frames(flc2_filt)
    hdu2 = None
    if flc2_filt is not None:
        header = fix_header(header.copy())
        header["NAXIS3"] = len(flc2_filt)
        header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
        header["U_FLC"] = "B", "VAMPIRES FLC polarization state"
        header = sort_header(header)
        hdu2 = fits.PrimaryHDU(flc2_filt, header=header)

    return hdu1, hdu2


def filter_empty_frames(cube) -> np.ndarray | None:
    finite_mask = np.isfinite(cube)
    nonzero_mask = cube != 0
    combined = finite_mask & nonzero_mask
    inds = np.any(combined, axis=(-2, -1))
    if not np.any(inds):
        return None

    return cube[inds]


def normalize_file(filename: str, deinterleave: bool = False, discard_empty: bool = True, **kwargs):
    if deinterleave:
        # determine if files already exist
        path, outpath1 = get_paths(filename, suffix="FLC1_fix", **kwargs)
        _, outpath2 = get_paths(filename, suffix="FLC2_fix", **kwargs)
        if outpath1.exists() and outpath2.exists():
            return outpath1, outpath2
    else:
        path, outpath = get_paths(filename, suffix="fix", **kwargs)
        if outpath.exists():
            return outpath
    logger.debug(f"Loading {path} for normalization")
    data, header = load_fits(path, header=True)
    # Danger! you've encountered an AncientHeader^TM
    if "ACQNCYCS" in header:
        msg = "WARNING: Unsupported VAMPIRES data type found, cannot guarantee all operations will be successful"
        warnings.warn(msg, stacklevel=2)
        header2 = load_fits_header(path, ext=1)
        header.update(header2)
    # determine how many frames to discard
    ndiscard = 0 if "U_FLCSTT" in header else 2
    data_filt = data[ndiscard:]
    if deinterleave:
        hdu1, hdu2 = deinterleave_cube(data_filt, header, discard_empty=discard_empty)
        if hdu1 is not None:
            hdu1.writeto(outpath1, overwrite=True)
        if hdu2 is not None:
            hdu2.writeto(outpath2, overwrite=True)
    else:
        if discard_empty:
            data_filt = filter_empty_frames(data_filt)
        if data_filt is not None:
            header["NAXIS3"] = len(data_filt)
            header = sort_header(fix_header(header))
            fits.writeto(outpath, data_filt, header=header, overwrite=True)
