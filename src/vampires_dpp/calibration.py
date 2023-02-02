# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import biweight_location
from astropy.time import Time
from numpy.typing import ArrayLike

from vampires_dpp.constants import SUBARU_LOC
from vampires_dpp.headers import fix_header
from vampires_dpp.image_processing import collapse_cube, correct_distortion_cube
from vampires_dpp.util import get_paths
from vampires_dpp.wcs import apply_wcs, get_coord_header


def calibrate_file(
    filename: str,
    hdu: int = 0,
    dark_filename: Optional[str] = None,
    flat_filename: Optional[str] = None,
    transform_filename: Optional[str] = None,
    force: bool = False,
    deinterleave: bool = False,
    coord: Optional[SkyCoord] = None,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file():
        return outpath
    header = fits.getheader(path, hdu)
    # have to also check if deinterleaving
    if deinterleave:
        outpath_FLC1 = outpath.with_stem(f"{outpath.stem}_FLC1")
        outpath_FLC2 = outpath.with_stem(f"{outpath.stem}_FLC2")
        if not force and outpath_FLC1.is_file() and outpath_FLC2.is_file():
            return outpath_FLC1, outpath_FLC2

    raw_cube = fits.getdata(path, hdu)
    # fix header
    header = apply_wcs(fix_header(header))
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    if coord is None:
        coord_now = get_coord_header(header, time)
    else:
        coord_now = coord.apply_space_motion(time)

    header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":")
    header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":")
    # Discard frames in OG VAMPIRES
    if "U_FLCSTT" in header:
        deinterleave = False
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[2:].astype("f4")
    # remove empty and NaN frames
    cube = filter_empty_frames(cube)
    # dark correction
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        dark = fits.getdata(dark_path)
        cube -= dark
        header["MDARK"] = dark_path.name, "DPP master dark filename"
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        flat = fits.getdata(flat_path)
        cube /= flat
        header["MFLAT"] = flat_path.name, "DPP master flat filename"
    # flip cam 1 data
    if header["U_CAMERA"] == 1:
        cube = np.flip(cube, axis=-2)
    # distortion correction
    if transform_filename is not None:
        transform_path = Path(transform_filename)
        distort_coeffs = pd.read_csv(transform_path, index_col=0)
        header["MDIST"] = transform_path.name, "DPP distortion transform filename"
        params = distort_coeffs.loc[f"cam{header['U_CAMERA']}"]
        cube, header = correct_distortion_cube(cube, *params, header=header)
    # deinterleave
    if deinterleave:
        header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        header["RET-ANG2"] = 0, "Position angle of second retarder plate (deg)"
        header["RETPLAT2"] = "FLC(VAMPIRES)", "Identifier of second retarder plate"
        fits.writeto(outpath_FLC1, cube[::2], header, overwrite=True)

        header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
        header["RET-ANG2"] = 45, "Position angle of second retarder plate (deg)"
        fits.writeto(outpath_FLC2, cube[1::2], header, overwrite=True)
        return outpath_FLC1, outpath_FLC2

    fits.writeto(outpath, cube, header, overwrite=True)
    return outpath


def make_dark_file(filename: str, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file():
        return outpath
    raw_cube, header = fits.getdata(path, ext=0, header=True)
    if "U_FLCSTT" in header:
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[1:].astype("f4")
    master_dark, header = collapse_cube(cube, header=header, **kwargs)
    fits.writeto(outpath, master_dark, header=header, overwrite=True)
    return outpath


def make_flat_file(filename: str, force=False, dark_filename=None, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file():
        return outpath
    raw_cube, header = fits.getdata(path, ext=0, header=True)
    if "U_FLCSTT" in header:
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[1:].astype("f4")
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        header["MDARK"] = (dark_path.name, "file used for dark subtraction")
        master_dark = fits.getdata(dark_path)
        cube = cube - master_dark
    master_flat, header = collapse_cube(cube, header=header, **kwargs)
    master_flat = master_flat / biweight_location(master_flat, c=6, ignore_nan=True)

    fits.writeto(outpath, master_flat, header=header, overwrite=True)
    return outpath


def filter_empty_frames(cube):
    finite_mask = np.isfinite(cube)
    nonzero_mask = cube != 0
    combined = finite_mask & nonzero_mask
    inds = np.any(combined, axis=(1, 2))
    return cube[inds]
