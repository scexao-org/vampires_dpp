# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
import multiprocessing as mp
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import biweight_location
from astropy.time import Time
from tqdm.auto import tqdm

from vampires_dpp.constants import DEFAULT_NPROC, SUBARU_LOC
from vampires_dpp.headers import fix_header
from vampires_dpp.image_processing import (
    collapse_cube,
    collapse_frames_files,
    correct_distortion_cube,
)
from vampires_dpp.util import get_paths
from vampires_dpp.wcs import apply_wcs, get_coord_header


def filter_empty_frames(cube):
    finite_mask = np.isfinite(cube)
    nonzero_mask = cube != 0
    combined = finite_mask & nonzero_mask
    inds = np.any(combined, axis=(1, 2))
    return cube[inds]


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


def sort_calib_files(filenames: List[PathLike]) -> Dict[Tuple, Path]:
    darks_dict = {}
    for filename in filenames:
        path = Path(filename)
        header = fits.getheader(path)
        key = (header["U_CAMERA"], header["U_EMGAIN"], header["U_AQTINT"])
        if key in darks_dict:
            darks_dict[key].append(path)
        else:
            darks_dict[key] = [path]
    return darks_dict


def make_master_dark(
    filenames: List[PathLike],
    collapse: str = "median",
    name: str = "master_dark",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> List[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    # make darks for each camera
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master darks and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            cam, gain, exptime = key
            outname = outdir / f"{name}_em{gain:.0f}_{exptime:09.0f}us_cam{cam:.0f}.fits"
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each dark
            for path in filelist:
                kwds = dict(
                    output_directory=path.parent.parent / "collapsed",
                    force=force,
                    method=collapse,
                )
                jobs.append(pool.apply_async(make_dark_file, args=(path,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Collapsing dark frames")
        frames = [job.get() for job in iter]
        # create master frames from collapsed files
        collapsed_inputs = sort_calib_files(frames)
        jobs = []
        for key, filelist in collapsed_inputs.items():
            kwds = dict(
                output=outnames[key],
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(collapse_frames_files, args=(filelist,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Making master darks")
        [job.get() for job in iter]

    return list(outnames.values())


def make_master_flat(
    filenames: List[PathLike],
    master_darks: Optional[List[PathLike]] = None,
    collapse: str = "median",
    name: str = "master_flat",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> List[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    master_dark_inputs = {key: None for key in file_inputs.keys()}
    if master_darks is not None:
        inputs = sort_calib_files(master_darks)
        for key in file_inputs.keys():
            master_dark_inputs[key] = inputs.get(key, None)
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master darks and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            cam, gain, exptime = key
            outname = outdir / f"{name}_em{gain:.0f}_{exptime:09.0f}us_cam{cam:.0f}.fits"
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each flat
            for path in filelist:
                kwds = dict(
                    output_directory=path.parent.parent / "collapsed",
                    dark_filename=master_dark_inputs[key],
                    force=force,
                    method=collapse,
                )
                jobs.append(pool.apply_async(make_flat_file, args=(path,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Collapsing flat frames")
        frames = [job.get() for job in iter]
        # create master frames from collapsed files
        collapsed_inputs = sort_calib_files(frames)
        jobs = []
        for key, filelist in collapsed_inputs.items():
            kwds = dict(
                output=outnames[key],
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(collapse_frames_files, args=(filelist,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Making master flats")
        [job.get() for job in iter]

    return outnames
