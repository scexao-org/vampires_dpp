# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
import multiprocessing as mp
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import biweight_location
from astropy.time import Time
from astroscrappy import detect_cosmics
from tqdm.auto import tqdm

from vampires_dpp.constants import DEFAULT_NPROC, PA_OFFSET, READNOISE, SUBARU_LOC
from vampires_dpp.headers import fix_header, parallactic_angle
from vampires_dpp.image_processing import (
    collapse_cube,
    collapse_frames_files,
    correct_distortion_cube,
)
from vampires_dpp.util import get_paths, wrap_angle
from vampires_dpp.wcs import apply_wcs, get_coord_header

__all__ = [
    "calibrate_file",
    "make_dark_file",
    "make_flat_file",
    "make_master_dark",
    "make_master_flat",
]


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
    bpfix: bool = False,
    deinterleave: bool = False,
    coord: Optional[SkyCoord] = None,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    # have to also check if deinterleaving
    if deinterleave:
        outpath_FLC1 = outpath.with_stem(f"{outpath.stem}_FLC1")
        outpath_FLC2 = outpath.with_stem(f"{outpath.stem}_FLC2")
        if not force and outpath_FLC1.is_file() and outpath_FLC2.is_file():
            return outpath_FLC1, outpath_FLC2

    raw_cube, header = fits.getdata(path, ext=hdu, header=True)
    # fix header
    header = fix_header(header)
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    if coord is None:
        coord_now = get_coord_header(header, time)
    else:
        coord_now = coord.apply_space_motion(time)

    header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":")
    header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":")
    parang = parallactic_angle(time, coord_now)
    header["PARANG"] = parang, "[deg] derotation angle for North up"
    header["PA"] = wrap_angle(parang - PA_OFFSET), "[deg] parallactic angle of target"
    header = apply_wcs(header, parang=parang)

    # Discard frames in OG VAMPIRES
    if "U_FLCSTT" in header:
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[2:].astype("f4")
    # remove empty and NaN frames
    cube = filter_empty_frames(cube)
    # dark correction
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        dark = fits.getdata(dark_path).astype("f4")
        cube -= dark
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        flat = fits.getdata(flat_path).astype("f4")
        cube /= flat
    # bad pixel correction
    if bpfix:
        mask, _ = detect_cosmics(
            cube.mean(0),
            gain=header.get("DETGAIN", 4.5),
            readnoise=READNOISE,
            satlevel=2**16 * header.get("DETGAIN", 4.5),
            niter=1,
        )
        cube_copy = cube.copy()
        for i in range(cube.shape[0]):
            smooth_im = cv2.medianBlur(cube_copy[i], 5)
            cube[i, mask] = smooth_im[mask]
    # flip cam 1 data
    if header["U_CAMERA"] == 1:
        cube = np.flip(cube, axis=-2)
    # distortion correction
    if transform_filename is not None:
        transform_path = Path(transform_filename)
        distort_coeffs = pd.read_csv(transform_path, index_col=0)
        params = distort_coeffs.loc[f"cam{header['U_CAMERA']:.0f}"]
        cube, header = correct_distortion_cube(cube, *params, header=header)

    # deinterleave
    if deinterleave:
        header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        header["U_FLCANG"] = 0, "VAMPIRES FLC angle (deg)"
        fits.writeto(
            outpath_FLC1,
            cube[::2],
            header=header,
            overwrite=True,
        )

        header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
        header["U_FLCANG"] = 45, "VAMPIRES FLC angle (deg)"

        fits.writeto(
            outpath_FLC2,
            cube[1::2],
            header=header,
            overwrite=True,
        )
        return outpath_FLC1, outpath_FLC2

    fits.writeto(outpath, cube, header=header, overwrite=True)
    return outpath


def make_dark_file(filename: str, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    raw_cube, header = fits.getdata(
        path,
        ext=0,
        header=True,
    )
    if "U_FLCSTT" in header:
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[1:].astype("f4")
    master_dark, header = collapse_cube(cube, header=header, **kwargs)
    _, clean_dark = detect_cosmics(
        master_dark,
        gain=header.get("DETGAIN", 4.5),
        readnoise=READNOISE,
        satlevel=2**16 * header.get("DETGAIN", 4.5),
    )
    fits.writeto(
        outpath,
        clean_dark,
        header=header,
        overwrite=True,
    )
    return outpath


def make_flat_file(filename: str, force=False, dark_filename=None, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    raw_cube, header = fits.getdata(
        path,
        ext=0,
        header=True,
    )
    if "U_FLCSTT" in header:
        cube = raw_cube.astype("f4")
    else:
        cube = raw_cube[1:].astype("f4")
    if dark_filename is not None:
        dark_path = Path(dark_filename)
        header["MDARK"] = (dark_path.name, "file used for dark subtraction")
        master_dark = fits.getdata(
            dark_path,
        )
        cube = cube - master_dark
    master_flat, header = collapse_cube(cube, header=header, **kwargs)
    _, clean_flat = detect_cosmics(
        master_flat,
        gain=header.get("DETGAIN", 4.5),
        readnoise=READNOISE,
        satlevel=2**16 * header.get("DETGAIN", 4.5),
    )
    clean_flat /= np.nanmedian(clean_flat)

    fits.writeto(
        outpath,
        clean_flat,
        header=header,
        overwrite=True,
    )
    return outpath


def sort_calib_files(filenames: list[PathLike]) -> dict[Tuple, Path]:
    darks_dict = {}
    for filename in filenames:
        path = Path(filename)
        header = fits.getheader(path)
        sz = header["NAXIS1"], header["NAXIS2"]
        key = (header["U_CAMERA"], header["U_EMGAIN"], header["U_AQTINT"], sz, header["U_FILTER"])
        if key in darks_dict:
            darks_dict[key].append(path)
        else:
            darks_dict[key] = [path]
    return darks_dict


def make_master_dark(
    filenames: list[PathLike],
    collapse: str = "median",
    name: str = "master_dark",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
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
            cam, gain, exptime, sz, filt = key
            outname = (
                outdir
                / f"{name}_em{gain:.0f}_{exptime/1e3:05.0f}ms_{sz[0]:03d}x{sz[1]:03d}_cam{cam:.0f}.fits"
            )
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
    filenames: list[PathLike],
    master_darks: Optional[list[PathLike]] = None,
    collapse: str = "median",
    name: str = "master_flat",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    master_dark_dict = {key: None for key in file_inputs.keys()}
    if master_darks is not None:
        dark_inputs = sort_calib_files(master_darks)
        for key in file_inputs.keys():
            # don't need to match filter for darks
            dark_key = key[:-1]
            # input should be list with one file in it
            if key in dark_inputs:
                master_dark_dict[key] = dark_inputs[dark_key][0]

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
            cam, gain, exptime, sz, filt = key
            outname = (
                outdir
                / f"{name}_{filt}_em{gain:.0f}_{exptime/1e3:05.0f}ms_{sz[0]:03d}x{sz[1]:03d}_cam{cam:.0f}.fits"
            )
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each flat
            for path in filelist:
                kwds = dict(
                    output_directory=path.parent.parent / "collapsed",
                    dark_filename=master_dark_dict[key],
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
