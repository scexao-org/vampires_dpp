# library functions for common calibration tasks like
# background subtraction, collapsing cubes
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
from vampires_dpp.headers import parallactic_angle
from vampires_dpp.image_processing import (
    collapse_cube,
    collapse_frames_files,
    correct_distortion_cube,
)
from vampires_dpp.util import get_paths, wrap_angle
from vampires_dpp.wcs import apply_wcs, get_coord_header

__all__ = [
    "calibrate_file",
    "make_background_file",
    "make_flat_file",
    "make_master_background",
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
    back_filename: Optional[str] = None,
    flat_filename: Optional[str] = None,
    transform_filename: Optional[str] = None,
    force: bool = False,
    bpfix: bool = False,
    coord: Optional[SkyCoord] = None,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    raw_cube, header = fits.getdata(path, ext=hdu, header=True)
    cube = raw_cube.astype("f4")
    # fix header
    time_str = Time(header["MJD-STR"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time_end = Time(header["MJD-END"], format="mjd", scale="ut1", location=SUBARU_LOC)
    if coord is None:
        coord_now = get_coord_header(header, time)
    else:
        coord_now = coord.apply_space_motion(time)
    for _time, _key in zip((time_str, time_end), ("STR", "END")):
        if coord is None:
            _coord = get_coord_header(header, _time)
        else:
            _coord = coord.apply_space_motion(_time)
        pa = parallactic_angle(_time, _coord)
        header[f"PA-{_key}"] = pa, "[deg] parallactic angle of target"

    header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":")
    header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":")
    pa = parallactic_angle(time, coord_now)
    header["PA"] = pa, "[deg] parallactic angle of target"
    parang = wrap_angle(pa + PA_OFFSET)
    header["PARANG"] = parang, "[deg] derotation angle for North up"
    header = apply_wcs(header, parang=parang)

    # remove empty and NaN frames
    cube = filter_empty_frames(cube)
    # background subtraction
    if back_filename is not None:
        back_path = Path(back_filename)
        header["DPP_BACK"] = back_path.name
        background = fits.getdata(back_path).astype("f4")
        cube -= background
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        header["DPP_FLAT"] = flat_path.name
        flat = fits.getdata(flat_path).astype("f4")
        cube /= flat
    # bad pixel correction
    if bpfix:
        mask, _ = detect_cosmics(
            cube.mean(0),
            gain=header.get("GAIN", 0.11),
            readnoise=READNOISE[header["U_DETMOD"].lower()],
            satlevel=2**15 * header.get("GAIN", 0.11),
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

    fits.writeto(outpath, cube, header=header, overwrite=True)
    return outpath


def make_background_file(filename: str, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    cube, header = fits.getdata(
        path,
        ext=0,
        header=True,
    )
    master_background, header = collapse_cube(cube, header=header, **kwargs)
    header["CAL_TYPE"] = "BACKGROUND", "DPP calibration file type"
    bkg = np.full_like(master_background, header.get("BIAS", 200))
    _, clean_background = detect_cosmics(
        master_background,
        inbkg=bkg,
        gain=header.get("GAIN", 0.11),
        readnoise=READNOISE[header["U_DETMOD"].lower()],
        satlevel=2**16 * header.get("GAIN", 0.11),
    )
    fits.writeto(
        outpath,
        clean_background,
        header=header,
        overwrite=True,
    )
    return outpath


def make_flat_file(filename: str, force=False, back_filename=None, **kwargs):
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    cube, header = fits.getdata(
        path,
        ext=0,
        header=True,
    )
    header["CAL_TYPE"] = "FLATFIELD", "DPP calibration file type"
    if back_filename is not None:
        back_path = Path(back_filename)
        header["DPP_BACK"] = (back_path.name, "DPP file used for background subtraction")
        master_back = fits.getdata(
            back_path,
        )
        cube = cube - master_back
    master_flat, header = collapse_cube(cube, header=header, **kwargs)
    bkg = np.full_like(master_flat, header.get("BIAS", 200))
    _, clean_flat = detect_cosmics(
        master_flat,
        inbkg=bkg,
        gain=header.get("GAIN", 0.11),
        readnoise=READNOISE[header["U_DETMOD"].lower()],
        satlevel=2**16 * header.get("GAIN", 0.11),
    )
    clean_flat /= np.nanmedian(clean_flat)

    fits.writeto(
        outpath,
        clean_flat,
        header=header,
        overwrite=True,
    )
    return outpath


def sort_calib_files(filenames: list[PathLike], backgrounds=False, ext=0) -> dict[Tuple, Path]:
    file_dict = {}
    for filename in filenames:
        path = Path(filename)
        header = fits.getheader(path, ext=ext)
        sz = header["NAXIS1"], header["NAXIS2"]
        key = (header["U_CAMERA"], header["EXPTIME"], sz, header["FILTER01"], header["FILTER02"])
        if backgrounds:
            key = key[:-2]
        if key in file_dict:
            file_dict[key].append(path)
        else:
            file_dict[key] = [path]
    return file_dict


def make_master_background(
    filenames: list[PathLike],
    collapse: str = "median",
    name: str = "master_back",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames, backgrounds=True)
    # make backgrounds for each camera
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master backgrounds and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            cam, exptime, sz = key
            outname = (
                outdir / f"{name}_{exptime*1e6:09.0f}us_{sz[0]:04d}x{sz[1]:04d}_cam{cam:.0f}.fits"
            )
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each background
            for path in filelist:
                kwds = dict(
                    output_directory=outdir / "collapsed",
                    force=force,
                    method=collapse,
                )
                jobs.append(pool.apply_async(make_background_file, args=(path,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Collapsing background frames")
        frames = [job.get() for job in iter]
        # create master frames from collapsed files
        collapsed_inputs = sort_calib_files(frames, backgrounds=True)
        jobs = []
        for key, filelist in collapsed_inputs.items():
            kwds = dict(
                output=outnames[key],
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(collapse_frames_files, args=(filelist,), kwds=kwds))
        iter = jobs if quiet else tqdm(jobs, desc="Making master backgrounds")
        [job.get() for job in iter]

    return list(outnames.values())


def make_master_flat(
    filenames: list[PathLike],
    master_backgrounds: Optional[list[PathLike]] = None,
    collapse: str = "median",
    name: str = "master_flat",
    force: bool = False,
    output_directory: Optional[PathLike] = None,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    # prepare input filenames
    file_inputs = sort_calib_files(filenames)
    master_back_dict = {key: None for key in file_inputs.keys()}
    if master_backgrounds is not None:
        back_inputs = sort_calib_files(master_backgrounds, backs=True)
        for key in file_inputs.keys():
            # don't need to match filter for backs
            back_key = key[:-2]
            # input should be list with one file in it
            if back_key in back_inputs:
                master_back_dict[key] = back_inputs[back_key][0]

    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()

    # get names for master backs and remove
    # files from queue if they exist
    outnames = {}
    with mp.Pool(num_proc) as pool:
        jobs = []
        for key, filelist in file_inputs.items():
            cam, exptime, sz, filt1, filt2 = key
            outname = (
                outdir
                / f"{name}_{filt1}_{filt2}_{exptime*1e6:09.0f}us_{sz[0]:04d}x{sz[1]:04d}_cam{cam:.0f}.fits"
            )
            outnames[key] = outname
            if not force and outname.is_file():
                continue
            # collapse the files required for each flat
            for path in filelist:
                kwds = dict(
                    output_directory=outdir / "collapsed",
                    back_filename=master_back_dict[key],
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
