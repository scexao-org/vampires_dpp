# library functions for common calibration tasks like
# background subtraction, collapsing cubes
import functools
import itertools
import multiprocessing as mp
import os
from collections.abc import Iterable
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from loguru import logger
from tqdm.auto import tqdm

from .indexing import cutout_inds
from .organization import header_table

# need to safely import astroscrappy because it
# forces CPU affinity to drop to 0
try:
    _CORES = os.sched_getaffinity(0)

    os.sched_setaffinity(0, _CORES)
except AttributeError:
    # not on a UNIX system
    pass

from vampires_dpp.constants import DEFAULT_NPROC, SUBARU_LOC
from vampires_dpp.headers import fix_header, parallactic_angle
from vampires_dpp.image_processing import collapse_cube, correct_distortion_cube
from vampires_dpp.util import load_fits, load_fits_header, wrap_angle
from vampires_dpp.wcs import apply_wcs, get_coord_header

from .paths import get_paths


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
            fits.writeto(outpath, data_filt, header=fix_header(header), overwrite=True)


def filter_empty_frames(cube) -> np.ndarray | None:
    finite_mask = np.isfinite(cube)
    nonzero_mask = cube != 0
    combined = finite_mask & nonzero_mask
    inds = np.any(combined, axis=(-2, -1))
    if not np.any(inds):
        return None

    return cube[inds]


def deinterleave_cube(
    data: np.ndarray, header: fits.Header, discard_empty: bool = True
) -> tuple[fits.PrimaryHDU | None, fits.PrimaryHDU | None]:
    flc1_filt = data[::2]
    if discard_empty:
        flc1_filt = filter_empty_frames(flc1_filt)
    hdu1 = None
    if flc1_filt is not None:
        hdu1 = fits.PrimaryHDU(flc1_filt, header=header.copy())
        hdu1.header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        hdu1.header["U_FLC"] = "A"
        fix_header(hdu1.header)

    flc2_filt = data[1::2]
    if discard_empty:
        flc2_filt = filter_empty_frames(flc2_filt)
    hdu2 = None
    if flc2_filt is not None:
        hdu2 = fits.PrimaryHDU(flc2_filt, header=header.copy())
        hdu2.header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
        hdu2.header["U_FLC"] = "B"
        fix_header(hdu2.header)

    return hdu1, hdu2


def apply_coordinate(header, coord: SkyCoord | None = None):
    time_str = Time(header["MJD-STR"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time_end = Time(header["MJD-END"], format="mjd", scale="ut1", location=SUBARU_LOC)
    coord_now = get_coord_header(header, time) if coord is None else coord.apply_space_motion(time)
    for _time, _key in zip((time_str, time_end), ("STR", "END"), strict=True):
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
    derotang = wrap_angle(pa + header["PAOFFSET"])
    header["DEROTANG"] = derotang, "[deg] derotation angle for North up"
    return apply_wcs(header, angle=derotang)


def calibrate_file(
    filename: str,
    back_filename: str | None = None,
    flat_filename: str | None = None,
    transform_filename: str | None = None,
    force: bool = False,
    bpmask: bool = False,
    coord: SkyCoord | None = None,
    **kwargs,
) -> fits.HDUList:
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        with fits.open(outpath) as hdul:
            return hdul

    # load data and mask saturated pixels
    raw_cube, header = load_fits(path, header=True)
    header = fix_header(header)
    # mask values above saturation
    satlevel = header["FULLWELL"] / header["GAIN"]
    cube = np.where(raw_cube >= satlevel, np.nan, raw_cube.astype("f4"))
    # apply proper motion correction to coordinate
    header = apply_coordinate(header, coord)
    cube_err = np.zeros_like(cube)
    # background subtraction
    if back_filename is not None:
        back_path = Path(back_filename)
        header["BACKFILE"] = back_path.name
        with fits.open(back_path) as hdul:
            background = hdul[0].data.astype("f4")
            back_hdr = hdul[0].header
            back_err = hdul["ERR"].data.astype("f4")
            header["NOISEADU"] = back_hdr["NOISEADU"], back_hdr.comments["NOISEADU"]
            header["NOISE"] = back_hdr["NOISE"], back_hdr.comments["NOISEADU"]
        cube -= np.where(np.isnan(background), header["BIAS"], background)
    else:
        back_err = 0
    cube_err = np.sqrt(np.maximum(cube / header["EFFGAIN"], 0) * header["ENF"] ** 2 + back_err**2)
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        header["FLATFILE"] = flat_path.name
        with fits.open(flat_path) as hdul:
            flat = hdul[0].data.astype("f4")
            flat_hdr = hdul[0].header
            flat[flat == 0] = np.nan
            flat_err = hdul["ERR"].data.astype("f4")
            header["FLATNORM"] = flat_hdr["FLATNORM"], flat_hdr.comments["FLATNORM"]

        unnorm_cube = cube.copy()
        unnorm_cube[unnorm_cube == 0] = np.nan
        rel_err = cube_err / unnorm_cube
        rel_flat_err = flat_err / flat

        cube /= flat
        cube_err = np.abs(cube) * np.hypot(rel_err, rel_flat_err)
    # bad pixel correction
    if bpmask:
        mask = adaptive_sigma_clip_mask(cube)
        cube[mask] = np.nan
        cube_err[mask] = np.nan

    # flip cam 1 data on y-axis
    if header["U_CAMERA"] == 1:
        cube = np.flip(cube, axis=-2)
        cube_err = np.flip(cube_err, axis=-2)
    # distortion correction
    # TODO tbh this is scuffed
    if transform_filename is not None:
        transform_path = Path(transform_filename)
        distort_coeffs = pd.read_csv(transform_path, index_col=0)
        params = distort_coeffs.loc[f"cam{header['U_CAMERA']:.0f}"]
        cube, header = correct_distortion_cube(cube, *params, header=header)
        cube_err, _ = correct_distortion_cube(cube_err, *params)

    # clip fot float32 to limit data size
    prim_hdu = fits.PrimaryHDU(cube.astype("f4"), header=header)
    err_hdu = fits.ImageHDU(cube_err.astype("f4"), header=header, name="ERR")
    return fits.HDUList([prim_hdu, err_hdu])


def make_background_file(filename: str, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="coll", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    raw_cube, raw_header = load_fits(path, header=True)
    header = fix_header(raw_header)
    # mask saturated pixels
    satlevel = header["FULLWELL"] / header["GAIN"]
    cube = np.where(raw_cube >= satlevel, np.nan, raw_cube.astype("f4"))
    # collapse cube (median by default)
    if cube.shape[0] > 1:
        master_background, header = collapse_cube(cube, header=header, **kwargs)
        back_std = np.nanstd(cube, axis=0)
        back_err = np.hypot(back_std, back_std / np.sqrt(cube.shape[0]))
    else:
        master_background = cube[0]
        bkgnoise = np.sqrt(
            header["DC"] * header["EXPTIME"] * header["ENF"] ** 2 + header["RN"] ** 2
        )
        back_std = np.full_like(master_background, bkgnoise / header["EFFGAIN"])
        back_err = back_std

    noise = np.nanmedian(back_std)
    header["NOISEADU"] = noise, "[adu] median noise in background file"
    header["NOISE"] = noise * header["GAIN"], "[e-] median noise in background file"
    header["CALTYPE"] = "BACKGROUND", "DPP calibration file type"
    # get bad pixel mask from adaptive sigma clip
    bpmask = adaptive_sigma_clip_mask(master_background)
    # get bad pixel mask using lacosmic
    # bpmask, _ = detect_cosmics(
    #     master_background,
    #     invar=back_std**2,
    #     inmask=np.isnan(master_background),
    #     satlevel=satlevel,
    #     sigclip=10,
    #     gain=header["EFFGAIN"]
    # )
    # mask out bad pixels with nan- use np.isnan
    # to extract mask in future
    master_background[bpmask] = np.nan
    back_err[bpmask] = np.nan
    # save to disk
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(master_background, header=header),
            fits.ImageHDU(back_err, header=header, name="ERR"),
        ]
    )
    hdul.writeto(outpath, overwrite=True)
    return outpath


def make_flat_file(filename: str, force=False, back_filename=None, **kwargs):
    path, outpath = get_paths(filename, suffix="coll", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    raw_cube, raw_header = load_fits(path, header=True)
    header = fix_header(raw_header)
    # mask saturated pixels
    satlevel = header["FULLWELL"] / header["GAIN"]
    cube = np.where(raw_cube >= satlevel, np.nan, raw_cube.astype("f4"))
    # do back subtraction
    if back_filename is not None:
        back_path = Path(back_filename)
        header["BACKFILE"] = back_path.name, "File used for background subtraction"
        with fits.open(back_path) as hdul:
            back = hdul[0].data
            back_err = hdul["ERR"].data
        cube -= back
    else:
        cube -= header["BIAS"]
        bkgnoise = np.sqrt(
            header["DC"] * header["EXPTIME"] * header["ENF"] ** 2 + header["RN"] ** 2
        )
        back_err = bkgnoise / header["EFFGAIN"]  # adu

    cube_err = np.sqrt(np.maximum(cube / header["EFFGAIN"], 0) * header["ENF"] ** 2 + back_err**2)
    # collapse cube (median by default)
    if cube.shape[0] > 1:
        master_flat, header = collapse_cube(cube, header=header, **kwargs)
        flat_var, _ = collapse_cube(cube_err**2, **kwargs)
        flat_err = np.sqrt(flat_var)
    else:
        master_flat = cube[0]
        flat_err = cube_err[0]

    normval = np.nanmedian(master_flat)
    master_flat /= normval
    flat_err /= normval
    header["FLATNORM"] = normval, "[adu] Flat field normalization factor"
    header["CALTYPE"] = "FLAT", "DPP calibration file type"
    header["BUNIT"] = "", "Unit of original values"
    bpmask = adaptive_sigma_clip_mask(master_flat)
    # get bad pixel mask using lacosmic
    # bpmask, _ = detect_cosmics(
    #     master_flat, invar=flat_err**2, inmask=np.isnan(master_flat), satlevel=satlevel,
    #     sigclip=10,
    #     gain=header["EFFGAIN"]
    # )
    # mask out bad pixels with nan- use np.isnan
    # to extract mask in future
    master_flat[bpmask] = np.nan
    flat_err[bpmask] = np.nan
    # save to disk
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(master_flat, header=header),
            fits.ImageHDU(flat_err, header=header, name="ERR"),
        ]
    )
    hdul.writeto(outpath, overwrite=True)
    return outpath


def match_calib_files(filenames, calib_files):
    cal_table = header_table(calib_files)
    rows = []
    for path in tqdm(map(Path, filenames), total=len(filenames), desc="Matching calibration files"):
        hdr = fix_header(load_fits_header(path))
        obstime = Time(hdr["MJD"], format="mjd", scale="utc")
        subset = cal_table.query(
            f"NAXIS1 == {hdr['NAXIS1']} and NAXIS2 == {hdr['NAXIS2']} and U_CAMERA == {hdr['U_CAMERA']}"
        )
        if len(subset) == 0:
            rows.append(dict(path=str(path.absolute()), backfile=None, flatfile=None))
            continue

        # background files
        back_mask = subset["CALTYPE"] == "BACKGROUND"
        if np.any(back_mask):
            if "U_EMGAIN" in cal_table.columns:
                mask = subset["U_EMGAIN"] == hdr["U_EMGAIN"]
            else:
                mask = subset["U_DETMOD"] == hdr["U_DETMOD"]
            if np.any(back_mask & mask):
                back_mask &= mask
                mask = np.abs(subset["EXPTIME"] - hdr["EXPTIME"]) < 0.1
                if np.any(back_mask & mask):
                    back_mask &= mask
            back_subset = subset.loc[back_mask]
            delta_time = Time(back_subset["MJD"], format="mjd", scale="utc") - obstime
            back_path = back_subset["path"].iloc[np.abs(delta_time.jd).argmin()]
        else:
            back_path = None

        # flat files
        flat_mask = subset["CALTYPE"] == "FLAT"
        if flat_mask.any():
            if "U_EMGAIN" in cal_table.columns:
                mask = subset["U_EMGAIN"] == hdr["U_EMGAIN"]
            else:
                mask = subset["U_DETMOD"] == hdr["U_DETMOD"]
            if np.any(flat_mask & mask):
                flat_mask &= mask
                mask = subset["FILTER01"] == hdr["FILTER01"]
                if np.any(flat_mask & mask):
                    flat_mask &= mask
                    mask = subset["FILTER02"] == hdr["FILTER02"]
                    if np.any(flat_mask & mask):
                        flat_mask &= mask
            flat_subset = subset.loc[flat_mask]
            delta_time = Time(flat_subset["MJD"], format="mjd", scale="utc") - obstime
            flat_path = flat_subset["path"].iloc[np.abs(delta_time.jd).argmin()]
        else:
            flat_path = None

        rows.append(dict(path=str(path.absolute()), backfile=back_path, flatfile=flat_path))
    return pd.DataFrame(rows)


def process_background_files(
    filenames: Iterable[str | Path],
    collapse: str = "median",
    output_directory: str | Path | None = None,
    force: bool = False,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd() / "background"

    # print(outdir)
    with mp.Pool(num_proc) as pool:
        jobs = []
        for path in map(Path, filenames):
            func = functools.partial(
                make_background_file, path, output_directory=outdir, method=collapse, force=force
            )
            jobs.append(pool.apply_async(func))
        job_iter = jobs if quiet else tqdm(jobs, desc="Collapsing background frames")
        frames = [job.get() for job in job_iter]

    return frames


def process_flat_files(
    filenames: Iterable[str | Path],
    collapse: str = "median",
    background_files: str | Path | None = None,
    output_directory: str | Path | None = None,
    force: bool = False,
    num_proc: int = DEFAULT_NPROC,
    quiet: bool = False,
) -> list[Path]:
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd() / "flat"

    if background_files is not None:
        calib_match_table = match_calib_files(filenames, background_files)
        mapping = zip(calib_match_table["path"], calib_match_table["backfile"], strict=True)
    else:
        mapping = zip(filenames, itertools.repeat(None), strict=False)
    with mp.Pool(num_proc) as pool:
        jobs = []
        for path, back_path in mapping:
            func = functools.partial(
                make_flat_file,
                path,
                back_filename=back_path,
                output_directory=outdir,
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(func))
        job_iter = jobs if quiet else tqdm(jobs, desc="Collapsing flat frames")
        frames = [job.get() for job in job_iter]

    return frames


def adaptive_sigma_clip_mask(data, sigma=10, boxsize=8):
    grid = np.arange(boxsize // 2, data.shape[0], step=boxsize)
    output_mask = np.zeros_like(data, dtype=bool)
    boxsize / 2
    for yi in grid:
        for xi in grid:
            inds = cutout_inds(data, center=(yi, xi), window=boxsize)
            cutout = data[inds]
            med = np.nanmedian(cutout, keepdims=True)
            std = np.nanstd(cutout, keepdims=True)
            output_mask[inds] = np.abs(cutout - med) > sigma * std

    return output_mask
