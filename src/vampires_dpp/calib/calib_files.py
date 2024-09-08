# library functions for common calibration tasks like
# background subtraction, collapsing cubes
import functools
import multiprocessing as mp
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from scipy import optimize
from skimage import filters, morphology
from tqdm.auto import tqdm

from vampires_dpp.coadd import collapse_cube
from vampires_dpp.headers import fix_header, sort_header
from vampires_dpp.image_processing import adaptive_sigma_clip_mask
from vampires_dpp.organization import header_table
from vampires_dpp.paths import get_paths
from vampires_dpp.util import load_fits, load_fits_header

__all__ = (
    "make_background_file",
    "make_flat_file",
    "match_calib_file",
    "process_background_files",
    "process_flat_files",
)


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

    noise = np.sqrt(np.nanmean(back_std**2))
    header["NOISEADU"] = noise, "[adu] RMS noise in background file"
    header["NOISE"] = noise * header["EFFGAIN"], "[e-] RMS noise in background file"
    header["CALTYPE"] = "BACKGROUND", "DPP calibration file type"
    header = sort_header(header)
    # get bad pixel mask from adaptive sigma clip
    bpmask = adaptive_sigma_clip_mask(master_background)
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


def make_flat_file(
    filename: str, normalize: bool = True, force: bool = False, back_filename=None, **kwargs
):
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
        header["BACKFILE"] = back_path.name
        with fits.open(back_path) as hdul:
            back = hdul[0].data
            back_err = hdul["ERR"].data
        cube -= back
    else:
        cube -= header["BIAS"]
        rn = header["RN"] / header["DETGAIN"]
        bkgnoise = np.sqrt(header["DC"] * header["EXPTIME"] * header["ENF"] ** 2 + rn**2)
        back_err = bkgnoise / header["EFFGAIN"]  # adu

    cube_err = np.sqrt(np.maximum(cube / header["EFFGAIN"], 0) * header["ENF"] ** 2 + back_err**2)
    # collapse cube (median by default)
    if cube.shape[0] > 1:
        master_flat, header = collapse_cube(cube, header=header, **kwargs)
        flat_err = np.sqrt(np.sum(np.power(cube_err / cube_err.shape[0], 2), axis=0))
    else:
        master_flat = cube[0]
        flat_err = cube_err[0]

    # for MBI data need to normalize each field individually
    # otherwise use frame median
    if normalize:
        if "MBI" in header["OBS-MOD"]:
            master_flat, flat_err, header = normalize_multiband_flats(
                master_flat, flat_err, header=header
            )
        else:
            normval = np.nanmedian(master_flat)
            master_flat /= normval
            flat_err /= normval
            header["NORMVAL"] = (
                normval,
                f"[{header['BUNIT'].lower()}] Flat field normalization factor",
            )
        master_flat[master_flat < 0.2] = np.nan
        flat_err[master_flat < 0.2] = np.nan
    header["CALTYPE"] = "FLAT", "DPP calibration file type"
    header["BUNIT"] = "", "Unit of original values"
    # bpmask = adaptive_sigma_clip_mask(master_flat)
    # mask out bad pixels with nan- use np.isnan
    # to extract mask in future
    # master_flat[bpmask] = np.nan
    # flat_err[bpmask] = np.nan
    # save to disk
    header = sort_header(header)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(master_flat, header=header),
            fits.ImageHDU(flat_err, header=header, name="ERR"),
        ]
    )
    hdul.writeto(outpath, overwrite=True)
    return outpath


def match_calib_file(filename, calib_table):
    path = Path(filename)
    hdr = fix_header(load_fits_header(path))
    obstime = Time(hdr["MJD"], format="mjd", scale="utc")
    keys_to_match = ("PRD-MIN1", "PRD-MIN2", "PRD-RNG1", "PRD-RNG2", "U_CAMERA")
    subset = calib_table.query(" and ".join(f"`{k}` == {hdr[k]}" for k in keys_to_match))
    if len(subset) == 0:
        return dict(backfile=None, flatfile=None)

    # background files
    back_mask = subset["CALTYPE"] == "BACKGROUND"
    if np.any(back_mask):
        if "U_EMGAIN" in calib_table.columns:
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
        if "U_EMGAIN" in calib_table.columns:
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

    return dict(backfile=back_path, flatfile=flat_path)


def process_background_files(
    filenames: Iterable[str | Path],
    collapse: str = "median",
    output_directory: str | Path | None = None,
    force: bool = False,
    num_proc: int = 1,
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
    normalize: bool = True,
    background_files: str | Path | None = None,
    output_directory: str | Path | None = None,
    force: bool = False,
    num_proc: int = 1,
    quiet: bool = False,
) -> list[Path]:
    if output_directory is not None:
        outdir = Path(output_directory)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd() / "flat"

    if background_files is not None:
        calib_table = header_table(background_files, quiet=True)
    with mp.Pool(num_proc) as pool:
        jobs = []
        for path in filenames:
            calib_match = match_calib_file(path, calib_table)
            func = functools.partial(
                make_flat_file,
                path,
                normalize=normalize,
                back_filename=calib_match["backfile"],
                output_directory=outdir,
                method=collapse,
                force=force,
            )
            jobs.append(pool.apply_async(func))
        job_iter = jobs if quiet else tqdm(jobs, desc="Collapsing flat frames")
        frames = [job.get() for job in job_iter]

    return frames


## MBI flat-finding


@dataclass
class MBIField:
    x: float
    y: float
    width: float
    height: float
    theta: float


def normalize_multiband_flats(flat, flat_err, header: fits.Header, **kwargs):
    # expect 3 fields for MBI reduced, 4 for full frame
    nfields = 3 if "MBIR" in header["OBS-MOD"] else 4
    fields = find_multiband_flat_fields(flat, nfields=nfields, **kwargs)
    # for each field, get a crop and determine normalized value
    norm_flat = np.zeros_like(flat)
    norm_flat_err = np.zeros_like(flat_err)
    for i, field in enumerate(fields):
        # get undersized cutout
        cutout = Cutout2D(flat, (field.x, field.y), (field.height - 50, field.width - 50))
        normval = np.median(cutout.data)
        # now get oversized cutout for correction and store in output array
        cutout = Cutout2D(flat, (field.x, field.y), (field.height + 50, field.width + 50))
        norm_flat[cutout.slices_original] = cutout.data[cutout.slices_cutout] / normval
        norm_flat_err[cutout.slices_original] = flat_err[cutout.slices_original] / normval
        # update header
        header[f"FIELDX{i:1d}"] = field.x, f"[pix] x centroid estimate of field {i:1d}"
        header[f"FIELDY{i:1d}"] = field.y, f"[pix] y centroid estimate of field {i:1d}"
        header[f"FIELDW{i:1d}"] = field.width, f"[pix] width estimate of field {i:1d}"
        header[f"FIELDH{i:1d}"] = field.height, f"[pix] height estimate of field {i:1d}"
        header[f"FIELDTH{i:1d}"] = field.theta, f"[deg] rotation estimate of field {i:1d}"
        header[f"NORMVAL{i:1d}"] = (
            normval,
            f"[{header['BUNIT'].lower()}] flat-field norm value for field {i:1d}",
        )

    return norm_flat, norm_flat_err, header


def find_multiband_flat_fields(
    flat, *, nfields: Literal[3, 4], threshold: float = 1e3, closing_size: int = 5, pad: int = 10
) -> list[MBIField]:
    # find pixel values above background signal
    where_light = flat > threshold
    # perform binary closing using a morphological element
    # of custom size with corners removed
    element = np.ones((closing_size, closing_size))
    element[0, 0] = element[0, -1] = element[-1, 0] = element[-1, -1] = 0

    # Do a little loop auto-adjusting threshold until we find expected number of fields
    for _ in range(10):
        closed = morphology.binary_closing(where_light, element)
        connected_fields = morphology.label(closed)
        num_connected = np.max(connected_fields)

        # if too few, need lower threshold
        if num_connected < nfields:
            threshold *= 0.6
            where_light = flat > threshold
        # if too high, need higher threshold
        elif num_connected > nfields:
            threshold *= 2
            where_light = flat > threshold
        else:
            break
    else:
        msg = f"Could not find {nfields} fields in flat frame after 10 retries"
        raise RuntimeError(msg)

    # TODO no clue what this is doing
    label_surface = (np.sum(connected_fields == k) for k in range(1, nfields + 1))
    assert all(s > 220_000 and s < 300_000 for s in label_surface)

    # Get the crops
    crop_params = []
    crops = []
    for k in range(1, nfields + 1):
        fields = connected_fields == k
        val_cols = np.where(np.max(fields, axis=0))[0]
        val_rows = np.where(np.max(fields, axis=1))[0]

        crop_params += [[val_rows[0], val_rows[-1], val_cols[0], val_cols[-1]]]
        crops += [fields[val_rows[0] : val_rows[-1] + 1, val_cols[0] : val_cols[-1] + 1]]

    fields = []
    for crop, crop_param in zip(crops, crop_params, strict=True):
        padded = np.pad(crop, pad)
        edge_map = filters.sobel(padded) > 0.1
        y, x, lr, lc, th = _fit_edgemap_rectangle(edge_map, pad=pad)

        corner_r = y - pad + crop_param[0]
        corner_c = x - pad + crop_param[2]
        center_r = corner_r + lc * np.sin(th) / 2 + lr * np.cos(th) / 2
        center_c = corner_c + lc * np.cos(th) / 2 - lr * np.sin(th) / 2

        field = MBIField(x=center_c, y=center_r, width=lc, height=lr, theta=np.rad2deg(th))
        fields.append(field)

    return fields


def _rectangle_scoring_func(param_array, r_val, c_val) -> float:
    r_A = param_array[0]
    c_A = param_array[1]
    length_rowwise = param_array[2]
    length_colwise = param_array[3]
    angle = param_array[4]

    r_C = r_A + np.sin(angle) * length_colwise + np.cos(angle) * length_rowwise
    c_C = c_A + np.cos(angle) * length_colwise - np.sin(angle) * length_rowwise

    # Compute distance of all points to the 4 sides...
    # TODO rewrite this algorithm
    n_pts = len(r_val)
    dists = np.zeros((4, n_pts))

    dists[0] = -(r_val - r_A) * np.sin(angle) - (c_val - c_A) * np.cos(angle)
    dists[1] = (r_val - r_C) * np.cos(angle) - (c_val - c_C) * np.sin(angle)
    dists[2] = -(r_val - r_C) * np.sin(angle) - (c_val - c_C) * np.cos(angle)
    dists[3] = (r_val - r_A) * np.cos(angle) - (c_val - c_A) * np.sin(angle)

    return np.sum(np.min(dists**2, axis=0))


def _fit_edgemap_rectangle(edge_map, pad: int = 10):
    r_val, c_val = np.where(edge_map)
    init_params = (pad, pad, 536 + 2 * pad, 536 + 2 * pad, 0)
    bounds = ((0, 100), (0, 100), (400, 700), (400, 700), (np.deg2rad(-5), np.deg2rad(10)))
    result = optimize.minimize(
        _rectangle_scoring_func,
        init_params,
        args=(r_val, c_val),
        bounds=bounds,
        method="L-BFGS-B",
        jac="3-point",
    )
    assert result.success
    yA, xA, lr, lc, th = result.x
    return yA, xA, lr, lc, th
