from pathlib import Path
from typing import Final, Literal

import numpy as np
import scipy.stats as st
from astropy.io import fits
from astropy.nddata import Cutout2D

from .analysis import add_frame_statistics
from .image_processing import collapse_cube, combine_frames_headers, shift_frame
from .image_registration import offset_centroids
from .indexing import cutout_inds, frame_center
from .specphot.specphot import convert_to_surface_brightness, specphot_calibration
from .util import get_center

FRAME_SELECT_MAP: Final[dict[str, str]] = {
    "peak": "max",
    # "strehl": "strel",
    "normvar": "nvar",
    "var": "var",
    "fwhm": "fwhm",
}


def get_frame_select_mask(metrics, input_key, quantile=0):
    frame_select_key = FRAME_SELECT_MAP[input_key]
    # create masked metrics
    values = metrics[frame_select_key]
    # get the quantile along the time dimension
    cutoff = np.nanquantile(values, quantile, axis=-1, keepdims=True)
    # get mask by tossing if _any_ field or psf is below spec.
    # this way mask can be applied to time dimension
    if input_key == "fwhm":
        return np.all(values < cutoff, axis=(0, 1))
    else:
        return np.all(values > cutoff, axis=(0, 1))


def get_centroids_from(metrics, input_key):
    cx = metrics[f"{input_key[:4]}x"]
    cy = metrics[f"{input_key[:4]}y"]
    # if there are values from multiple PSFs (e.g. satspots)
    # take the mean centroid of each PSF
    if cx.ndim == 3:
        cx = np.mean(cx, axis=-2)
    if cy.ndim == 3:
        cy = np.mean(cy, axis=-2)

    # stack so size is (Nfields, Nframes, x/y)
    centroids = np.stack((cy, cx), axis=2)
    return centroids


def get_recenter_offset(frame, method, offsets, window=30, **kwargs):
    frame_ctr = frame_center(frame)
    ctr = 0
    n = 1
    for off in offsets - offsets.mean(0):
        inds = cutout_inds(frame, window=window, center=frame_ctr + off)
        offsets = offset_centroids(frame, None, inds)
        ctr += (offsets[method] - ctr) / n
        n += 1
    return frame_ctr - ctr


def lucky_image_file(
    hdul,
    outpath,
    centroids,
    metric_file: Path,
    method: str = "median",
    register: Literal["com", "peak", "gauss", "dft"] | None = "com",
    frame_select: Literal["peak", "normvar", "var", "strehl", "fwhm"] | None = None,
    force: bool = False,
    recenter: Literal["com", "peak", "gauss", "dft"] | None = None,
    select_cutoff: float = 0,
    crop_width=None,
    preproc_dir=None,
    specphot=None,
    **kwargs,
) -> Path:
    if (
        not force
        and outpath.is_file()
        and metric_file
        and Path(metric_file).stat().st_mtime < outpath.stat().st_mtime
    ):
        with fits.open(outpath) as hdul:
            return hdul

    cube = hdul[0].data
    cube_err = hdul["ERR"].data
    header = hdul[0].header

    if crop_width is None:
        crop_width = 536 if "MBI" in header.get("OBS-MOD", "") else np.min(cube.shape[-2:])

    cam_num = header["U_CAMERA"]
    cam_key = f"cam{cam_num:.0f}"
    fields = centroids[cam_key] if cam_key in centroids else {"": [frame_center(cube)]}

    metrics = np.load(metric_file)

    # mask metrics
    masked_metrics = metrics
    if frame_select and select_cutoff > 0:
        mask = get_frame_select_mask(metrics, input_key=frame_select, quantile=select_cutoff)
        masked_metrics = {key: metric[..., mask] for key, metric in metrics.items()}
        # frame select this cube
        cube = cube[mask]
        cube_err = cube_err[mask]

    if register:
        psf_centroids = get_centroids_from(masked_metrics, input_key=register)
    frames = []
    frame_errs = []
    headers = []
    for i, (field, psf_ctr) in enumerate(fields.items()):
        ctr = np.mean(psf_ctr, axis=0)
        offs = psf_ctr - ctr
        field_ctr = get_center(cube, ctr, cam_num)
        offsets = psf_centroids[i] - field_ctr
        aligned_frames = []
        aligned_err_frames = []
        for frame, frame_err, offset in zip(cube, cube_err, offsets, strict=True):
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            cutout_err = Cutout2D(frame_err, field_ctr[::-1], size=crop_width, mode="partial")
            if register:
                # determine maximum padding, with sqrt(2)
                # for radial coverage
                rad_factor = (np.sqrt(2) - 1) * np.max(cutout.shape[-2:]) / 2
                npad = np.ceil(rad_factor + 1).astype(int)
                frame_padded = np.pad(cutout.data, npad, constant_values=np.nan)
                shifted = shift_frame(frame_padded, -offset, **kwargs)
                aligned_frames.append(shifted)
                frame_err_padded = np.pad(cutout_err.data, npad, constant_values=np.nan)
                shifted_err = shift_frame(frame_err_padded, -offset, **kwargs)
                aligned_err_frames.append(shifted_err)
            else:
                aligned_frames.append(cutout.data)
                aligned_err_frames.append(shifted_err)
        ## Step 3: Collapse
        coll_frame, header = collapse_cube(np.array(aligned_frames), header=header, method=method)
        # collapse error in quadrature
        N = len(aligned_frames)
        header["TINT"] = header["EXPTIME"] * N
        coll_var_frame, _ = collapse_cube(np.power(aligned_err_frames, 2), method=method)
        coll_err_frame = np.sqrt(coll_var_frame / N)
        ## Step 4: Recenter
        if recenter is not None:
            recenter_offset = get_recenter_offset(coll_frame, method=recenter, offsets=offs)
            coll_frame = shift_frame(coll_frame, recenter_offset)
            coll_err_frame = shift_frame(coll_err_frame, recenter_offset)
        hdr = header.copy()
        hdr["FIELD"] = field
        ## handle header metadata
        add_metrics_to_header(hdr, masked_metrics, index=i)
        if specphot is not None:
            hdr = specphot_calibration(hdr, outdir=preproc_dir, config=specphot)
            coll_frame = convert_to_surface_brightness(coll_frame, hdr)
            coll_err_frame = convert_to_surface_brightness(coll_err_frame, hdr)
            hdr["BUNIT"] = "Jy/arcsec^2"

        hdr = add_frame_statistics(coll_frame, coll_err_frame, hdr)

        frames.append(coll_frame)
        frame_errs.append(coll_err_frame)
        headers.append(hdr)
    comb_header = combine_frames_headers(headers, wcs=True)
    prim_hdu = fits.PrimaryHDU(np.array(frames), header=comb_header)
    hdul = fits.HDUList(prim_hdu)
    for field, frame, hdr in zip(fields.keys(), frames, headers, strict=True):
        hdu = fits.ImageHDU(frame, header=hdr, name=field)
        hdul.append(hdu)
    for field, frame_err, hdr in zip(fields.keys(), frame_errs, headers, strict=True):
        hdu = fits.ImageHDU(frame_err, header=hdr, name=f"{field}ERR")
        hdul.append(hdu)
    # write to disk
    hdul.writeto(outpath, overwrite=True)
    return hdul


COMMENT_FSTRS: Final[dict[str, str]] = {
    "max": "[adu] Peak signal{}in window {}",
    "sum": "[adu] Total signal{}in window {}",
    "mean": "[adu] Mean signal{}in window {}",
    "med": "[adu] Median signal{}in window {}",
    "var": "[adu^2] Signal variance{}in window {}",
    "nvar": "[adu] Normed variance{}in window {}",
    "photr": "[pix] Photometric aperture radius",
    "photf": "[adu] Photometric flux{}in window {}",
    "phote": "[adu] Photometric fluxerr{}in window {}",
    "psff": "[adu] PSF flux{}in window {}",
    "comx": "[pix] COM x{}in window {}",
    "comy": "[pix] COM y{}in window {}",
    "peakx": "[pix] Peak index x{}in window {}",
    "peaky": "[pix] Peak index y{}in window {}",
    "gausx": "[pix] Gauss. fit x{}in window {}",
    "gausy": "[pix] Gauss. fit y{}in window {}",
    "dftx": "[pix] Cross-corr. x{}in window {}",
    "dfty": "[pix] Cross-corr. y{}in window {}",
    "fwhm": "[pix] Gauss. fit fwhm{}in window {}",
}


def add_metrics_to_header(hdr: fits.Header, metrics: dict, index=0) -> fits.Header:
    for key, field_arrs in metrics.items():
        arr = field_arrs[index]
        if key not in COMMENT_FSTRS:
            continue
        key_up = key.upper()
        if key_up == "PHOTR":
            hdr[key_up] = arr[0][0], COMMENT_FSTRS[key]
            continue
        mean_val = 0
        N = len(arr)
        for i, psf in enumerate(arr):
            # mean val
            comment = COMMENT_FSTRS[key].format(" ", i)
            psf_val = np.mean(psf)
            mean_val += (psf_val - mean_val) / (i + 1)
            hdr[f"{key_up}{i}"] = np.nan_to_num(psf_val), comment
            # sem
            err_comment = COMMENT_FSTRS[key].format(" err ", i)
            if len(psf) == 1:
                sem = 0
            elif "PHOTE" in key_up:
                sem = np.sqrt(np.mean(psf**2) / N)
            else:
                sem = st.sem(psf, nan_policy="omit")
            hdr[f"{key_up[:5]}ER{i}"] = np.nan_to_num(sem), err_comment
        hdr[f"{key_up[:5]}"] = np.nan_to_num(mean_val), comment.split(" in window")[0]
    return hdr
