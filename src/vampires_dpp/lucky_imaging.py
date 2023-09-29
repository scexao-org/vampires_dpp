from pathlib import Path
from typing import Final, Literal, Optional

import numpy as np
import scipy.stats as st
from astropy.io import fits
from astropy.nddata import Cutout2D
from numpy.typing import ArrayLike

from .image_processing import collapse_cube, pad_cube, shift_cube, shift_frame
from .image_registration import register_frame
from .indexing import cutout_inds, frame_center
from .util import get_center

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
    # get the quantile along the time dimension
    cutoff = np.nanquantile(values, quantile, axis=-1, keepdims=True)
    # get mask by tossing if _any_ field or psf is below spec.
    # this way mask can be applied to time dimension
    if input_key == "fwhm":
        return np.all(values < cutoff, axis=(0, 1))
    else:
        return np.all(values > cutoff, axis=(0, 1))


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
    if cx.ndim == 3:
        cx = np.mean(cx, axis=-2)
    if cy.ndim == 3:
        cy = np.mean(cy, axis=-2)

    # stack so size is (Nfields, Nframes, x/y)
    centroids = np.stack((cy, cx), axis=2)
    return centroids


def recenter_frame(frame, method, offsets, window=30, **kwargs):
    frame_ctr = frame_center(frame)
    ctr = 0
    for off in offsets - offsets.mean(0):
        inds = cutout_inds(frame, window=window, center=frame_ctr + off)
        ctr += register_frame(frame, inds, method=method, **kwargs)
    ctr = ctr / offsets.shape[0]

    return shift_frame(frame, frame_ctr - ctr)


def lucky_image_file(
    hdu,
    outpath,
    centroids,
    method: str = "median",
    register: Optional[Literal["com", "peak", "model"]] = "com",
    frame_select: Optional[Literal["peak", "normvar", "var", "strehl", "model", "fwhm"]] = None,
    metric_file: Optional[Path] = None,
    force: bool = False,
    recenter: bool = True,
    select_cutoff: float = 0,
    crop_width=None,
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

    if crop_width is None:
        if "MBI" in header.get("OBS-MOD", ""):
            crop_width = 536
        else:
            crop_width = np.min(cube.shape[-2:])

    cam_num = header["U_CAMERA"]
    cam_key = f"cam{cam_num:.0f}"
    fields = centroids[cam_key]

    # if no metric file assume straight collapsing
    if not metric_file:
        frame, header = collapse_cube(cube, method=method, header=header, **kwargs)
        frames = []
        headers = []
        for field, ctr in fields.items():
            field_ctr = get_center(frame, ctr, cam_num)
            if field_ctr.ndim > 1:
                field_ctr = field_ctr.mean(axis=0)
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            frames.append(cutout.data)
            hdr = header.copy()
            hdr["FIELD"] = field
            headers.append(hdr)
        prim_hdu = fits.PrimaryHDU(np.array(frames), header=header)
        hdus = [fits.ImageHDU(frame, header=hdr) for frame, hdr in zip(frames, headers)]
        hdu = fits.HDUList([prim_hdu, *hdus])
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
        psf_centroids = get_centroids_from(masked_metrics, input_key=register)

    frames = []
    headers = []
    for i, (field, psf_ctr) in enumerate(fields.items()):
        ctr = psf_ctr.mean(axis=0)
        offs = psf_ctr - ctr
        field_ctr = get_center(cube, ctr, cam_num)
        offsets = psf_centroids[i] - field_ctr
        aligned_frames = []
        for frame, offset in zip(cube, offsets):
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            if register:
                # determine maximum padding, with sqrt(2)
                # for radial coverage
                rad_factor = (np.sqrt(2) - 1) * np.max(cutout.shape[-2:]) / 2
                npad = np.ceil(rad_factor + 1).astype(int)
                frame_padded = np.pad(cutout.data, npad, constant_values=np.nan)
                shifted = shift_frame(frame_padded, -offset, **kwargs)
                aligned_frames.append(shifted)
            else:
                aligned_frames.append(cutout.data)
        # ## Step 3: Collapse
        coll_frame, header = collapse_cube(np.array(aligned_frames), header=header)
        ## Step 4: Recenter
        if recenter:
            coll_frame = recenter_frame(coll_frame, method=register, offsets=offs)
        frames.append(coll_frame)
        hdr = header.copy()
        hdr["FIELD"] = field
        ## handle header metadata
        add_metrics_to_header(hdr, masked_metrics, index=i)
        headers.append(hdr)
    prim_hdu = fits.PrimaryHDU(np.array(frames), header=header)
    hdus = [fits.ImageHDU(frame, header=hdr) for frame, hdr in zip(frames, headers)]
    hdu = fits.HDUList([prim_hdu, *hdus])
    # write to disk
    hdu.writeto(outpath, overwrite=True)
    return hdu


COMMENT_FSTRS: Final[dict[str, str]] = {
    "max": "[adu] Peak signal{}measured in window {}",
    "sum": "[adu] Total signal{}measured in window {}",
    "mean": "[adu] Mean signal{}measured in window {}",
    "med": "[adu] Median signal{}measured in window {}",
    "var": "[adu^2] Signal variance{}measured in window {}",
    "nvar": "[adu] Normalized variance{}measured in window {}",
    "photr": "[px] Radius of photometric aperture",
    "photf": "[adu] Photometric flux{}in window {}",
    # "phote": "[adu] Photometric flux error in window {}",
    "com_x": "[px] Center-of-mass{}along x axis in window {}",
    "com_y": "[px] Center-of-mass{}along y axis in window {}",
    "pk_x": "[px] Peak signal index{}along x axis in window {}",
    "pk_y": "[px] Peak signal index{}along y axis in window {}",
    "psfmod": "Model used for PSF fitting",
    "mod_x": "[px] Model center fit{}along x axis in window {}",
    "mod_y": "[px] Model center fit{}along y axis in window {}",
    "mod_w": "[px] Model FWHM fit{}in window {}",
    "mod_a": "[adu] Model amplitude fit{}in window {}",
}


def add_metrics_to_header(hdr: fits.Header, metrics: dict, index=0) -> fits.Header:
    for key, field_arrs in metrics.items():
        arr = field_arrs[index]
        if key not in COMMENT_FSTRS:
            continue
        key_up = key.upper()
        if key_up in ("PSFMOD", "PHOTR"):
            hdr[key_up] = arr[0][0], COMMENT_FSTRS[key]
            continue
        for i, psf in enumerate(arr):
            comment = COMMENT_FSTRS[key].format(" ", i)
            hdr[f"{key_up}{i}"] = np.mean(psf), comment
            err_comment = COMMENT_FSTRS[key].format(" err ", i)
            if len(psf) == 1:
                sem = 0
            else:
                sem = st.sem(psf, nan_policy="omit")
            hdr[f"{key_up[:5]}ER{i}"] = sem, err_comment
    return hdr