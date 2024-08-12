from pathlib import Path
from typing import Final, Literal

import numpy as np
import scipy.stats as st
from astropy.io import fits
from astropy.nddata import Cutout2D
from loguru import logger

from .analysis import add_frame_statistics
from .constants import SATSPOT_REF_ANGLE, SATSPOT_REF_NACT
from .headers import sort_header
from .image_processing import collapse_cube, combine_frames_headers, shift_frame
from .image_registration import offset_centroids
from .indexing import cutout_inds, frame_center, get_mbi_centers
from .specphot.filters import update_header_with_filt_info
from .specphot.specphot import convert_to_surface_brightness, specphot_calibration
from .util import get_center, wrap_angle
from .wcs import apply_wcs

FRAME_SELECT_MAP: Final[dict[str, str]] = {
    "peak": "max",
    # "strehl": "strel",
    "normvar": "nvar",
    "var": "var",
    "fwhm": "fwhm",
}
logger.disable(__name__)


def get_frame_select_mask(metrics, input_key, quantile=0):
    frame_select_key = FRAME_SELECT_MAP[input_key]
    # create masked metrics
    values = metrics[frame_select_key]
    # get the quantile along the time dimension
    cutoff = np.nanquantile(values, quantile, axis=-1, keepdims=True)
    # get mask by tossing if _all_ field or psf is below spec.
    # this way mask can be applied to time dimension
    if input_key == "fwhm":
        return np.any(values <= cutoff, axis=(0, 1))
    else:
        return np.any(values >= cutoff, axis=(0, 1))


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


def get_recenter_offset(frame, method, offsets, window=30, psf=None, **kwargs):
    frame_ctr = frame_center(frame)
    ctr = 0
    n = 1
    for off in offsets - offsets.mean(0):
        inds = cutout_inds(frame, window=window, center=frame_ctr + off)
        offsets = offset_centroids(frame, None, inds, psf=psf)
        ctr += (offsets[method] - ctr) / n
        n += 1
    return frame_ctr - ctr


def reproject_header(
    header: fits.Header,
    astrometry: dict,
    field: str,
    refsep=SATSPOT_REF_NACT,
    refang=SATSPOT_REF_ANGLE,
):
    # if "U_MBI" in header:
    # refang += 180
    match header["U_CAMERA"]:
        case 1:
            astrom_info = astrometry["cam1"]
            instang = refang + astrom_info["angle"]
        case 2:
            astrom_info = astrometry["cam2"]
            instang = refang + (90 - astrom_info["angle"])

    # the reference angle comes from CMOS detectors, need to subtract 180 for EMCCD
    # due to parity flip
    if "U_MBI" not in header:
        instang += 180
    field_idx = astrom_info["fields"].index(field)
    # Get expected separation
    waffle_sep = refsep * header["RESELEM"]  # mas
    platescale = waffle_sep / astrom_info["separation"][field_idx]
    header["PXSCALE"] = platescale, header.comments["PXSCALE"]
    header["INST-PA"] = wrap_angle(instang), header.comments["INST-PA"]
    header["PAOFFSET"] = (
        wrap_angle(header["INST-PA"] - 180 + header["D_IMRPAP"]),
        header.comments["PAOFFSET"],
    )
    header["DEROTANG"] = wrap_angle(header["PA"] + header["PAOFFSET"]), header.comments["DEROTANG"]
    return header


def lucky_image_file(
    hdul,
    outpath,
    centroids,
    metric_file: Path,
    method: str = "median",
    register: Literal["com", "peak", "gauss", "dft"] | None = None,
    frame_select: Literal["peak", "normvar", "var", "fwhm"] | None = None,
    force: bool = False,
    recenter: Literal["com", "peak", "gauss", "dft"] | None = None,
    reproject: bool = True,
    astrometry=None,
    refsep=45,
    refang=90,
    select_cutoff: float = 0,
    crop_width=None,
    aux_dir=None,
    specphot=None,
    window: int = 30,
    psfs=None,
    **kwargs,
) -> Path:
    if (
        not force
        and outpath.is_file()
        and metric_file
        and Path(metric_file).stat().st_mtime < outpath.stat().st_mtime
    ):
        logger.debug(f"Skipped lucky imaging loading {outpath} from disk.")
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

    logger.debug(f"Loading metric file: {metric_file}")
    metrics = np.load(metric_file)

    # mask metrics
    masked_metrics = metrics
    if frame_select and select_cutoff > 0:
        header["FS_METH"] = frame_select, "DPP frame selection method"
        header["FS_cut"] = select_cutoff, "DPP frame selection cutoff quantile"
        mask = get_frame_select_mask(metrics, input_key=frame_select, quantile=select_cutoff)
        masked_metrics = {key: metric[..., mask] for key, metric in metrics.items()}
        # frame select this cube
        cube = cube[mask]
        cube_err = cube_err[mask]

    N = len(cube)
    if register is not None:
        psf_centroids = get_centroids_from(masked_metrics, input_key=register)
        header["REG_METH"] = register, "DPP registration method"
    elif "MBIR" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(cube, reduced=True)
        psf_centroids = np.zeros((3, N, 2))
        for idx, key in enumerate(fields):
            psf_centroids[idx] = ctr_dict[key]
    elif "MBI" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(cube)
        psf_centroids = np.zeros((4, N, 2))
        for idx, key in enumerate(fields):
            psf_centroids[idx] = ctr_dict[key]
    else:
        psf_centroids = np.zeros((1, N, 2))
        psf_centroids[:] = frame_center(cube)
    frames = []
    frame_errs = []
    headers = []
    if psfs is None:
        psfs = [None for _ in range(len(fields))]
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
                aligned_err_frames.append(cutout_err.data)
        aligned_cube = np.array(aligned_frames)
        aligned_err_cube = np.array(aligned_err_frames)
        ## Step 3: Collapseo
        if method.lower() != "none":
            coll_frame, header = collapse_cube(aligned_cube, header=header, method=method)
            # collapse error in quadrature
            coll_err_frame = np.sqrt(np.nansum(aligned_err_cube**2, axis=0)) / N
            ## Step 4: Recenter
            if recenter is not None:
                recenter_offset = get_recenter_offset(
                    coll_frame, method=recenter, offsets=offs, window=window, psf=psfs[i]
                )
                header["REC_METH"] = register, "DPP recenter method"
                coll_frame = shift_frame(coll_frame, recenter_offset)
                coll_err_frame = shift_frame(coll_err_frame, recenter_offset)
        hdr = header.copy()
        hdr["FIELD"] = field
        ## handle header metadata
        add_metrics_to_header(hdr, masked_metrics, index=i)
        hdr, _ = update_header_with_filt_info(hdr)

        ## Step 5. Reprojection
        if reproject:
            hdr = reproject_header(hdr, astrometry, field=field, refsep=refsep, refang=refang)
        if method.lower() != "none":
            hdr = apply_wcs(coll_frame, hdr, angle=hdr["DEROTANG"])
        else:
            hdr = apply_wcs(aligned_cube, hdr, angle=hdr["DEROTANG"])

        ## Step 6. Specphot cal
        # if desired, use synthetic photometry
        if specphot is not None:
            # TODO update for frame-by-frame flux
            hdr = specphot_calibration(hdr, outdir=aux_dir, config=specphot)
            # TODO add optional units
            hdr["BUNIT"] = "Jy/arcsec^2"

            if method.lower() != "none":
                coll_frame = convert_to_surface_brightness(coll_frame, hdr)
                coll_err_frame = convert_to_surface_brightness(coll_err_frame, hdr)
                hdr = add_frame_statistics(coll_frame, coll_err_frame, hdr)
            else:
                aligned_cube = convert_to_surface_brightness(aligned_cube, hdr)
                aligned_err_cube = convert_to_surface_brightness(aligned_err_cube, hdr)
                hdrs = [
                    add_frame_statistics(frame, frame_err, hdr.copy())
                    for frame, frame_err in zip(aligned_cube, aligned_err_cube, strict=True)
                ]
                hdr = combine_frames_headers(hdrs, wcs=True)

        if method.lower() != "none":
            frames.append(coll_frame)
            frame_errs.append(coll_err_frame)
        else:
            frames.append(aligned_cube)
            frame_errs.append(aligned_err_cube)
        headers.append(hdr)

    comb_header = combine_frames_headers(headers, wcs=True)
    comb_header["NCOADD"] = N, "Number of frames combined in collapsed file"
    comb_header["TINT"] = comb_header["EXPTIME"] * N
    comb_header = sort_header(comb_header)
    prim_hdu = fits.PrimaryHDU(np.array(frames), header=comb_header)
    err_hdu = fits.ImageHDU(np.array(frame_errs), header=comb_header, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    # add headers from each field
    hdul.extend([fits.ImageHDU(header=sort_header(hdr), name=hdr["FIELD"]) for hdr in headers])
    # write to disk
    logger.debug(f"Saving collapsed output to {outpath}")
    hdul.writeto(outpath, overwrite=True)
    return hdul


COMMENT_FSTRS: Final[dict[str, str]] = {
    "max": "[{}] Peak signal{}in window {}",
    "sum": "[{}] Total signal{}in window {}",
    "mean": "[{}] Mean signal{}in window {}",
    "med": "[{}] Median signal{}in window {}",
    "var": "[({})^2] Signal variance{}in window {}",
    "nvar": "[{}] Normed variance{}in window {}",
    "photr": "[pix] Photometric aperture radius",
    "photf": "[{}] Photometric flux{}in window {}",
    "phote": "[{}] Photometric fluxerr{}in window {}",
    "psff": "[{}] PSF flux{}in window {}",
}
CENTROID_COMM_FSTRS: Final[dict[str, str]] = {
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
        unit = hdr["BUNIT"]
        N = len(arr)
        for i, psf in enumerate(arr):
            # mean val
            if key in COMMENT_FSTRS:
                comment = COMMENT_FSTRS[key].format(unit, " ", i)
                err_comment = COMMENT_FSTRS[key].format(unit, " err ", i)
            elif key in CENTROID_COMM_FSTRS:
                comment = CENTROID_COMM_FSTRS[key].format(" ", i)
                err_comment = CENTROID_COMM_FSTRS[key].format(" err ", i)

            psf_val = np.mean(psf)
            mean_val += (psf_val - mean_val) / (i + 1)
            hdr[f"{key_up}{i}"] = np.nan_to_num(psf_val), comment
            # sem
            if len(psf) == 1:
                sem = 0
            elif "PHOTE" in key_up:
                sem = np.sqrt(np.mean(psf**2) / N)
            else:
                sem = st.sem(psf, nan_policy="omit")
            hdr[f"{key_up[:5]}ER{i}"] = np.nan_to_num(sem), err_comment
        hdr[f"{key_up[:5]}"] = np.nan_to_num(mean_val), comment.split(" in window")[0]
    return hdr
