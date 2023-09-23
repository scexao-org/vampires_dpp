from typing import Sequence

import numpy as np
import sep

from vampires_dpp.image_processing import radial_profile_image
from vampires_dpp.indexing import cutout_inds, frame_center

from .image_registration import offset_centroid, offset_modelfit, offset_peak
from .psf_models import fit_model
from .util import append_or_create


def safe_aperture_sum(frame, r, center=None, ann_rad=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circle(
        np.ascontiguousarray(frame).astype("f4"),
        (center[1],),
        (center[0],),
        r,
        mask=mask,
        bkgann=ann_rad,
    )

    return flux[0], fluxerr[0]


def estimate_strehl(*args, **kwargs):
    raise NotImplementedError()


def analyze_fields(
    cube,
    inds,
    aper_rad,
    ann_rad=None,
    model="gaussian",
    strehl: bool = False,
    refmethod="com",
    **kwargs,
):
    output = {}
    cutout = cube[inds]
    ## Simple statistics
    output["max"] = np.nanmax(cutout, axis=(-2, -1))
    output["sum"] = np.nansum(cutout, axis=(-2, -1))
    output["mean"] = np.nanmean(cutout, axis=(-2, -1))
    output["median"] = np.nanmedian(cutout, axis=(-2, -1))
    output["var"] = np.nanvar(cutout, axis=(-2, -1))
    output["normvar"] = output["var"] / output["mean"]
    if model is not None:
        output["psfmodel"] = np.full(cube.shape[0], model)
    ## Centroids
    for frame in cube:
        pk = offset_peak(frame, inds)
        com_ctr = offset_centroid(frame, inds)
        if refmethod == "com":
            ctr_est = com_ctr
        elif refmethod == "peak":
            ctr_est = pk
        append_or_create(output, "com_x", com_ctr[1])
        append_or_create(output, "com_y", com_ctr[0])
        append_or_create(output, "peak_x", pk[1])
        append_or_create(output, "peak_y", pk[0])
        if aper_rad is not None:
            append_or_create(output, "photrad", aper_rad)
            phot, photerr = safe_aperture_sum(frame, r=aper_rad, center=ctr_est, ann_rad=ann_rad)
            append_or_create(output, "photflux", phot)
            append_or_create(output, "photerr", photerr)

        ## fit PSF to center
        if model is not None:
            model_fit = fit_model(frame, inds, model)
            append_or_create(output, "model_x", model_fit["x"])
            append_or_create(output, "model_y", model_fit["y"])
            append_or_create(output, "model_fwhm", model_fit["fwhm"])
            append_or_create(output, "model_amp", model_fit["amplitude"])
    return output


def get_center(frame, centroid, cam_num):
    # IMPORTANT we need to flip the centroids for cam1 since they
    # are saved from raw data but we have y-flipped the data
    # during calibration

    if cam_num == 2:
        return centroid
    # for cam 1 data, need to flip coordinate about x-axis
    Ny = frame.shape[-2]
    ctr = np.asarray((Ny - 1 - centroid[0], centroid[1]))
    return ctr


def analyze_file(
    hdu,
    outpath,
    centroids,
    subtract_radprof: bool = False,
    model="gaussian",
    refmethod="com",
    aper_rad=None,
    ann_rad=None,
    strehl=False,
    force=False,
    **kwargs,
):
    if not force and outpath.is_file():
        return outpath

    data = hdu.data
    ## subtract radial profile
    if subtract_radprof:
        mean_frame = np.nanmean(data, axis=0)
        radial_profile_image(mean_frame)
    cam_num = hdu.header["U_CAMERA"]
    metrics: dict[str, Sequence[Sequence]] = {}
    for ctrs in centroids.values():
        for ctr in ctrs:
            inds = cutout_inds(data, center=get_center(data, ctr, cam_num), **kwargs)
            results = analyze_fields(
                data,
                header=hdu.header,
                inds=inds,
                model=model,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                strehl=strehl,
                refmethod=refmethod,
            )
            for k, v in results.items():
                append_or_create(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath
