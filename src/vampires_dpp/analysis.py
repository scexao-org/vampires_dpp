import numpy as np
import sep

from vampires_dpp.image_processing import radial_profile_image
from vampires_dpp.indexing import cutout_inds, frame_center

from .image_registration import offset_centroid, offset_modelfit, offset_peak


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


def append_or_create(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]


def analyze_fields(cube, inds, aper_rad, ann_rad=None, model="gaussian", **kwargs):
    output = {}
    cutout = cube[inds]
    ## Simple statistics
    output["peak"] = np.nanmax(cutout, axis=0)
    output["sum"] = np.nansum(cutout, axis=0)
    output["mean"] = np.nanmean(cutout, axis=0)
    output["median"] = np.nanmedian(cutout, axis=0)
    output["var"] = np.nanvar(cutout, axis=0)
    output["normvar"] = output["var"] / output["mean"]

    ## Centroids
    for frame in cube:
        pk = offset_peak(cube, inds)
        append_or_create(output, "centroid", offset_centroid(cube, inds))
        append_or_create(output, "peak_centroid", pk)
        if aper_rad is not None:
            append_or_create(output, "photrad", aper_rad)
            phot, photerr = safe_aperture_sum(frame, r=aper_rad, center=pk, ann_rad=ann_rad)
            append_or_create(output, "photflux", phot)
            append_or_create(output, "photerr", photerr)

        ## fit PSF to center
        if model is not None:
            model_fit = offset_modelfit(frame, inds, model)
            append_or_create(output, "model_x", model_fit["x"])
            append_or_create(output, "model_y", model_fit["y"])
            append_or_create(output, "model_fwhm", model_fit["fwhm"])
            append_or_create(output, "model_amp", model_fit["amp"])

    return output


def analyze_file(
    hdu,
    outpath,
    centroids,
    subtract_radprof: bool = False,
    model="gaussian",
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
        mean_frame = np.nanmean(hdu.data, axis=0)
        profile = radial_profile_image(mean_frame)
        data = hdu.data - profile

    metrics: dict[str, list[list]] = {}
    for ctrs in centroids.values():
        for ctr in ctrs:
            inds = cutout_inds(data, center=ctr, **kwargs)
            results = analyze_fields(
                data,
                header=hdu.header,
                inds=inds,
                model=model,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                strehl=strehl,
            )
            for k, v in results.items():
                append_or_create(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath
