import numpy as np
import sep

from .image_processing import radial_profile_image
from .image_registration import offset_centroid, offset_modelfit, offset_peak
from .indexing import cutout_inds, frame_center
from .util import append_or_create, get_center


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
    fit_model: bool=True,
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
    output["med"] = np.nanmedian(cutout, axis=(-2, -1))
    output["var"] = np.nanvar(cutout, axis=(-2, -1))
    output["nvar"] = output["var"] / output["mean"]
    if fit_model:
        output["psfmod"] = np.full(cube.shape[0], model)
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
        append_or_create(output, "pk_x", pk[1])
        append_or_create(output, "pk_y", pk[0])
        if aper_rad is not None:
            append_or_create(output, "photr", aper_rad)
            phot, photerr = safe_aperture_sum(frame, r=aper_rad, center=ctr_est, ann_rad=ann_rad)
            append_or_create(output, "photf", phot)
            append_or_create(output, "phote", photerr)

        ## fit PSF to center
        if fit_model:
            model_fit = offset_modelfit(frame, inds, model=model)
            append_or_create(output, "mod_x", model_fit["x"])
            append_or_create(output, "mod_y", model_fit["y"])
            append_or_create(output, "mod_w", model_fit["fwhm"])
            append_or_create(output, "mod_a", model_fit["amplitude"])
    return output


def analyze_file(
    hdu,
    outpath,
    centroids,
    subtract_radprof: bool = False,
    fit_model: bool = True,
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
        data -= radial_profile_image(mean_frame)

    cam_num = hdu.header["U_CAMERA"]
    metrics: dict[str, list[list[list]]] = {}
    for field, ctrs in centroids.items():
        field_metrics = {}
        for ctr in ctrs:
            inds = cutout_inds(data, center=get_center(data, ctr, cam_num), **kwargs)
            results = analyze_fields(
                data,
                header=hdu.header,
                inds=inds,
                fit_model=fit_model,
                model=model,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                strehl=strehl,
                refmethod=refmethod,
            )
            # append psf result to this field's dictionary
            for k, v in results.items():
                append_or_create(field_metrics, k, v)
        # append this field's results to the global output
        for k, v in field_metrics.items():
            append_or_create(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath
