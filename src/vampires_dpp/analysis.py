import numpy as np
import sep
from photutils import profiles
from photutils.utils import calc_total_error

from .image_processing import radial_profile_image
from .image_registration import offset_centroids
from .indexing import cutout_inds, frame_center
from .util import append_or_create, get_center


def safe_aperture_sum(frame, r, err=None, center=None, ann_rad=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circle(
        np.ascontiguousarray(frame).astype("f4"),
        (center[1],),
        (center[0],),
        r,
        err=err,
        mask=mask,
        bkgann=ann_rad,
    )

    return flux[0], fluxerr[0]


def safe_annulus_sum(frame, Rin, Rout, center=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circann(
        np.ascontiguousarray(frame).astype("f4"),
        (center[1],),
        (center[0],),
        Rin,
        Rout,
        mask=mask,
    )

    return flux[0], fluxerr[0]


def estimate_strehl(*args, **kwargs):
    raise NotImplementedError()


def analyze_fields(
    cube,
    cube_err,
    inds,
    aper_rad,
    ann_rad=None,
    strehl: bool = False,
    window_size=30,
    **kwargs,
):
    output = {}
    cutout = cube[inds]
    cube_err[inds]
    radii = np.arange(window_size)
    ## Simple statistics
    output["max"] = np.nanmax(cutout, axis=(-2, -1))
    output["sum"] = np.nansum(cutout, axis=(-2, -1))
    output["mean"] = np.nanmean(cutout, axis=(-2, -1))
    output["med"] = np.nanmedian(cutout, axis=(-2, -1))
    output["var"] = np.nanvar(cutout, axis=(-2, -1))
    output["nvar"] = output["var"] / output["mean"]
    ## Centroids
    for fidx in range(cube.shape[0]):
        frame = cube[fidx]
        frame_err = cube_err[fidx]
        centroids = offset_centroids(frame, inds)

        append_or_create(output, "comx", centroids["com"][1])
        append_or_create(output, "comy", centroids["com"][0])
        append_or_create(output, "peakx", centroids["peak"][1])
        append_or_create(output, "peaky", centroids["peak"][0])
        append_or_create(output, "quadx", centroids["quad"][1])
        append_or_create(output, "quady", centroids["quad"][0])
        append_or_create(output, "gausx", centroids["gauss"][1])
        append_or_create(output, "gausy", centroids["gauss"][0])

        prof = profiles.RadialProfile(frame, centroids["quad"][::-1], radii, error=frame_err)
        append_or_create(output, "fwhm", prof.gaussian_fwhm)
        cog = profiles.CurveOfGrowth(frame, centroids["quad"][::-1], radii, error=frame_err)
        snr = cog.profile / cog.profile_error
        aper_rad = cog.radius[np.argmax(snr)]
        append_or_create(output, "photr", aper_rad)
        ann_fwhm = 4
        ann_rad = aper_rad + ann_fwhm, aper_rad + 2 * ann_fwhm
        phot, photerr = safe_aperture_sum(
            frame, err=frame_err, r=aper_rad, center=ctr_est, ann_rad=ann_rad
        )
        append_or_create(output, "photf", phot)
        append_or_create(output, "phote", photerr)

    return output


def analyze_file(
    hdu,
    outpath,
    centroids,
    aper_rad=None,
    ann_rad=None,
    strehl=False,
    force=False,
    window_size=30,
    **kwargs,
):
    if not force and outpath.is_file():
        return outpath

    data = hdu.data
    bkg_noise = (
        np.sqrt(hdu.header["DC"] * hdu.header["EXPTIME"] + hdu.header["RN"] ** 2)
        / hdu.header["GAIN"]
    )
    bkg_err = np.full_like(data, bkg_noise)
    data_err = calc_total_error(data, bkg_err, effective_gain=hdu.header["GAIN"])

    cam_num = hdu.header["U_CAMERA"]
    metrics: dict[str, list[list[list]]] = {}
    if centroids is None:
        centroids = {"": [frame_center(data)]}
    for field, ctrs in centroids.items():
        field_metrics = {}
        for ctr in ctrs:
            inds = cutout_inds(data, center=get_center(data, ctr, cam_num), window=window_size)
            results = analyze_fields(
                data,
                data_err,
                header=hdu.header,
                inds=inds,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                strehl=strehl,
                window_size=window_size,
                **kwargs,
            )
            # append psf result to this field's dictionary
            for k, v in results.items():
                append_or_create(field_metrics, k, v)
        # append this field's results to the global output
        for k, v in field_metrics.items():
            append_or_create(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath
