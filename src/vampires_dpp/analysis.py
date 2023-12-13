import itertools
import warnings

import numpy as np
import sep
from photutils import profiles

from .image_registration import offset_centroids
from .indexing import cutout_inds, frame_center, get_mbi_centers
from .util import create_or_append, get_center


def add_frame_statistics(frame, frame_err, header):
    ## Simple statistics
    unit = header["BUNIT"]
    N = frame.size
    header["TOTMAX"] = np.nanmax(frame), f"[{unit}] Peak signal in frame"
    header["TOTSUM"] = np.nansum(frame), f"[{unit}] Summed signal in frame"
    header["TOTSUME"] = (np.sqrt(np.nansum(frame_err**2)), f"[{unit}] Summed signal error in frame")
    header["TOTMEAN"] = np.nanmean(frame), f"[{unit}] Mean signal in frame"
    header["TOMEANE"] = (np.sqrt(np.nanmean(frame_err**2)), f"[{unit}] Mean signal error in frame")
    header["TOTMED"] = np.nanmedian(frame), f"[{unit}] Median signal in frame"
    header["TOTMEDE"] = (
        header["TOMEANE"] * np.sqrt(np.pi / 2),
        f"[{unit}] Median signal error in frame",
    )
    header["TOTVAR"] = np.nanvar(frame), f"[{unit}^2] Signal variance in frame"
    header["TOTVARE"] = (header["TOTVAR"] / N**2, f"[{unit}^2] Signal variance error in frame")
    header["TOTNVAR"] = (header["TOTVAR"] / header["TOTMEAN"], f"[{unit}] Normed variance in frame")
    header["TONVARE"] = (
        header["TOTNVAR"]
        * np.hypot(header["TOTVARE"] / header["TOTVAR"], header["TOMEANE"] / header["TOTMEAN"]),
        f"[{unit}] Normed variance error in frame",
    )
    return header


def safe_aperture_sum(frame, r, err=None, center=None, ann_rad=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circle(
        np.ascontiguousarray(frame.byteswap().newbyteorder()).astype("f4"),
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
        np.ascontiguousarray(frame.byteswap().newbyteorder()).astype("f4"),
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
    dft_factor=10,
    psf=None,
    **kwargs,
):
    output = {}
    cutout = cube[inds]
    cube_err[inds]
    radii = np.arange(1, window_size)
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
        centroids = offset_centroids(frame, frame_err, inds, psf, dft_factor)

        create_or_append(output, "comx", centroids["com"][1])
        create_or_append(output, "comy", centroids["com"][0])
        create_or_append(output, "peakx", centroids["peak"][1])
        create_or_append(output, "peaky", centroids["peak"][0])
        create_or_append(output, "gausx", centroids["gauss"][1])
        create_or_append(output, "gausy", centroids["gauss"][0])
        if "dft" in centroids:
            create_or_append(output, "dftx", centroids["dft"][1])
            create_or_append(output, "dfty", centroids["dft"][0])

        ctr_est = centroids["com"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prof = profiles.RadialProfile(
                frame, ctr_est[::-1], radii, error=frame_err, mask=np.isnan(frame)
            )
            try:
                fwhm = prof.gaussian_fwhm
            except Exception:
                fwhm = 0
            create_or_append(output, "fwhm", fwhm)

        if aper_rad == "auto":
            r = max(min(fwhm, radii.max() / 2), 3)
            ann_rad = r + 5, r + fwhm + 5
        else:
            r = aper_rad
        create_or_append(output, "photr", r)
        phot, photerr = safe_aperture_sum(
            frame, err=frame_err, r=r, center=ctr_est, ann_rad=ann_rad
        )
        create_or_append(output, "photf", phot)
        create_or_append(output, "phote", photerr)

    return output


def analyze_file(
    hdul,
    outpath,
    centroids,
    aper_rad="auto",
    ann_rad=None,
    strehl=False,
    force=False,
    window_size=30,
    psfs=None,
    **kwargs,
):
    if not force and outpath.is_file():
        return outpath

    data = hdul[0].data
    hdr = hdul[0].header
    data_err = hdul["ERR"].data

    cam_num = hdr["U_CAMERA"]
    metrics: dict[str, list[list[list]]] = {}
    if centroids is None:
        if "MBIR" in hdr["OBS-MOD"]:
            centroids = get_mbi_centers(data, reduced=True)
        elif "MBI" in hdr["OBS-MOD"]:
            centroids = get_mbi_centers(data)
        else:
            centroids = {"": frame_center(data)}
    if psfs is None:
        psfs = itertools.repeat(None)
    for ctrs, psf in zip(centroids.values(), psfs, strict=True):
        field_metrics = {}
        for ctr in ctrs:
            inds = cutout_inds(data, center=get_center(data, ctr, cam_num), window=window_size)
            results = analyze_fields(
                data,
                data_err,
                header=hdr,
                inds=inds,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                strehl=strehl,
                psf=psf,
                window_size=window_size,
                **kwargs,
            )
            # append psf result to this field's dictionary
            for k, v in results.items():
                create_or_append(field_metrics, k, v)
        # append this field's results to the global output
        for k, v in field_metrics.items():
            create_or_append(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath
