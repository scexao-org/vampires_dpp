import itertools
from typing import Literal

# import time
import numpy as np
import sep

from .image_registration import offset_dft, offset_peak_and_com
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
    _frame = frame.astype("=f4")
    _err = err.astype("=f4") if err is not None else None
    mask = ~np.isfinite(_frame)
    if not ann_rad:
        ann_rad = None
    flux, fluxerr, flag = sep.sum_circle(
        _frame, (center[1],), (center[0],), r, err=_err, mask=mask, bkgann=ann_rad
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
    *,
    do_phot: bool = True,
    aper_rad=4,
    ann_rad=None,
    psf=None,
    fit_psf_model: bool = False,
    psf_model="moffat",
):
    output = {}
    cutout = cube[inds]
    cube_err[inds]
    ## Simple statistics
    # t0 = time.perf_counter()
    output["max"] = np.nanmax(cutout, axis=(-2, -1))
    output["sum"] = np.nansum(cutout, axis=(-2, -1))
    output["mean"] = np.nanmean(cutout, axis=(-2, -1))
    output["med"] = np.nanmedian(cutout, axis=(-2, -1))
    output["var"] = np.nanvar(cutout, axis=(-2, -1))
    output["nvar"] = output["var"] / output["mean"]
    # t1 = time.perf_counter()
    # print(f"Time for full-frame statistics: {t1 - t0} [s]")
    ## Centroids
    for fidx in range(cube.shape[0]):
        frame = cube[fidx]
        frame_err = cube_err[fidx]
        # highpass_frame = frame - filters.median(frame, np.ones((9, 9)))
        # t3 = time.perf_counter()
        centroids = offset_peak_and_com(frame, inds)

        create_or_append(output, "comx", centroids["com"][1])
        create_or_append(output, "comy", centroids["com"][0])
        create_or_append(output, "peakx", centroids["peak"][1])
        create_or_append(output, "peaky", centroids["peak"][0])
        ctr_est = centroids["com"]
        if fit_psf_model:
            msg = "TODO :)"
            raise NotImplementedError(msg)
            # psf_info = fit_psf_model(frame, inds, model=psf_model)
            # create_or_append(output, "gausx", centroids["gauss"][1])
            # create_or_append(output, "gausy", centroids["gauss"][0])
            # ctr_est = centroids["gauss"]
        if psf is not None:
            dft_ctrs = offset_dft(frame, inds, psf=psf)
            create_or_append(output, "dftx", dft_ctrs[1])
            create_or_append(output, "dfty", dft_ctrs[0])
            ctr_est = dft_ctrs

        # t4 = time.perf_counter()
        # print(f"Time to measure centroids for one frame: {t4 - t3} [s]")

        # t3 = time.perf_counter()
        if do_phot:
            create_or_append(output, "photr", aper_rad)
            phot, photerr = safe_aperture_sum(
                frame, r=aper_rad, err=frame_err, center=ctr_est, ann_rad=ann_rad
            )
            create_or_append(output, "photf", phot)
            create_or_append(output, "phote", photerr)
        # t4 = time.perf_counter()
        # print(f"Time to radial profile for one frame: {t4 - t3} [s]")

    # t2 = time.perf_counter()
    # print(f"Average time for centroids: {(t2 - t1)/cube.shape[0]} [s]")
    return output


def analyze_file(
    hdul,
    outpath,
    centroids,
    aper_rad: int = 4,
    ann_rad=None,
    force=False,
    window_size=21,
    do_phot: bool = True,
    psfs=None,
    fit_psf_model: bool = False,
    psf_model: Literal["moffat", "gauss"] = "moffat",
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
            centroids = {"": [frame_center(data)]}
    if psfs is None:
        psfs = itertools.repeat(None)
    for ctrs, psf in zip(centroids.values(), psfs, strict=False):
        field_metrics = {}
        for ctr in ctrs:
            inds = cutout_inds(data, center=get_center(data, ctr, cam_num), window=window_size)
            results = analyze_fields(
                data,
                data_err,
                inds=inds,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                do_phot=do_phot,
                psf=psf,
                fit_psf_model=fit_psf_model,
                psf_model=psf_model,
            )
            # append psf result to this field's dictionary
            for k, v in results.items():
                create_or_append(field_metrics, k, v)
        # append this field's results to the global output
        for k, v in field_metrics.items():
            create_or_append(metrics, k, v)

    np.savez_compressed(outpath, **metrics)
    return outpath


def update_hdul_with_metrics(hdul, metrics):
    # TODO

    return hdul
