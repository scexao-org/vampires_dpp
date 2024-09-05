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
    do_strehl: bool = False,
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
        if do_strehl and psf is not None:
            strehl = measure_strehl(frame, psf, pos=ctr_est)
            create_or_append(output, "strehl", strehl)

    # t2 = time.perf_counter()
    # print(f"Average time for centroids: {(t2 - t1)/cube.shape[0]} [s]")
    return output


def measure_strehl(image, psf_model, pos=None, phot_rad=8):
    if pos is None:
        pos = frame_center(image)

    image_norm_peak = find_norm_peak(image, pos, phot_rad=phot_rad)
    ## Step 3: Calculate flux of PSF model
    # note: our models are alrady centered
    # note: our models have zero background signal
    model_norm_peak = find_norm_peak(psf_model, frame_center(psf_model), phot_rad=phot_rad)
    ## Step 4: Calculate Strehl via normalized ratio
    strehl = image_norm_peak / model_norm_peak
    # bad strehls become negative
    if strehl < 0 or strehl > 1:
        return -1

    return strehl


## This is not my code, I'd love to see ways to improve it
def find_norm_peak(image, center, window_size=10, phot_rad=8, oversamp=4) -> float:
    """
    usage: peak = find_peak(image, xc, yc, boxsize)
    finds the subpixel peak of an image

    image: an image of a point source for which we would like to find the peak
    xc, yc: approximate coordinate of the point source
    boxsize: region in which most of the flux is contained (typically 20)
    oversamp: how many times to oversample the image in the FFT interpolation in order to find the peak

    :return: peak of the oversampled image

    Marcos van Dam, October 2022, translated from IDL code of the same name
    """
    boxhalf = int(np.ceil(window_size / 2))
    window_size = 2 * boxhalf
    ext = np.array(window_size * oversamp, dtype=int)

    # need to deconvolve the image by dividing by a sinc in order to "undo" the sampling
    fftsinc = np.zeros(ext)
    fftsinc[:oversamp] = 1

    sinc = (
        window_size
        * np.fft.fft(fftsinc, norm="forward")
        * np.exp(
            1j * np.pi * (oversamp - 1) * np.roll(np.arange(-ext / 2, ext / 2), int(ext / 2)) / ext
        )
    )
    sinc = sinc.real
    sinc = np.roll(sinc, int(ext / 2))
    sinc = sinc[int(ext / 2) - window_size // 2 : ext // 2 + window_size // 2]
    sinc2d = np.outer(sinc, sinc)

    # define a box around the center of the star
    blx = int(np.floor(center[1] - boxhalf))
    bly = int(np.floor(center[0] - boxhalf))

    # make sure that the box is contained by the image
    blx = np.clip(blx, 0, image.shape[0] - window_size)
    bly = np.clip(bly, 0, image.shape[1] - window_size)

    # extract the star
    subim = image[bly : bly + window_size, blx : blx + window_size]

    # deconvolve the image by dividing by a sinc in order to "undo" the pixelation
    fftim1 = np.fft.fft2(subim, norm="forward")
    shfftim1 = np.roll(fftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    shfftim1 /= sinc2d  # deconvolve

    zpshfftim1 = np.zeros((oversamp * window_size, oversamp * window_size), dtype=np.complex64)
    zpshfftim1[0:window_size, 0:window_size] = shfftim1

    zpfftim1 = np.roll(zpshfftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    subimupsamp = np.real(np.fft.ifft2(zpfftim1, norm="forward"))

    peak = np.nanmax(subimupsamp)

    norm_val, _, _ = sep.sum_circle(
        image.astype("f4"), (center[1],), (center[0],), phot_rad, mask=np.isnan(image)
    )

    return peak / norm_val[0]


def analyze_file(
    hdul,
    outpath,
    centroids,
    aper_rad: int = 4,
    ann_rad=None,
    force=False,
    window_size=21,
    do_phot: bool = True,
    do_strehl: bool = False,
    psfs=None,
    fit_psf_model: bool = False,
    psf_model: Literal["moffat", "gauss"] = "moffat",
):
    if do_strehl and psfs is None:
        msg = "Cannot measure strehl without PSF models!"
        raise ValueError(msg)
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
                do_strehl=do_strehl,
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
