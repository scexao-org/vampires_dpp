import itertools
from typing import Final, Literal

import numpy as np
import scipy.stats as st
import sep
from astropy import modeling
from astropy.io import fits
from astropy.nddata import Cutout2D

# import time
from .indexing import frame_center, get_mbi_centers
from .registration import offset_dft, offset_peak_and_com
from .util import create_or_append, get_center
from .constants import NBS_INSTALL_MJD


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
    template=None,
    do_psf_model: bool = False,
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
        if do_psf_model:
            psf_info = fit_psf_model(frame, frame_err, model=psf_model)
            create_or_append(output, "modelx", psf_info["model_x"])
            create_or_append(output, "modely", psf_info["model_y"])
            ctr_est = psf_info["model_y"], psf_info["model_x"]
        if template is not None:
            dft_ctrs = offset_dft(frame, inds, psf=template)
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
            strehl = measure_strehl(frame, psf, pos=ctr_est, phot_rad=aper_rad)
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
    # bad strehls become -1
    if strehl < 0 or strehl > 1:
        return np.nan

    return strehl


## This is not my code, I'd love to see ways to improve it
def find_norm_peak(image, center, window_size=20, phot_rad=8, oversamp=4) -> float:
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
    blx = np.clip(blx, 0, image.shape[1] - window_size)
    bly = np.clip(bly, 0, image.shape[0] - window_size)

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
    nbs_flag = hdr["MJD"] > NBS_INSTALL_MJD
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
            center = get_center(data, ctr, cam_num, nbs_flag=nbs_flag)
            _inds = Cutout2D(data[0], center[::-1], window_size, mode="partial").slices_original
            inds = np.s_[..., _inds[0], _inds[1]]
            # inds = cutout_inds(data, center=get_center(data, ctr, cam_num), window=window_size)
            # cutouts = [
            #     Cutout2D(frame, center[::-1], window_size, mode="partial").data for frame in data
            # ]
            template = psf  # bn.median(cutouts, axis=0)
            results = analyze_fields(
                data,
                data_err,
                inds=inds,
                aper_rad=aper_rad,
                ann_rad=ann_rad,
                do_phot=do_phot,
                do_strehl=do_strehl,
                psf=psf,
                template=template,
                do_psf_model=fit_psf_model,
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


COMMENT_FSTRS: Final = {
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
CENTROID_COMM_FSTRS: Final = {
    "comx": "[pix] COM x{}in window {}",
    "comy": "[pix] COM y{}in window {}",
    "peakx": "[pix] Peak index x{}in window {}",
    "peaky": "[pix] Peak index y{}in window {}",
    "modx": "[pix] Model fit x{}in window {}",
    "mody": "[pix] Model fit y{}in window {}",
    "dftx": "[pix] Cross-corr. x{}in window {}",
    "dfty": "[pix] Cross-corr. y{}in window {}",
    "fwhm": "[pix] Model fit fwhm{}in window {}",
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


def moffat_fwhm(gamma, alpha):
    return 2 * gamma * np.sqrt(2 ** (1 / alpha) - 1)


def moffat_fwhm_err(gamma, gamma_err, alpha, alpha_err):
    d_gamma = 2 * np.sqrt(2 ** (1 / alpha) - 1)
    d_alpha = -np.log(2) * 2 ** (1 / alpha) * gamma / (alpha**2 * np.sqrt(2 ** (1 / alpha) - 1))
    fwhm_err = np.hypot(d_gamma * gamma_err, d_alpha * alpha_err)
    return fwhm_err


def moffat_gamma(fwhm, alpha):
    return fwhm / (2 * np.sqrt(2 ** (1 / alpha) - 1))


## moffat
class Moffat(modeling.Fittable2DModel):
    x0 = modeling.Parameter()
    y0 = modeling.Parameter()
    gammax = modeling.Parameter(default=1, min=0)
    gammay = modeling.Parameter(default=1, min=0)
    theta = modeling.Parameter(default=0, min=-np.pi / 4, max=np.pi / 4)
    alpha = modeling.Parameter(default=1, min=0)
    amplitude = modeling.Parameter(default=1, min=0)
    background = modeling.Parameter(default=0)

    @property
    def fwhmx(self) -> modeling.Parameter:
        return moffat_fwhm(self.gammax, self.alpha)

    @fwhmx.setter
    def fwhmx(self, fwhmx: float):
        self.gammax = moffat_gamma(fwhmx, self.alpha)

    @property
    def fwhmy(self) -> modeling.Parameter:
        return moffat_fwhm(self.gammay, self.alpha)

    @fwhmy.setter
    def fwhmy(self, fwhmy: float):
        self.gammay = moffat_gamma(fwhmy, self.alpha)

    @staticmethod
    def evaluate(x, y, x0, y0, gammax, gammay, theta, alpha, amplitude, background):
        diffx = x - x0
        diffy = y - y0

        cost = np.cos(theta)
        sint = np.sin(theta)

        a = (cost / gammax) ** 2 + (sint / gammay) ** 2
        b = (sint / gammax) ** 2 + (cost / gammay) ** 2
        c = 2 * sint * cost * (1 / gammax**2 - 1 / gammay**2)

        rad = a * diffx**2 + b * diffy**2 + c * diffx * diffy
        return amplitude / (1 + rad) ** alpha + background

    @staticmethod
    def fit_deriv(x, y, x0, y0, gammax, gammay, theta, alpha, amplitude, background):
        diffx = x - x0
        diffy = y - y0

        cost = np.cos(theta)
        sint = np.sin(theta)
        cos2t = np.cos(2 * theta)
        sin2t = np.sin(2 * theta)

        a = (cost / gammax) ** 2 + (sint / gammay) ** 2
        b = (sint / gammax) ** 2 + (cost / gammay) ** 2
        inv_gamma2 = 1 / gammax**2 - 1 / gammay**2
        c = 2 * sint * cost * inv_gamma2

        rad = a * diffx**2 + b * diffy**2 + c * diffx * diffy

        d_amp = (1 + rad) ** (-alpha)
        d_alpha = -amplitude * d_amp * np.log(1 + rad)

        f = -amplitude * alpha * (1 + rad) ** (-alpha - 1)
        d_x0 = f * (-2 * diffx * a - diffy * c)
        d_y0 = f * (-2 * diffy * b - diffx * c)
        d_theta = f * (
            diffx**2 * sin2t * inv_gamma2
            + 2 * diffx * diffy * inv_gamma2 * cos2t
            - diffy**2 * inv_gamma2 * sin2t
        )
        d_gammax = f * (
            -2 / gammax**3 * (cost**2 * diffx**2 + sint**2 * diffy**2 + diffx * diffy * sin2t)
        )
        d_gammay = f * (
            -2 / gammay**3 * (sint**2 * diffx**2 + cost**2 * diffy**2 - diffx * diffy * sin2t)
        )
        d_back = np.ones_like(d_x0)

        return [d_x0, d_y0, d_gammax, d_gammay, d_theta, d_alpha, d_amp, d_back]


def fit_psf_model(frame, err=None, model: Literal["moffat"] = "moffat") -> dict:
    assert model == "moffat"
    weights = 1 / np.abs(err) if err is not None else None

    Ny, Nx = frame.shape
    ys, xs = np.indices(frame.shape)
    fitter = modeling.fitting.LevMarLSQFitter(calc_uncertainties=False)
    inity, initx = np.unravel_index(frame.argmax(), frame.shape)
    model = Moffat(
        x0=initx,
        y0=inity,
        alpha=2,
        gammax=10,
        gammay=10,
        theta=0,
        amplitude=frame.max(),
        background=0,
    )
    model.x0.min = 0
    model.x0.max = Nx
    model.y0.min = 0
    model.y0.max = Ny
    model.alpha.min = 0.5
    model.alpha.max = 3
    model.gammax.min = 1
    model.gammax.max = Nx / 2
    model.gammay.min = 1
    model.gammay.max = Ny / 2
    model.theta.min = -np.pi / 4
    model.theta.max = np.pi / 4
    model.amplitude.min = 0
    model.amplitude.max = 2 * frame.max()
    model.background.fixed = True
    fit_model = fitter(model, xs, ys, frame, weights=weights, filter_non_finite=True, maxiter=5000)
    # re-offset position
    return {
        "model_amp": fit_model.amplitude,
        "model_x": fit_model.x0,
        "model_y": fit_model.y0,
        "model_fwhmx": fit_model.fwhmx,
        "model_fwhmy": fit_model.fwhmy,
        "model_alpha": fit_model.alpha,
        "model_bkg": fit_model.background,
    }
