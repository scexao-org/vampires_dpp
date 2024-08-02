import warnings

import numpy as np
import sep
from astropy.nddata import Cutout2D
from skimage.registration import phase_cross_correlation

from .synthpsf import create_synth_psf


def find_peak(image, xc, yc, boxsize, oversamp=8):
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

    boxhalf = np.ceil(boxsize / 2.0).astype(int)
    boxsize = 2 * boxhalf
    ext = np.array(boxsize * oversamp, dtype=int)

    # need to deconvolve the image by dividing by a sinc in order to "undo" the sampling
    fftsinc = np.zeros(ext)
    fftsinc[0:oversamp] = 1.0

    sinc = (
        boxsize
        * np.fft.fft(fftsinc, norm="forward")
        * np.exp(
            1j * np.pi * (oversamp - 1) * np.roll(np.arange(-ext / 2, ext / 2), int(ext / 2)) / ext
        )
    )
    sinc = sinc.real
    sinc = np.roll(sinc, int(ext / 2))
    sinc = sinc[int(ext / 2) - int(boxsize / 2) : int(ext / 2) + int(boxsize / 2)]
    sinc2d = np.outer(sinc, sinc)

    # define a box around the center of the star
    blx = np.floor(xc - boxhalf).astype(int)
    bly = np.floor(yc - boxhalf).astype(int)

    # make sure that the box is contained by the image
    blx = np.clip(blx, 0, image.shape[0] - boxsize)
    bly = np.clip(bly, 0, image.shape[1] - boxsize)

    # extract the star
    subim = image[blx : blx + boxsize, bly : bly + boxsize]

    # deconvolve the image by dividing by a sinc in order to "undo" the pixelation
    fftim1 = np.fft.fft2(subim, norm="forward")
    shfftim1 = np.roll(fftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    shfftim1 /= sinc2d  # deconvolve

    zpshfftim1 = np.zeros((oversamp * boxsize, oversamp * boxsize), dtype="complex64")
    zpshfftim1[0:boxsize, 0:boxsize] = shfftim1

    zpfftim1 = np.roll(zpshfftim1, (-boxhalf, -boxhalf), axis=(1, 0))
    subimupsamp = np.fft.ifft2(zpfftim1, norm="forward").real

    peak = np.nanmax(subimupsamp)

    return peak


def measure_strehl(image, psf_model, pos=None, phot_rad=0.5, peak_search_rad=0.1, dft_factor: int=30, pxscale=5.9):
    ## Step 1: find approximate location of PSF in image

    # If no position given, start at the nan-max
    if pos is None:
        pos = np.unravel_index(np.nanargmax(image), image.shape)
    center = np.array(pos)
    # Now, refine this centroid using cross-correlation
    # this cutout must have same shape as PSF reference (chance for errors here)
    cutout = Cutout2D(image, center[::-1], psf_model.shape, mode="partial")
    assert cutout.data.shape == psf_model.shape

    shift, _, _ = phase_cross_correlation(
        psf_model.astype("=f4"), cutout.data.astype("=f4"), upsample_factor=dft_factor, normalization=None
    )
    refined_center = center + shift
    if np.any(np.abs(refined_center - center) > 5):
        msg = f"PSF centroid appears to have failed, got {refined_center!r}"
        warnings.warn(msg, stacklevel=2)

    ## Step 2: Calculate peak flux with subsampling and flux
    aper_rad_px = phot_rad / (pxscale * 1e-3)
    image_flux, image_fluxerr, _ = sep.sum_circle(
        image.astype("=f4"),
        (refined_center[1],),
        (refined_center[0],),
        aper_rad_px,
        err=np.sqrt(np.maximum(image, 0)),
    )
    peak_search_rad_px = peak_search_rad / (pxscale * 1e-3)
    image_peak = find_peak(image, refined_center[0], refined_center[1], peak_search_rad_px)

    ## Step 3: Calculate flux of PSF model
    # note: our models are alrady centered
    model_center = np.array(psf_model.shape[-2:]) / 2 - 0.5
    # note: our models have zero background signal
    model_flux, model_fluxerr, _ = sep.sum_circle(
        psf_model.astype("=f4"),
        (model_center[1],),
        (model_center[0],),
        aper_rad_px,
        err=np.sqrt(np.maximum(psf_model, 0)),
    )
    model_peak = find_peak(psf_model, model_center[0], model_center[1], peak_search_rad_px)

    ## Step 4: Calculate Strehl via normalized ratio
    image_norm_peak = image_peak / image_flux[0]
    model_norm_peak = model_peak / model_flux[0]
    strehl = image_norm_peak / model_norm_peak
    return strehl


def get_mbi_cutout(data, camera: int, field: str, reduced: bool = False):
    hy, hx = np.array(data.shape[-2:]) / 2 - 0.5
    # use cam2 as reference
    match field:
        case "F610":
            x = hx * 0.25
            y = hy * 1.5
        case "F670":
            x = hx * 0.25
            y = hy * 0.5
        case "F720":
            x = hx * 0.75
            y = hy * 0.5
        case "F760":
            x = hx * 1.75
            y = hy * 0.5
        case _:
            msg = f"Invalid MBI field {field}"
            raise ValueError(msg)
    if reduced:
        y *= 2
    # flip y axis for cam 1 indices
    if camera == 1:
        y = data.shape[-2] - y
    return Cutout2D(data, position=(x, y), size=500, mode="partial")


def measure_strehl_mbi(image, cam: int, pxscale: float = 5.9, **kwargs):
    filters = ("F610", "F670", "F720", "F760")
    results = {}
    for _i, filt in enumerate(filters):
        psf = create_synth_psf(filt, 201, pixel_scale=pxscale)
        cutout = get_mbi_cutout(image, cam, filt)
        results[filt] = measure_strehl(cutout.data, psf, pxscale=pxscale, **kwargs)
        print(f"{filt}: measured Strehl {results[filt]*100:.01f}%")
    return results


def measure_strehl_frame(frame, header, psf=None, **kwargs):
    pxscale = header["PXSCALE"]
    if "MBI" in header["OBS-MOD"] or header["U_MBI"].strip() == "DICHROICS":
        return measure_strehl_mbi(frame, cam=header["U_CAMERA"], pxscale=pxscale, **kwargs)

    # return image
    if psf is None:
        psf = create_synth_psf(header, header["FILTER01"].strip(), 201)
    if header["U_CAMERA"] == 1:
        psf = np.flipud(psf)
    return measure_strehl(frame, psf, pxscale=pxscale, **kwargs)
