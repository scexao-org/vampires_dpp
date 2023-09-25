import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from numpy.typing import ArrayLike
from skimage.measure import centroid
from skimage.registration import phase_cross_correlation

from .indexing import frame_center
from .psf_models import fit_model_airy, fit_model_gaussian, fit_model_moffat


def offset_dft(frame, inds, psf, *, upsample_factor):
    cutout = frame[inds]
    dft_offset = phase_cross_correlation(
        psf, cutout, return_error=False, upsample_factor=upsample_factor
    )
    ctr = np.array(frame_center(psf)) - dft_offset
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_centroid(frame, inds):
    """NaN-friendly centroid"""
    # wy, wx = np.ogrid[inds[-2], inds[-1]]
    cutout = frame[inds]
    ctr = centroid(cutout)
    # mask = np.isfinite(cutout)
    # cy = np.sum(wy * cutout, where=mask)
    # cx = np.sum(wx * cutout, where=mask)
    # ctr = np.asarray((cy, cx)) / np.sum(cutout, where=mask)
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_peak(frame, inds):
    view = frame[inds]
    ctr = np.asarray(np.unravel_index(np.nanargmax(view), view.shape))
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_modelfit(frame, inds, *, model="gaussian", fitter=fitting.LevMarLSQFitter()):
    model = model.lower()
    view = frame[inds]

    match model:
        case "gaussian":
            model_dict = fit_model_gaussian(view, fitter=fitter)
        case "moffat":
            model_dict = fit_model_moffat(view, fitter=fitter)
        case "airy":
            model_dict = fit_model_airy(view, fitter=fitter)

    # fix the offsetting due to the indices
    model_dict["y"] += inds[-2].start
    model_dict["x"] += inds[-1].start
    return model_dict
