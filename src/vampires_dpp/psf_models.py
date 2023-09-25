import numpy as np
from astropy.modeling import fitting, models
from skimage.measure import centroid

from vampires_dpp.indexing import frame_center


def fit_model_gaussian(frame, fitter) -> dict[str, float]:
    y, x = np.mgrid[0 : frame.shape[-1], 0 : frame.shape[-2]]
    center = centroid(frame)
    amp = np.nanmax(frame)

    model_init = models.Gaussian2D(
        amplitude=amp,
        x_mean=center[-1],
        y_mean=center[-2],
        x_stddev=2,
        y_stddev=2,
        bounds=dict(
            x_mean=(-0.5, frame.shape[-1] - 0.5),
            y_mean=(-0.5, frame.shape[-2] - 0.5),
            x_stddev=(2 / 2.355, frame.shape[-1] / 2 / 2.355),
            y_stddev=(2 / 2.355, frame.shape[-2] / 2 / 2.355),
        ),
    )
    model_fit = fitter(model_init, x, y, frame, filter_non_finite=True)
    model_dict = {
        "amplitude": model_fit.amplitude.value,
        "y": model_fit.y_mean.value,
        "x": model_fit.x_mean.value,
        "fwhm": 2.355 * np.hypot(model_fit.x_stddev.value, model_fit.y_stddev.value),
    }
    return model_dict


def fit_model_moffat(frame, fitter) -> dict[str, float]:
    y, x = np.mgrid[0 : frame.shape[-1], 0 : frame.shape[-2]]
    center = centroid(frame)
    amp = np.nanmax(frame)

    model_init = models.Moffat2D(
        amplitude=amp,
        x_0=center[-1],
        y_0=center[-2],
        gamma=2,
        alpha=2,
        bounds=dict(
            x_0=(-0.5, frame.shape[-1] - 0.5),
            y_0=(-0.5, frame.shape[-2] - 0.5),
            gamma=(1, frame.shape[-1] / 4),
        ),
    )
    model_fit = fitter(model_init, x, y, frame, filter_non_finite=True)
    model_dict = {
        "amplitude": model_fit.amplitude.value,
        "y": model_fit.y_0.value,
        "x": model_fit.x_0.value,
        "fwhm": 2 * model_fit.gamma.value,
    }
    return model_dict


def fit_model_airy(frame, fitter) -> dict[str, float]:
    y, x = np.mgrid[0 : frame.shape[-1], 0 : frame.shape[-2]]
    center = centroid(frame)
    amp = np.nanmax(frame)

    model_init = models.AiryDisk2D(
        amplitude=amp,
        x_0=center[-1],
        y_0=center[-2],
        radius=2,
        bounds=dict(
            x_0=(-0.5, frame.shape[-1] - 0.5),
            y_0=(-0.5, frame.shape[-2] - 0.5),
            radius=(2 / 0.8038, frame.shape[-1] / 2 / 0.8038),
        ),
    )
    model_fit = fitter(model_init, x, y, frame, filter_non_finite=True)
    model_dict = {
        "amplitude": model_fit.amplitude.value,
        "y": model_fit.y_0.value,
        "x": model_fit.x_0.value,
        "fwhm": 0.8038 * model_fit.radius,
    }
    return model_dict
