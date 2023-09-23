import numpy as np
from astropy.modeling import fitting, models

from vampires_dpp.indexing import frame_center


def fit_model(frame, inds, model, fitter=fitting.LevMarLSQFitter()):
    model = model.lower()
    view = frame[inds]
    y, x = np.mgrid[0 : view.shape[-1], 0 : view.shape[-2]]
    view_center = frame_center(view)
    peak = np.quantile(view.ravel(), 0.9)

    if model == "gaussian":
        model_init = models.Gaussian2D(
            amplitude=peak,
            x_mean=view_center[-1],
            y_mean=view_center[-2],
            x_stddev=2,
            y_stddev=2,
            bounds=dict(
                x_mean=(-0.5, view.shape[-1] - 0.5),
                y_mean=(-0.5, view.shape[-2] - 0.5),
                x_stddev=(2 / 2.355, view.shape[-1] / 2 / 2.355),
                y_stddev=(2 / 2.355, view.shape[-2] / 2 / 2.355),
            ),
        )
    elif model == "moffat":
        # bounds = {"amplitude": (0, peak), "gamma": (1, 15), "alpha": }
        model_init = models.Moffat2D(
            amplitude=peak,
            x_0=view_center[-1],
            y_0=view_center[-2],
            gamma=2,
            alpha=2,
            bounds=dict(
                x_0=(-0.5, view.shape[-1] - 0.5),
                y_0=(-0.5, view.shape[-2] - 0.5),
                gamma=(1, view.shape[-1] / 4),
            ),
        )
    elif model == "airydisk":
        model_init = models.AiryDisk2D(
            amplitude=peak,
            x_0=view_center[-1],
            y_0=view_center[-2],
            radius=2,
            bounds=dict(
                x_0=(-0.5, view.shape[-1] - 0.5),
                y_0=(-0.5, view.shape[-2] - 0.5),
                radius=(2 / 0.8038, view.shape[-1] / 2 / 0.8038),
            ),
        )

    model_fit = fitter(model_init, x, y, view, filter_non_finite=True)

    # offset the model centroids
    model_dict = {"amplitude": model_fit.amplitude.value}
    if model == "moffat":
        model_dict["y"] = model_fit.y_0.value + inds[-2].start
        model_dict["x"] = model_fit.x_0.value + inds[-1].start
        model_dict["fwhm"] = 2 * model_fit.gamma.value
    elif model == "airydisk":
        model_dict["y"] = model_fit.y_0.value + inds[-2].start
        model_dict["x"] = model_fit.x_0.value + inds[-1].start
        model_dict["fwhm"] = 0.8038 * model_fit.radius
    elif model == "gaussian":
        model_dict["y"] = model_fit.y_mean.value + inds[-2].start
        model_dict["x"] = model_fit.x_mean.value + inds[-1].start
        model_dict["fwhm"] = 2.355 * np.hypot(model_fit.x_stddev.value, model_fit.y_stddev.value)
    return model_dict
