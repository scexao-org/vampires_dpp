import numpy as np
from astropy.modeling import fitting, models

from vampires_dpp.indexing import frame_center


def fit_model(frame, inds, model, fitter=fitting.LevMarLSQFitter()):
    model = model.lower()
    view = frame[inds[0], inds[1]]
    y, x = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    view_center = frame_center(view)
    peak = np.quantile(view.ravel(), 0.9)
    if model == "moffat":
        # bounds = {"amplitude": (0, peak), "gamma": (1, 15), "alpha": }
        model_init = models.Moffat2D(
            amplitude=peak, x_0=view_center[1], y_0=view_center[0], gamma=2, alpha=2
        )
    elif model == "gaussian":
        model_init = models.Gaussian2D(
            amplitude=peak,
            x_mean=view_center[1],
            y_mean=view_center[0],
            x_stddev=2,
            y_stddev=2,
        )
    elif model == "airydisk":
        model_init = models.AiryDisk2D(
            amplitude=peak, x_0=view_center[1], y_0=view_center[0], radius=2
        )

    model_fit = fitter(model_init, x, y, view)

    # offset the model centroids
    model_dict = {"amplitude": model_fit.amplitude.value}
    if model == "moffat":
        model_dict["y"] = model_fit.y_0.value + inds[0].start
        model_dict["x"] = model_fit.x_0.value + inds[1].start
    elif model == "airydisk":
        model_dict["y"] = model_fit.y_0.value + inds[0].start
        model_dict["x"] = model_fit.x_0.value + inds[1].start
    elif model == "gaussian":
        model_dict["y"] = model_fit.y_mean.value + inds[0].start
        model_dict["x"] = model_fit.x_mean.value + inds[1].start

    return model_dict
