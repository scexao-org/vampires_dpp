# lucky imaging library functions
import numpy as np

from .satellite_spots import cutout_slice, window_slices

# from .satellite_spots import window_slice


def lucky_image(
    cube,
    q=0,
    metric="l2norm",
    register="max",
    refidx=None,
    window=None,
    mask=None,
    **kwargs,
):
    """
    Traditional lucky imaging

    Parameters
    ----------
    cube : np.ndarray
        Input data
    q : float, optional
        The quantile for frame selection, by default 0. Must be between 0 and 1.
    register : str, optional
        The algorithm for registering frames. "max" will align the peak pixel to the center, "com" will align the center of mass, and "dft" uses phase cross-correlation for registering frames. If None, will assume the cube is already coregistered. By default "max".
    metric : str, optional
        The metric for which frame selection will be applied. The available metrics are "max" and "l2norm". By default "l2norm".
    refidx: ind, optional
        If using DFT cross-correlation for registration, sets the reference index, or "best frame". If None, will find using the lucky-imaging metric, by default None.
    mask : boolean array or list of boolean arrays, optional
        If a mask is provided (trues/ones are included, falses/zeros are excluded), the metric will only be calculated within the masked region, and the registration will only use this area for centroiding. If a list of masks is provided, it is assumed that each mask corresponds to a separate PSF (e.g., from satellite spots). In this case, the metric will be averaged from each region, and the centroid will be averaged between each region. By default, None.
    **kwargs
        additional keyword arguments will be passed to the registration method (e.g. `upsample_factor`)
    """
    if q < 0 or q >= 1:
        raise ValueError(
            "The frame selection quantile must be less than or equal to 0 (no discard) and less than 1"
        )


def measure_metric(cube, metric="l2norm", center=None, window=None):
    if window is not None:
        inds = cutout_slice(cube[0], center=center, window=window)
        view = cube[..., inds[0], inds[1]]
    else:
        view = cube

    if metric == "max":
        values = np.max(view, axis=(-2, -1))
    elif metric == "l2norm":
        values = np.mean(view**2, axis=(-2, -1))
    elif metric == "normvar":
        var = np.var(view, axis=(-2, -1))
        mean = np.mean(view, axis=(-2, -1))
        values = var / mean
    return values


def measure_satellite_spot_metrics(cube, metric="l2norm", **kwargs):
    slices = window_slices(cube[0], **kwargs)
    values = np.zeros_like(cube, shape=(cube.shape[0]))
    for sl in slices:
        view = cube[..., sl[0], sl[1]]
        if metric == "max":
            values += np.max(view, axis=(-2, -1))
        elif metric == "l2norm":
            values += np.mean(view**2, axis=(-2, -1))
        elif metric == "normvar":
            var = np.var(view, axis=(-2, -1))
            mean = np.mean(view, axis=(-2, -1))
            values += var / mean
    return values / len(slices)
