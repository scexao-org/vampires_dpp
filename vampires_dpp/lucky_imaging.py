# lucky imaging library functions

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.measure import centroid
from skimage.transform import AffineTransform, warp
from scipy.ndimage import fourier_shift


from .image_processing import frame_center, shift_frame


def lucky_image(
    cube, q=0, metric="max", register="max", window=None, refidx=None, **kwargs
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
    window : int, optional
        If not None, will only measure the frame selection metric in a centered box with this width, by default None.
    refidx: ind, optional
        If using DFT cross-correlation for registration, sets the reference index, or "best frame". If None, will find using the lucky-imaging metric, by default None.
    **kwargs
        additional keyword arguments will be passed to the registration method (e.g. `upsample_factor`)
    """

    if q < 0 or q >= 1:
        raise ValueError(
            "The frame selection quantile must be less than or equal to 0 (no discard) and less than 1"
        )
    # get endianness correct
    tmp_cube = cube.byteswap().newbyteorder()
    # do frame selection
    if q > 0:
        values = measure_metric(tmp_cube, metric)
        cut = np.quantile(values, q)
        tmp_cube = tmp_cube[values >= cut]

    center = frame_center(tmp_cube)
    # if using DFT upsampling, choose a reference index, or "best frame"
    if register == "dft" and refidx is None:
        if q > 0:
            refidx = values.argmax()
        else:
            refidx = measure_metric(tmp_cube, metric).argmax()
        refframe = tmp_cube[refidx]
        refshift = np.unravel_index(refframe.argmax(), refframe.shape) - center

    out = np.zeros(tmp_cube.shape[1:], "f4")
    # for each frame in time
    N = tmp_cube.shape[0]
    for i in range(tmp_cube.shape[0]):
        frame = tmp_cube[i]
        # measure offset
        if register == "max":
            idx = np.unravel_index(np.argmax(frame), frame.shape)
            delta = idx - center
        elif register == "com":
            idx = centroid(frame)
            delta = idx - center
        elif register == "dft":
            delta = refshift - phase_cross_correlation(
                refframe, frame, return_error=False, **kwargs
            )
        # shift frame using Fourier phase offset
        shifted = shift_frame(frame, delta)
        # update mean online
        out += shifted / N

    return out


def measure_metric(cube, metric="l2norm"):
    if metric == "max":
        values = np.max(cube, axis=(1, 2))
    elif metric == "l2norm":
        values = np.mean(cube**2, axis=(1, 2))
    else:
        raise ValueError(
            f"Did not recognize frame selection metric {metric}. Please choose between 'max' and 'l2norm'."
        )

    return values
