# lucky imaging library functions

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.measure import centroid
from skimage.transform import AffineTransform, warp
from scipy.ndimage import fourier_shift


from .image_processing import frame_center, shift_frame

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
    # get endianness correct
    tmp_cube = cube.byteswap().newbyteorder()
    # do frame selection
    if q > 0:
        values = measure_metric(tmp_cube, metric, mask)
        cut = np.quantile(values, q)
        tmp_cube = tmp_cube[values >= cut]

    center = np.asarray(frame_center(tmp_cube))
    # if using DFT upsampling, choose a reference index, or "best frame"
    if register == "dft" and refidx is None:
        if q > 0:
            refidx = values.argmax()
        else:
            refidx = measure_metric(tmp_cube, metric, mask).argmax()
        # get COM (with masks)
        refframe = tmp_cube[refidx]
        if mask is not None and not multi_window:
            refframe_masked = refframe[mask]
            length = int(np.sqrt(refframe_masked.size))
            data = refframe_masked.reshape(length, length)
            # fix offset from mask
            offset = (np.unravel_index(mask.argmin(), mask.shape) - length) // 2
            peakidx = centroid(data) + offset
        elif mask is not None:
            peakidx = 0
            for m in mask:
                refframe_masked = refframe[m]
                length = int(np.sqrt(refframe_masked.size))
                data = refframe_masked.reshape(length, length)
                # fix offset from mask
                offset = (np.unravel_index(m.argmin(), m.shape) - length) // 2
                peakidx += centroid(data) + offset
        else:
            peakidx = centroid(refframe)
        refshift = center - peakidx

    out = np.zeros(tmp_cube.shape[1:], "f4")
    # for each frame in time
    N = tmp_cube.shape[0]
    for i in range(tmp_cube.shape[0]):
        frame = tmp_cube[i]
        if mask is not None and not multi_window:
            frame_masked = frame[mask]
            # measure offset
            length = int(np.sqrt(frame_masked.size))
            if register == "max":
                mask_offset = (
                    np.unravel_index(mask.argmin(), mask.shape) - length
                ) // 2
                idx = (
                    np.unravel_index(np.argmax(frame_masked), frame_masked.shape)
                    + mask_offset
                )
                delta = center - idx
            elif register == "com":
                mask_offset = (
                    np.unravel_index(mask.argmin(), mask.shape) - length
                ) // 2
                idx = centroid(frame) + mask_offset
                delta = center - idx
            elif register == "dft":
                offset = phase_cross_correlation(
                    refframe, frame, return_error=False, **kwargs
                )
                delta = refshift + offset
                # print(f"offset: {offset}")
                # print(f"delta: {delta}")
        elif mask is not None:
            avg_delta = 0
            for m in mask:
                frame_masked = frame[m]
                # measure offset
                length = int(np.sqrt(frame_masked.size))
                if register == "max":
                    mask_offset = (np.unravel_index(m.argmin(), m.shape) - length) // 2
                    idx = (
                        np.unravel_index(np.argmax(frame_masked), frame_masked.shape)
                        + mask_offset
                    )
                    avg_delta += center - idx
                elif register == "com":
                    mask_offset = (np.unravel_index(m.argmin(), m.shape) - length) // 2
                    idx = centroid(frame) + mask_offset
                    avg_delta += center - idx
                elif register == "dft":
                    offset = phase_cross_correlation(
                        refframe, frame, return_error=False, **kwargs
                    )
                    avg_delta += refshift + offset
                    # print(f"offset: {offset}")
                    # print(f"delta: {delta}")
                delta = avg_delta / len(mask)
        else:
            # measure offset
            if register == "max":
                idx = np.unravel_index(np.argmax(frame), frame.shape)
                delta = center - idx
            elif register == "com":
                idx = centroid(frame)
                delta = center - idx
            elif register == "dft":
                offset = phase_cross_correlation(
                    refframe, frame, return_error=False, **kwargs
                )
                delta = refshift + offset
                # print(f"offset: {offset}")
                # print(f"delta: {delta}")
        # shift frame using Fourier phase offset
        shifted = shift_frame(frame, delta)
        # update mean online
        out += shifted / N

    return out


def measure_metric(cube, metric="l2norm", mask=None):
    if metric not in ("max", "l2norm", "normvar"):
        raise ValueError(
            f"Did not recognize frame selection metric {metric}. Please choose between 'max' and 'l2norm'."
        )
    if metric == "max":
        values = np.max(cube, axis=(-2, -1))
    elif metric == "l2norm":
        values = np.mean(cube**2, axis=(-2, -1))
    elif metric == "normvar":
        var = np.var(cube, axis=(-2, -1))
        mean = np.mean(cube, axis=(-2, -1))
        values = var / mean
    return values
