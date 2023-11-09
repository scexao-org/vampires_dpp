from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


def frame_center(image: ArrayLike) -> tuple[float, float]:
    """
    Find the center of the frame or cube in pixel coordinates

    Parameters
    ----------
    image : ArrayLike
        N-D array with the final two dimensions as the (y, x) axes.

    Returns
    -------
    (cy, cx)
        A tuple of the image center in pixel coordinates
    """
    ny = image.shape[-2]
    nx = image.shape[-1]
    return (ny - 1) / 2, (nx - 1) / 2


def frame_radii(frame: ArrayLike, center=None) -> NDArray:
    """
    Return the radii of pixels around ``center`` in the image

    Parameters
    ----------
    frame : ArrayLike
        Input frame
    center : Tuple, optional
        The center to calculate radii from. If None, will default to the frame center. By default None

    Returns
    -------
    NDArray
        Matrix with frame radii
    """
    if center is None:
        center = frame_center(frame)
    Ys, Xs = np.ogrid[: frame.shape[-2], : frame.shape[-1]]
    radii = np.hypot(Ys - center[-2], Xs - center[-1])
    return radii


def frame_angles(frame: ArrayLike, center=None, conv: Literal["image", "astro"] = "image"):
    """
    Return the angles of pixels around ``center`` in the image

    Parameters
    ----------
    frame : ArrayLike
        Input frame
    center : Tuple, optional
        The center to calculate radii from. If None, will default to the frame center. By default None
    conv : str, optional
        The convention to use, either "image" (angle increases CCW from +x axis of image) or "astro" (angle increases degrees East of North). By default, "image".

    Returns
    -------
    NDArray
        Matrix with frame angles
    """
    if center is None:
        center = frame_center(frame)

    match conv.lower():
        case "image":
            return frame_angles_image(frame, center)
        case "astro":
            return frame_angles_astro(frame, center)


def frame_angles_image(frame, center):
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    thetas = np.arctan2(Ys - center[-2], Xs - center[-1])
    return thetas


def frame_angles_astro(frame, center):
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    # degrees East of North: phi = arctan(-x, y)
    thetas = np.arctan2(center[-2] - Xs, Ys - center[-1])
    return thetas


# def lamd_to_pixel(ld: float, filter: str = "Open", pxscale: float) -> float:
#     dist = FILTER_ANGULAR_SIZE[filter.strip().lower()]
#     return ld * dist / pxscale


def cutout_inds(frame, window, center=None, **kwargs):
    """
    Get the index slices for a window with size `window` at `center`, clipped to the boundaries of `frame`

    Parameters
    ----------
    frame : ArrayLike
        image frame for bound-checking
    center : Tuple
        (y, x) coordinate of the window
    window : float,Tuple
        window length, or tuple for each axis

    Returns
    -------
    (ys, xs)
        tuple of slices for the indices for the window
    """
    if center is None:
        center = frame_center(frame)
    half_width = np.asarray(window) / 2
    Ny, Nx = frame.shape[-2:]
    lower = np.maximum(0, np.round(center - half_width), dtype=int, casting="unsafe")
    upper = np.minimum((Ny - 1, Nx - 1), np.round(center + half_width), dtype=int, casting="unsafe")
    if np.any(upper - lower + 1 > window):
        upper -= 1
    elif np.any(upper - lower + 1 < window):
        upper += 1
    return np.s_[..., lower[0] : upper[0] + 1, lower[1] : upper[1] + 1]
