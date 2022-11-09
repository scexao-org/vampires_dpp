import numpy as np
from numpy.typing import ArrayLike

from .image_processing import frame_center

FILTER_ANGULAR_SIZE = {
    "open": np.rad2deg(700e-9 / 7.79) * 3.6e6,
    "625-50": np.rad2deg(625e-9 / 7.79) * 3.6e6,
    "675-50": np.rad2deg(675e-9 / 7.79) * 3.6e6,
    "725-50": np.rad2deg(725e-9 / 7.79) * 3.6e6,
    "750-50": np.rad2deg(750e-9 / 7.79) * 3.6e6,
    "775-50": np.rad2deg(775e-9 / 7.79) * 3.6e6,
}


def lamd_to_pixel(ld, filter="Open", pxscale=6.24):
    dist = FILTER_ANGULAR_SIZE[filter.strip().lower()]
    return ld * dist / pxscale


def window_centers(center, radius, theta=-4, n=4, **kwargs):
    """
    Get the centers (y, x) for each point `radius` away from `center` along `n` branches starting `theta` degrees CCW from the x-axis

    Parameters
    ----------
    center : Tuple
        cross center (y, x)
    radius : float
        radius of the spot separation, in pixels
    theta : float, optional
        Offset the branches by the given number of degrees CCW from the x-axis, by default 0
    n : int, optional
        the number of branches, by default 4

    Returns
    -------
    centers
        list of centers (y, x) for each spot
    """
    # get the angles for each branch
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.deg2rad(theta)
    xs = radius * np.cos(theta) + center[1]
    ys = radius * np.sin(theta) + center[0]
    return list(zip(ys, xs))


def cutout_slice(frame, window, center=None):
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
    upper = np.minimum(
        (Ny - 1, Nx - 1), np.round(center + half_width), dtype=int, casting="unsafe"
    )
    return slice(lower[0], upper[0] + 1), slice(lower[1], upper[1] + 1)


def cart_coords(ys, xs):
    Xg, Yg = np.meshgrid(ys, xs)
    return np.column_stack((Yg.ravel(), Xg.ravel()))


def window_slices(frame, window=30, center=None, **kwargs):
    """
    Get the linear indices for each satellite spot

    Parameters
    ----------
    frame : ArrayLike
        image frame
    window : float, Tuple, optional
        window size, or tuple for each axis, by default 30
    center : Tuple, optional
        (y, x) coordinate of cross center, by default None, which defaults to the frame center.
    **kwargs
        Extra keyword arguments are passed to `window_centers`

    Returns
    -------
    list
        List of linear indices for each spot
    """
    if center is None:
        center = frame_center(frame)
    centers = window_centers(center, **kwargs)
    slices = [cutout_slice(frame, center=cent, window=window) for cent in centers]
    return slices


def window_indices(frame, window=30, center=None, **kwargs):
    """
    Get the linear indices for each satellite spot

    Parameters
    ----------
    frame : ArrayLike
        image frame
    window : float, Tuple, optional
        window size, or tuple for each axis, by default 30
    center : Tuple, optional
        (y, x) coordinate of cross center, by default None, which defaults to the frame center.
    **kwargs
        Extra keyword arguments are passed to `window_centers`

    Returns
    -------
    list
        List of linear indices for each spot
    """
    slices = window_slices(frame, window=window, center=center, **kwargs)
    coords = (cart_coords(sl[0], sl[1]) for sl in slices)
    inds = [
        np.ravel_multi_index((coord[:, 0], coord[:, 1]), frame.shape)
        for coord in coords
    ]
    return inds
