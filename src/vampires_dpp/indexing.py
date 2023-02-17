import numpy as np
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.constants import FILTER_ANGULAR_SIZE, PIXEL_SCALE, SATSPOT_ANGLE


def frame_center(image: ArrayLike):
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
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    radii = np.hypot(Ys - center[0], Xs - center[1])
    return radii


def frame_angles(frame: ArrayLike, center=None):
    """
    Return the angles of pixels around ``center`` in the image

    Parameters
    ----------
    frame : ArrayLike
        Input frame
    center : Tuple, optional
        The center to calculate radii from. If None, will default to the frame center. By default None

    Returns
    -------
    NDArray
        Matrix with frame angles
    """
    if center is None:
        center = frame_center(frame)
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    # y flip + x flip TODO
    thetas = np.arctan2(Ys - center[0], Xs - center[1])
    return thetas


def lamd_to_pixel(ld, filter="Open", pxscale=PIXEL_SCALE):
    dist = FILTER_ANGULAR_SIZE[filter.strip().lower()]
    return ld * dist / pxscale


def window_centers(center, radius, theta=SATSPOT_ANGLE, n=4, **kwargs):
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
    upper = np.minimum((Ny - 1, Nx - 1), np.round(center + half_width), dtype=int, casting="unsafe")
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
        list of linear indices for each spot
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
        list of linear indices for each spot
    """
    slices = window_slices(frame, window=window, center=center, **kwargs)
    coords = (cart_coords(sl[0], sl[1]) for sl in slices)
    inds = [np.ravel_multi_index((coord[:, 0], coord[:, 1]), frame.shape) for coord in coords]
    return inds
