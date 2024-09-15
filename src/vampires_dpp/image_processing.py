import warnings

import bottleneck as bn
import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.indexing import cutout_inds, frame_center, frame_radii


def shift_frame(data: ArrayLike, shift: list | tuple, **kwargs) -> NDArray:
    """Shifts a single frame by the given offset

    Parameters
    ----------
    data : ArrayLike
        2D frame to shift
    shift : list | Tuple
        Shift (dy, dx) in pixels
    **kwargs
        Keyword arguments are passed to `warp_frame`

    Returns
    -------
    NDArray
        Shifted frame
    """
    M = np.float32(((1, 0, shift[1]), (0, 1, shift[0])))
    return warp_frame(data, M, **kwargs)


def derotate_frame(
    data: ArrayLike, angle: float, center: list | tuple | None = None, **kwargs
) -> NDArray:
    """Rotates a single frame clockwise by the given angle in degrees.

    Parameters
    ----------
    data : ArrayLike
        2D frame to derotate
    angle : float
        Angle, in degrees
    center : Optional[list | Tuple]
        Point defining the axis of rotation. If `None`, will use the frame center. Default is `None`.
    **kwargs
        Keyword arguments are passed to `warp_frame`

    Returns
    -------
    NDArray
        Derotated frame
    """
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    return warp_frame(data, M, **kwargs)


def warp_frame(data: ArrayLike, matrix, **kwargs) -> NDArray:
    """Geometric frame warping. By default will use bicubic interpolation with `NaN` padding.

    Parameters
    ----------
    data : ArrayLike
        2D image
    matrix : ArrayLike
        Geometric transformation matrix
    **kwargs
        Keyword arguments are passed to opencv. Important keywords like `borderValue`, `borderMode`, and `flags` can customize the padding and interpolation behavior of the transformation.

    Returns
    -------
    NDArray
        Warped frame
    """
    default_kwargs = {
        "flags": cv2.INTER_LANCZOS4,
        "borderMode": cv2.BORDER_CONSTANT,
        "borderValue": np.nan,
    }
    default_kwargs.update(**kwargs)
    shape = (data.shape[1], data.shape[0])
    if matrix[0, 0] < 1:
        data = cv2.GaussianBlur(data.astype("f4"), (3, 3), 1)
    return cv2.warpAffine(data.astype("f4"), matrix.astype("f4"), shape, **default_kwargs)


def derotate_cube(data: ArrayLike, angles: ArrayLike | float, **kwargs) -> NDArray:
    """Derotates a cube clockwise frame-by-frame with the corresponding derotation angle vector.

    Parameters
    ----------
    data : ArrayLike
        3D cube to derotate
    angles : ArrayLike | float
        If a vector, will derotate each frame by the corresponding angle. If a float, will derotate each frame by the same value.

    Returns
    -------
    NDArray
        Derotated cube
    """
    # reverse user-given center because scikit-image
    # uses swapped axes for this parameter only
    angles = np.asarray(angles)
    rotated = np.empty_like(data)
    # if angles is a scalar, broadcoast along frame index
    if angles.size == 1:
        angles = np.full(rotated.shape[0], angles)
    for idx in range(rotated.shape[0]):
        rotated[idx] = derotate_frame(data[idx], angles[idx], **kwargs)
    return rotated


def shift_cube(cube: ArrayLike, shifts: ArrayLike, **kwargs) -> NDArray:
    """Translate each frame in a cube.

    Parameters
    ----------
    cube : ArrayLike
        3D cube
    shifts : ArrayLike
        Array of (dy, dx) pairs, one for each frame in the input cube

    Returns
    -------
    NDArray
        Shifted cube
    """
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = shift_frame(cube[i], shifts[i], **kwargs)
    return out


def radial_profile_image(frame, fwhm=3):
    rs = frame_radii(frame)
    bins = np.arange(rs.min(), rs.max())
    output = np.zeros_like(frame)
    for r in bins:
        mask = (rs >= r - fwhm / 2) & (rs < r + fwhm / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            output[mask] = np.nanmedian(frame[mask])

    return output


def pad_cube(cube, pad_width: int, header=None, **pad_kwargs):
    new_shape = (cube.shape[0], cube.shape[1] + 2 * pad_width, cube.shape[2] + 2 * pad_width)
    output = np.empty_like(cube, shape=new_shape)

    for idx in range(cube.shape[0]):
        output[idx] = np.pad(cube[idx], pad_width, constant_values=np.nan)
    return output, header


def crop_to_nans_inds(data: NDArray) -> NDArray:
    """
    Crop numpy array to min/max indices that have finite values. In other words,
    trims the edges off where everything is NaN.
    """
    # determine first index that contains finite value
    is_finite = np.isfinite(data)
    ndim_range = range(data.ndim)
    # reduce over every axis except the image axes
    axes = tuple(set(ndim_range) - set(ndim_range[-2:]))
    finite_x = np.where(np.any(is_finite, axis=axes))[0]
    finite_y = np.where(np.any(is_finite, axis=axes))[0]

    min_x, max_x = finite_x[0], finite_x[-1]
    min_y, max_y = finite_y[0], finite_y[-1]
    cy, cx = frame_center(data)
    # don't just take min to max indices, calculate the radius
    # of each extreme to the center and keep everything centered
    radius = max(max_x - cx, cx - min_x, max_y - cy, cy - min_y)
    return cutout_inds(data, center=(cy, cx), window=int(radius * 2))


def adaptive_sigma_clip_mask(data, sigma=10, boxsize=8):
    grid = np.arange(boxsize // 2, data.shape[0], step=boxsize)
    output_mask = np.zeros_like(data, dtype=bool)
    boxsize / 2
    for yi in grid:
        for xi in grid:
            inds = cutout_inds(data, center=(yi, xi), window=boxsize)
            cutout = data[inds]
            med = np.nanmedian(cutout, keepdims=True)
            std = np.nanstd(cutout, keepdims=True)
            output_mask[inds] = np.abs(cutout - med) > sigma * std

    return output_mask


def create_footprint(cube, angles):
    mask = np.isfinite(cube)
    derot = derotate_cube(mask.astype(float), angles)
    return bn.nanmean(derot, axis=0)
