import numpy as np
from skimage.transform import rotate
from scipy.ndimage import fourier_shift
from numpy.typing import ArrayLike
from typing import Union
import cv2


def shift_frame_fft(data: ArrayLike, shift):
    data_freq = np.fft.fft2(data)
    filt = fourier_shift(data_freq, shift)
    shifted = np.real(np.fft.ifft2(filt))
    return shifted


def shift_frame(data: ArrayLike, shift, **kwargs):
    M = np.float32(((1, 0, shift[1]), (0, 1, shift[0])))
    shape = (data.shape[1], data.shape[0])
    default_kwargs = {
        "flags": cv2.INTER_CUBIC,
        "borderMode": cv2.BORDER_REFLECT,
    }
    default_kwargs.update(**kwargs)
    return cv2.warpAffine(data.astype("f4"), M, shape, **default_kwargs)


def derotate_frame(data: ArrayLike, angle, center=None, **kwargs):
    """_summary_

    Parameters
    ----------
    data : ArrayLike
        _description_
    angle : _type_
        ANGLE CONVENTION IS CW, OPPOSITE OF ASTROMETRIC
    center : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    shape = (data.shape[1], data.shape[0])
    default_kwargs = {
        "flags": cv2.INTER_CUBIC,
        "borderMode": cv2.BORDER_REFLECT,
    }
    default_kwargs.update(**kwargs)
    return cv2.warpAffine(data.astype("f4"), M, shape, **default_kwargs)


def warp_frame(data: ArrayLike, shift=0, angle=0, center=None, **kwargs):
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    M[::-1, 2] += shift
    shape = (data.shape[1], data.shape[0])
    default_kwargs = {
        "flags": cv2.INTER_CUBIC,
        "borderMode": cv2.BORDER_REFLECT,
    }
    default_kwargs.update(**kwargs)
    return cv2.warpAffine(data.astype("f4"), M, shape, **default_kwargs)


def derotate_cube(data: ArrayLike, angles: Union[ArrayLike, float], **kwargs):
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


def shift_cube(cube: ArrayLike, shifts: ArrayLike, **kwargs):
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = shift_frame(cube[i], shifts[i], **kwargs)
    return out


def weighted_collapse(
    data: ArrayLike, angles: ArrayLike, fill_value: float = 0, **kwargs
):
    variance_frame = np.var(data, axis=0, keepdims=True)
    # if the variance is zero, return the mean
    if np.allclose(variance_frame, 0):
        derotated = derotate_cube(data, angles, **kwargs)
        return np.mean(derotated, 0)

    # expand the variance frame into a cube
    variance_cube = np.repeat(variance_frame, data.shape[0], axis=0)
    # derotate both signal and variance
    derotated_data = derotate_cube(data, angles, **kwargs)
    derotated_variance = derotate_cube(variance_cube, angles, **kwargs)
    # calculate weighted sum
    numer = np.sum(derotated_data / derotated_variance, 0)
    denom = np.sum(1 / derotated_variance, 0)
    weighted_frame = numer / denom
    return weighted_frame


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
