import numpy as np
from skimage.transform import rotate
from scipy.ndimage import fourier_shift
from numpy.typing import ArrayLike


def shift_frame(data: ArrayLike, shift):
    data_freq = np.fft.fft2(data)
    filt = fourier_shift(data_freq, shift)
    shifted = np.real(np.fft.ifft2(filt))
    return shifted


def derotate_frame(data: ArrayLike, angle, center=None, **kwargs):
    # reverse user-given center because scikit-image
    # uses swapped axes for this parameter only
    if center is not None:
        center = center[::-1]
    rotate_kwargs = {"center": center, "mode": "reflect", "order": 3, **kwargs}
    rotated = rotate(data, angle, **rotate_kwargs)
    return rotated


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
