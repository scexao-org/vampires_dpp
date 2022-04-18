from multiprocessing.dummy import Array
import numpy as np
from skimage.transform import AffineTransform, warp
from scipy.ndimage import fourier_shift
from numpy.typing import ArrayLike
from typing import Tuple


def shift_frame(data: ArrayLike, shift):
    data_freq = np.fft.fft2(data)
    filt = fourier_shift(data_freq, shift)
    shifted = np.real(np.fft.ifft2(filt))
    return shifted


def rotate_frame(data: ArrayLike, angle):
    tform = None
    rotated = warp(data, tform)
    return rotated
