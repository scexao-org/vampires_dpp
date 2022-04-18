from itertools import product
import numpy as np
from numpy.typing import ArrayLike

from .image_processing import frame_center


def window_centers(center, radius, theta=0, n=4):
    # get the angles for each branch
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.deg2rad(theta)
    xs = radius * np.cos(theta) + center[1]
    ys = radius * np.sin(theta) + center[0]
    return list(zip(ys, xs))


def window_slice(frame, center, window):
    half_width = np.asarray(window) / 2
    Ny, Nx = frame.shape[-2:]
    lower = np.maximum(0, np.round(center - half_width), dtype=int, casting="unsafe")
    upper = np.minimum(
        (Ny - 1, Nx - 1), np.round(center + half_width), dtype=int, casting="unsafe"
    )
    return range(lower[0], upper[0] + 1), range(lower[1], upper[1] + 1)


def cart_coords(ys, xs):
    Xg, Yg = np.meshgrid(ys, xs)
    return np.column_stack((Yg.ravel(), Xg.ravel()))


def window_indices(frame, window=30, center=None, **kwargs):
    if center is None:
        center = frame_center(frame)
    centers = window_centers(center, **kwargs)
    slices = [window_slice(frame, center=cent, window=window) for cent in centers]
    coords = [cart_coords(sl[0], sl[1]) for sl in slices]
    inds = [
        np.ravel_multi_index((coord[:, 0], coord[:, 1]), frame.shape)
        for coord in coords
    ]
    return inds


def window_masks(frame, **kwargs):
    inds = window_indices(frame, **kwargs)
    out = np.zeros(frame.size, dtype=bool)
    for ind in inds:
        out[ind] = True
    return np.reshape(out, frame.shape)
