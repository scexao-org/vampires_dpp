import numpy as np
from numpy.typing import ArrayLike

from .image_processing import frame_center


def window_centers(center, radius, theta=0, n=4):
    # get the angles for each branch
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.deg2rad(theta)
    xs = radius * np.cos(theta) + center[1]
    ys = radius * np.sin(theta) + center[0]
    return list(zip(ys, xs))


def cutout_indices(frame, center, window):
    pass


def window_indices(frame, window=30, center=None, **kwargs):
    if center is None:
        center = frame_center(frame)
    centers = window_centers(center, **kwargs)
