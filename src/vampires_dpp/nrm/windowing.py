import numpy as np

from vampires_dpp.indexing import frame_radii


def super_gaussian(r, sigma, m, amp=1):
    expon = np.log(2) * (2 ** (2 * m - 1)) * (r**2 / sigma**2) ** m
    return amp * np.exp(-expon) ** 2


def window_cube(cube, size, m=3, header=None):
    radii = frame_radii(cube)
    sigma = size * 2
    # not flux preserving
    window = super_gaussian(radii, sigma=sigma, m=m)
    output = cube * window[None, ...]

    if header is not None:
        header["hieararch DPP NRM WINDOW"] = True, "Data has been windowed with super Gaussian"
        header["hieararch DPP NRM WINDOW SIZE"] = size, "[px] Gaussian window size"
        header["hieararch DPP NRM WINDOW M"] = size, "Gaussian window scale"

    return output, header
