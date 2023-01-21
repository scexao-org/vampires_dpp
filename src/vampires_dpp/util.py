from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import circmean


def average_angle(angles: ArrayLike):
    """
    Return the circular mean of the given angles in degrees.

    Parameters
    ----------
    angles : ArrayLike
        Angles in degrees, between [180, -180]

    Returns
    -------
    average_angle
        The average angle in degrees via the circular mean
    """
    rads = np.deg2rad(angles)
    radmean = circmean(rads, high=np.pi, low=-np.pi)
    return np.rad2deg(radmean)


def find_dark_settings(filelist):
    exp_set = set()
    for filename in filelist:
        with fits.open(filename) as hdus:
            hdr = hdus[0].header
            texp = hdr["EXPTIME"]  # exposure time in microseconds
            gain = hdr["U_EMGAIN"]
            exp_set.add((texp, gain))

    return exp_set
