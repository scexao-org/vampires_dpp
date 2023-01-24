import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike
from packaging import version
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


def check_version(config: str, vpp: str) -> bool:
    """
    Checks compatibility between versions following semantic versioning.

    Parameters
    ----------
    config : str
        Version string for the configuration
    vpp : str
        Version string for `vampires_dpp`

    Returns
    -------
    bool
    """
    config_maj, config_min, config_pat = version.parse(config).release
    vpp_maj, vpp_min, vpp_pat = version.parse(vpp).release
    if vpp_maj == 0:
        flag = config_maj == vpp_maj and config_min == vpp_min and vpp_pat >= config_pat
    else:
        flag = config_maj == vpp_maj and vpp_min >= config_min
        if vpp_min == config_min:
            flag = flag and vpp_pat >= config_pat
    return flag
