from pathlib import Path

import numpy as np
import pytz
from astropy.io import fits
from astropy.time import Time
from numpy.typing import ArrayLike
from packaging import version
from scipy.stats import circmean


def wrap_angle(angle: float) -> float:
    """Wraps an angle into the range [-180, 180]

    Parameters
    ----------
    angle: float
        Input angle, in degrees

    Returns
    -------
    angle: float
        output angle in degrees
    """
    if angle < -180:
        angle += 360
    elif angle > 180:
        angle -= 360
    return angle


def average_angle(angles: ArrayLike):
    """Return the circular mean of the given angles in degrees.

    Parameters
    ----------
    angles : ArrayLike
        Angles in degrees, between [-180, 180]

    Returns
    -------
    average_angle
        The average angle in degrees via the circular mean
    """
    rads = np.deg2rad(angles)
    radmean = circmean(rads, high=np.pi, low=-np.pi)
    return np.rad2deg(radmean)


def delta_angle(alpha: float, beta: float) -> float:
    """Given two angles, determine the total rotation between them"""
    alpha_mod = np.mod(alpha, 360)
    beta_mod = np.mod(beta, 360)
    return beta_mod - alpha_mod


def check_version(config: str, dpp: str) -> bool:
    """Checks compatibility between versions following semantic versioning.

    Parameters
    ----------
    config : str
        Version string for the configuration
    dpp : str
        Version string for `vampires_dpp`

    Returns
    -------
    bool
    """
    config_maj, config_min, config_pat = version.parse(config).release
    dpp_maj, dpp_min, dpp_pat = version.parse(dpp).release
    if dpp_maj == 0:
        flag = config_maj == dpp_maj and config_min == dpp_min and dpp_pat >= config_pat
    else:
        flag = config_maj == dpp_maj and dpp_min >= config_min
        if dpp_min == config_min:
            flag = flag and dpp_pat >= config_pat
    return flag


def load_fits(filename, ext=0, **kwargs):
    path = Path(filename)
    if ".fits.fz" in path.name:
        ext = 1
    return fits.getdata(path, ext=ext, **kwargs)


def load_fits_header(filename, ext=0, **kwargs):
    path = Path(filename)
    if ".fits.fz" in path.name:
        ext = 1
    return fits.getheader(path, ext=ext, **kwargs)


def load_fits_key(filename, key, ext=0, **kwargs):
    path = Path(filename)
    if ".fits.fz" in path.name:
        ext = 1
    return fits.getval(path, key, ext=ext, **kwargs)


def create_or_append(dict, key, value):
    if key in dict:
        dict[key].append(value)
    else:
        dict[key] = [value]


def get_center(frame, centroid, cam_num):
    # IMPORTANT we need to flip the centroids for cam1 since they
    # are saved from raw data but we have y-flipped the data
    # during calibration

    if cam_num == 2:
        return centroid
    # for cam 1 data, need to flip coordinate about x-axis
    Ny = frame.shape[-2]
    ctr = np.asarray((Ny - 1 - centroid[0], centroid[1]))
    return ctr


def iso_time_stats(date: str, start_time: str, end_time: str) -> tuple[Time, Time, Time]:
    # get start time
    t_str = Time(f"{date}T{start_time}", format="fits", scale="utc")
    # get end time
    t_end = Time(f"{date}T{end_time}", format="fits", scale="utc")
    # get typical time as midpoint of two times
    dt = (t_end - t_str) / 2
    t_typ = t_str + dt
    # split on the space to remove date from timestamp
    return t_str, t_typ, t_end


def mjd_time_stats(start_mjd: str, end_mjd: str) -> tuple[Time, Time, Time]:
    # get start time
    t_str = Time(start_mjd, format="mjd")
    # get end time
    t_end = Time(end_mjd, format="mjd")
    # get typical time as midpoint of two times
    dt = (t_end - t_str) / 2
    t_typ = t_str + dt
    # split on the space to remove date from timestamp
    return t_str, t_typ, t_end


def hst_from_ut_time(ut_time: Time) -> Time:
    utc_tz = pytz.timezone("UTC")
    ut_datetime = ut_time.to_datetime(utc_tz)
    hst_tz = pytz.timezone("HST")
    hst_datetime = ut_datetime.astimezone(hst_tz)
    return Time(hst_datetime.isoformat()[:-6], format="fits")
