import re
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from packaging import version
from scipy.stats import circmean


def wrap_angle(angle: float) -> float:
    """
    Wraps an angle into the range [-180, 180]

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


def check_version(config: str, dpp: str) -> bool:
    """
    Checks compatibility between versions following semantic versioning.

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


def get_paths(
    filename, /, suffix=None, outname=None, output_directory=None, filetype=".fits", **kwargs
):
    path = Path(filename)
    _suffix = "" if suffix is None else f"_{suffix}"
    if output_directory is None:
        output_directory = path.parent
    else:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
    if outname is None:
        outname = re.sub("\.fits(\..*)?", f"{_suffix}{filetype}", path.name)
    outpath = output_directory / outname
    return path, outpath


def any_file_newer(filenames, outpath):
    out_mt = Path(outpath).stat().st_mtime
    gen = (Path(f).stat().st_mtime > out_mt for f in filenames)
    return any(gen)
