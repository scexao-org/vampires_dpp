from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time


def parallactic_angle(header):
    if "D_IMRPAD" in header:
        return header["D_IMRPAD"] + header["LONPOLE"] - header["D_IMRPAP"]
    else:
        return parallactic_angle_altaz(header["ALTITUDE"], header["AZIMUTH"])


def parallactic_angle_hadec(ha, dec, lat=19.823806):
    """
    Calculate parallactic angle using the hour-angle and declination directly

    .. math::

        \\theta_\\mathrm{PA} = \\atan2{\\frac{\\sin\\theta_\\mathrm{HA}}{\\tan\\theta_\mathrm{lat}\\cos\\delta - \\sin\\delta \\cos\\theta_\\mathrm{HA}}}

    Parameters
    ----------
    ha : float
        hour-angle, in hour angles
    dec : float
        declination in degrees
    lat : float, optional
        latitude of observation in degrees, by default 19.823806

    Returns
    -------
    float
        parallactic angle, in degrees East of North
    """
    _ha = ha * np.pi / 12  # hour angle to radian
    _dec = np.deg2rad(dec)
    _lat = np.deg2rad(lat)
    sin_ha, cos_ha = np.sin(_ha), np.cos(_ha)
    sin_dec, cos_dec = np.sin(_dec), np.cos(_dec)
    pa = np.arctan2(sin_ha, np.tan(_lat) * cos_dec - sin_dec * cos_ha)
    return np.rad2deg(pa)


def parallactic_angle_altaz(alt, az, lat=19.823806):
    """
    Calculate parallactic angle using the altitude/elevation and azimuth directly

    .. math::
        \\theta_\\mathrm{PA} = \\atan2{\\frac{\\sin\\theta_\\mathrm{HA}}{\\tan\\theta_\mathrm{lat}\\cos\\delta - \\sin\\delta \\cos\\theta_\\mathrm{HA}}}

    Parameters
    ----------
    alt : float
        altitude or elevation, in degrees
    az : float
        azimuth, in degrees CCW from North
    lat : float, optional
        latitude of observation in degrees, by default 19.823806

    Returns
    -------
    float
        parallactic angle, in degrees East of North
    """
    ## Astronomical Algorithms, Jean Meeus
    # get angles, rotate az to S
    _az = np.deg2rad(az) - np.pi
    _alt = np.deg2rad(alt)
    _lat = np.deg2rad(lat)
    # calculate values ahead of time
    sin_az, cos_az = np.sin(_az), np.cos(_az)
    sin_alt, cos_alt = np.sin(_alt), np.cos(_alt)
    sin_lat, cos_lat = np.sin(_lat), np.cos(_lat)
    # get declination
    dec = np.arcsin(sin_alt * sin_lat - cos_alt * cos_lat * cos_az)
    # get hour angle
    ha = np.arctan2(sin_az, cos_az * sin_lat + np.tan(_alt) * cos_lat)
    # get parallactic angle
    pa = np.arctan2(np.sin(ha), np.tan(_lat) * np.cos(dec) - np.sin(dec) * np.cos(ha))
    return np.rad2deg(pa)


def fix_header(header):
    # check if the millisecond delimiter is a colon
    for key in ("UT-STR", "UT-END", "HST-STR", "HST-END"):
        if header[key].count(":") == 3:
            tokens = header[key].rpartition(":")
            header[key] = f"{tokens[0]}.{tokens[2]}"
    # fix UT/HST/MJD being time of file creation instead of typical time
    for key in ("UT", "HST"):
        header[key] = fix_typical_time_iso(header, key)
    header["MJD"] = fix_typical_time_mjd(header)
    # add RET-ANG1 and similar headers to be more consistent
    header["DETECTOR"] = f"iXon Ultra 897 - VAMPIRES {header['U_CAMERA']}", "Name of the detector"
    if "U_EMGAIN" in header:
        header["DETGAIN"] = header["U_EMGAIN"], "Detector multiplication factor"
    if "U_HWPANG" in header:
        header["RET-ANG1"] = header["U_HWPANG"], "Position angle of first retarder plate (deg)"
        header["RETPLAT1"] = "HWP(NIR)", "Identifier of first retarder plate"
    if "U_FLCSTT" in header:
        header["RET-ANG2"] = (
            0 if header["U_FLCSTT"] == 1 else 45,
            "Position angle of second retarder plate (deg)",
        )
        header["RETPLAT2"] = "FLC(VAMPIRES)", "Identifier of second retarder plate"
    return header


def fix_header_file(filename, output: Optional[str] = None, force: bool = False):
    """
    Apply fixes to headers based on known bugs

    Fixes
    -----
    1. "backpack" header (should be fixed as of 2020??)
        - Old data have second headers with instrument info
        - Will merge the second header into the first
    2. Too many colons in timestamps (should be fixed as of 2023/01/01)
        - Some STARS data have `UT` and `HST` timestamps like "10:03:34:342"
        - In this case, the last colon is replaced with a period, e.g. "10:03:34.342"
    3. "typical" exposure values are file creation time instead of midpoint
        - For `UT`, `HST`, and `MJD` keys will recalculate the midpoint based on the `*-STR` and `*-END` values

    Parameters
    ----------
    filename : pathlike
        Input FITS file
    output : Optional[str], optional
        Output file name, if None will append "_fix" to the input filename, by default None

    Returns
    -------
    Path
        path of the updated FITS file
    """
    path = Path(filename)
    if output is None:
        output = path.with_name(f"{path.stem}_fix{path.suffix}")
    else:
        output = Path(output)

    # skip file if output exists!
    if not force and output.exists() and path.stat().st_mtime < output.stat().st_mtime:
        return output

    data, hdr = fits.getdata(filename, header=True, output_verify="silentfix")
    hdr = fix_header(hdr)
    # save file
    fits.writeto(output, data, header=hdr, overwrite=True, checksum=True, output_verify="silentfix")
    return output


def fix_old_headers(filename, output=None, skip=False):
    path = Path(filename)
    if output is None:
        output = path.with_name(f"{path.stem}_hdr{path.suffix}")

    if skip and output.is_file() and path.stat().st_mtime < output.stat().st_mtime:
        return output

    # merge secondary headers
    with fits.open(path) as hdus:
        data = hdus[0].data
        hdr = hdus[0].header
        for i in range(1, len(hdus)):
            sec_hdr = hdus[i].header
            for k, v in sec_hdr.items():
                if k not in hdr:
                    hdr[k] = v

    ra_tokens = hdr["RA"].split(".")
    hdr["RA"] = ":".join(ra_tokens[:-1]) + f".{ra_tokens[-1]}"
    dec_tokens = hdr["DEC"].split(".")
    hdr["DEC"] = ":".join(dec_tokens[:-1]) + f".{dec_tokens[-1]}"
    hdr["U_CAMERA"] = 1 if "cam1" in filename else 2
    fits.writeto(output, data, header=hdr, overwrite=True, checksum=True, output_verify="silentfix")
    return output


def fix_typical_time_iso(hdr, key):
    """
    Return the middle point of the exposure for ISO-based timestamps

    Parameters
    ----------
    hdr : FITSHeader
    key : str
        key to fix, e.g. "UT"

    Returns
    -------
    str
        The ISO timestamp (hh:mm:ss.sss) for the middle point of the exposure
    """
    date = hdr["DATE-OBS"]
    # get start time
    key_str = f"{key}-STR"
    t_str = Time(f"{date}T{hdr[key_str]}", format="fits", scale="ut1")
    # get end time
    key_end = f"{key}-END"
    t_end = Time(f"{date}T{hdr[key_end]}", format="fits", scale="ut1")
    # get typical time as midpoint of two times
    dt = (t_end - t_str) / 2
    t_typ = t_str + dt
    # split on the space to remove date from timestamp
    return t_typ.iso.split()[-1]


def fix_typical_time_mjd(hdr):
    """
    Return the middle point of the exposure for MJD timestamps

    Parameters
    ----------
    hdr : FITSHeader

    Returns
    -------
    str
        The MJD timestamp for the middle point of the exposure
    """
    # repeat again for MJD with different format
    t_str = Time(hdr["MJD-STR"], format="mjd", scale="ut1")
    t_end = Time(hdr["MJD-END"], format="mjd", scale="ut1")
    # get typical time as midpoint of two times
    dt = (t_end - t_str) / 2
    t_typ = t_str + dt
    # split on the space to remove date from timestamp
    return t_typ.mjd
