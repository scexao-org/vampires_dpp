import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning

from vampires_dpp.constants import CMOSVAMPIRES, EMCCDVAMPIRES, SUBARU_LOC, InstrumentInfo
from vampires_dpp.util import hst_from_ut_time, iso_time_stats, mjd_time_stats, wrap_angle


def parallactic_angle(time, coord):
    pa = parallactic_angle_hadec(
        time.sidereal_time("apparent").hourangle - coord.ra.hourangle, coord.dec.deg
    )
    return wrap_angle(pa)


def parallactic_angle_hadec(ha, dec, lat=SUBARU_LOC.lat.deg):
    r"""Calculate parallactic angle using the hour-angle and declination directly

    .. math::

        \theta_\mathrm{PA} = \arctan2{\frac{\sin\theta_\mathrm{HA}}{\tan\theta_\mathrm{lat}\cos\delta - \sin\delta \cos\theta_\mathrm{HA}}}

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


def parallactic_angle_altaz(alt, az, lat=SUBARU_LOC.lat.deg):
    r"""Calculate parallactic angle using the altitude/elevation and azimuth directly

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
    # quickly check if we're dealing with pre-open-use data, which is unsupported
    if "ACQNCYCS" in header:
        msg = "WARNING: Unsupported VAMPIRES data type found, cannot guarantee all operations will be successful"
        warnings.warn(msg, stacklevel=2)
        return header
    # check if the millisecond delimiter is a colon
    for key in ("UT-STR", "UT-END", "HST-STR", "HST-END"):
        if key in header and header[key].count(":") == 3:
            tokens = header[key].rpartition(":")
            header[key] = f"{tokens[0]}.{tokens[2]}", header.comments[key]
    # fix UT/HST/MJD being time of file creation instead of typical time
    if "UT-STR" in header and "UT-END" in header:
        header = update_header_iso(header)
    # add RET-ANG1 and similar headers to be more consistent
    if "DETECTOR" not in header:
        header["DETECTOR"] = (f"VCAM{header['U_CAMERA']:.0f} - Ultra 897", "Name of the detector")
    if "OBS-MOD" not in header:
        header["OBS-MOD"] = "IPOL", "Observation mode"
    if "U_EMGAIN" in header:
        header["DETGAIN"] = header["U_EMGAIN"], "Detector multiplication factor"
    if "U_HWPANG" in header:
        header["RET-ANG1"] = (header["U_HWPANG"], "[deg] Position angle of first retarder plate")
        header["RETPLAT1"] = "HWP(NIR)", "Identifier of first retarder plate"
    if "U_FLCSTT" in header:
        header["U_FLC"] = "A" if header["U_FLCSTT"] == 1 else "B", "VAMPIRES FLC State"
    if "EXPTIME" not in header:
        header["EXPTIME"] = (header["U_AQTINT"] / 1e6, "[s] Total integration time of the frame")
    if "FILTER01" not in header:
        header["FILTER01"] = header["U_FILTER"], "Primary filter name"
        header["FILTER02"] = "Unknown", "Secondary filter name"
    if "PRD-MIN1" not in header:
        full_size = 512
        crop_size = header["NAXIS2"], header["NAXIS1"]
        start_idx = np.round((full_size - np.array(crop_size)) / 2)
        header["PRD-MIN1"] = start_idx[-1], "[pixel] Origin in X of the cropped window"
        header["PRD-MIN2"] = start_idx[-2], "[pixel] Origin in Y of the cropped window"
        header["PRD-RNG1"] = crop_size[-1], "[pixel] Range in X of the cropped window"
        header["PRD-RNG2"] = crop_size[-2], "[pixel] Range in Y of the cropped window"
    if "TINT" not in header:
        header["TINT"] = (
            header["EXPTIME"] * header.get("NAXIS3", 1),
            "[s] Total integration time of file",
        )
    if "NDIT" not in header:
        header["NDIT"] = header.get("NAXIS3", 1), "Number of frames in original file"
    if "BUNIT" not in header:
        header["BUNIT"] = "ADU", "Unit of original values"

    # add in detector charracteristics
    inst = get_instrument_from(header)
    header["BIAS"] = inst.bias, "[adu] Bias offset"
    header["GAIN"] = inst.gain, "[e-/adu] Detector gain"
    header["DC"] = inst.dark_current, "[e-/s/pix] Detector dark current"
    header["ENF"] = inst.excess_noise_factor, "Detector excess noise factor"
    header["EFFGAIN"] = inst.effgain, "[e-/adu] Detector effective gain"
    header["RN"] = inst.readnoise, "[e-] RMS read noise"
    header["PXSCALE"] = inst.pixel_scale, "[mas/pix] Pixel scale"
    header["PXAREA"] = (header["PXSCALE"] / 1e3) ** 2, "[arcsec^2/pix] Solid angle of each pixel"
    header["PAOFFSET"] = inst.pa_offset, "[deg] Parallactic angle offset"
    header["INST-PA"] = inst.pupil_offset, "[deg] Instrument angle offset"
    header["FULLWELL"] = inst.fullwell, "[e-] Full well of detector register"
    # during this time period the orcas had 2^15 full well from milk savefits bug
    if (
        Time("2023-06-01T00:00:00", format="fits")
        < Time(header["MJD"], format="mjd")
        < Time("2023-07-24T00:00:00", format="fits")
    ):
        header["FULLWELL"] /= 2

    return header


def update_header_iso(header: fits.Header) -> fits.Header:
    """Return the middle point of the exposure for ISO-based timestamps

    Parameters
    ----------
    header : FITSHeader
    """
    date = header["DATE-OBS"]
    # get start time
    key_str = "UT-STR"
    key_end = "UT-END"
    ut_str, ut_typ, ut_end = iso_time_stats(date, header[key_str], header[key_end])

    header["UT-STR"] = ut_str.iso.split()[-1], header.comments["UT-STR"]
    header["UT-END"] = ut_end.iso.split()[-1], header.comments["UT-END"]
    header["UT"] = ut_typ.iso.split()[-1], header.comments["UT"]
    header["DATE-OBS"] = ut_typ.iso.split()[0], header.comments["DATE-OBS"]

    hst_str = hst_from_ut_time(ut_str)
    hst_typ = hst_from_ut_time(ut_typ)
    hst_end = hst_from_ut_time(ut_end)

    header["HST-STR"] = hst_str.iso.split()[-1], header.comments["HST-STR"]
    header["HST-END"] = hst_end.iso.split()[-1], header.comments["HST-END"]
    header["HST"] = hst_typ.iso.split()[-1], header.comments["HST"]
    return header


def update_header_mjd(header: fits.Header) -> fits.Header:
    """Return the middle point of the exposure for MJD timestamps

    Parameters
    ----------
    header : FITSHeader

    Returns
    -------
    str
        The MJD timestamp for the middle point of the exposure
    """
    # repeat again for MJD with different format
    t_str, t_typ, t_end = mjd_time_stats(header["MJD-STR"], header["MJD-END"])
    header["MJD-STR"] = t_str.mjd, header.comments["MJD-STR"]
    header["MJD-END"] = t_end.mjd, header.comments["MJD-END"]
    header["MJD"] = t_typ.mjd, header.comments["MJD"]
    return header


def get_instrument_from(header: fits.Header) -> InstrumentInfo:
    """Get the instrument info from a FITS header"""
    cam_num = int(header["U_CAMERA"])
    if "U_EMGAIN" in header:
        inst = EMCCDVAMPIRES(cam_num=cam_num, emgain=header["U_EMGAIN"])
    elif "U_DETMOD" in header:
        inst = CMOSVAMPIRES(cam_num=cam_num, readmode=header["U_DETMOD"].lower())
    else:
        msg = "Could not determine VAMPIRES instrument (EMCCD vs. CMOS) from headers. Check if 'U_CAMERA' and 'U_EMGAIN' or 'U_DETMOD' keywords are present."
        raise ValueError(msg)

    return inst


def sort_header(header: fits.Header) -> fits.Header:
    """Sort all non-structural FITS header keys"""
    output_header = fits.Header()
    for key in sorted(header):
        # skip structural keys
        if key in ("SIMPLE", "BITPIX", "EXTEND", "COMMENT", "HISTORY") or key.startswith("NAXIS"):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            output_header[key] = header[key], header.comments[key]
    return output_header
