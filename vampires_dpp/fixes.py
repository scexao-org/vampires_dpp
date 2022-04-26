from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from typing import Optional


def fix_header(filename, output: Optional[str] = None, skip=False):
    """
    Apply fixes to headers based on known bugs

    Fixes
    -----
    1. Too many commas in timestamps
        - Some STARS data have `UT` and `HST` timestamps like "10:03:34:342"
        - In this case, the last colon is replaced with a comma, e.g. "10:03:34.342"
    2. "typical" exposure values are file creation time instead of midpoint
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
    if skip and output.exists():
        return output

    data, hdr = fits.getdata(path, header=True)
    # check if the millisecond delimiter is a color
    for key in ("UT-STR", "UT-END", "HST-STR", "HST-END"):
        if hdr[key].count(":") == 3:
            tokens = hdr[key].rpartition(":")
            hdr[key] = f"{tokens[0]}.{tokens[2]}"
    # fix UT/HST/MJD being time of file creation instead of typical time
    for key in ("UT", "HST"):
        hdr[key] = fix_typical_time_iso(hdr, key)
    hdr["MJD"] = fix_typical_time_mjd(hdr)

    # save file
    fits.writeto(output, data, header=hdr, overwrite=True)

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
