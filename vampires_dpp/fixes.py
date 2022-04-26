from astropy.io import fits
from astropy.time import Time
from typing import Optional


def fix_header(filename, output: Optional[str] = None):
    path = filename
    if output is None:
        output = path.with_name(f"{path.stem}_fix{path.suffix}")

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
    # repeat again for MJD with different format
    t_str = Time(hdr["MJD-STR"], format="mjd", scale="ut1")
    t_end = Time(hdr["MJD-END"], format="mjd", scale="ut1")
    # get typical time as midpoint of two times
    dt = (t_end - t_str) / 2
    t_typ = t_str + dt
    # split on the space to remove date from timestamp
    return t_typ.mjd
