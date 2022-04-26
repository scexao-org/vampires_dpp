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
        date = hdr["DATE-OBS"]
        for (key, fmt) in zip(("UT", "HST", "MJD"), ("fits", "fits", "mjd")):
            key_end = f"{key}-END"
            t_end = Time(f"{date}T{hdr[key_end]}", format=fmt, scale="ut1")
            t_typ = Time(f"{date}T{hdr[key]}", format=fmt, scale="ut1")
            if t_typ > t_end:
                key_str = f"{key}-STR"
                t_str = Time(f"{date}T{hdr[key_str]}", format=fmt, scale="ut1")
                dt = (t_end - t_str) / 2
                t_typ = t_str + dt
                hdr[key] = t_typ.iso.split()[
                    -1
                ]  # split on the space to remove date from timestamp

        # save file
        fits.writeto(output, data, header=hdr, overwrite=True)

    return output
