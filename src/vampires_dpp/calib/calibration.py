# library functions for common calibration tasks like
# background subtraction, collapsing cubes
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from numpy.typing import NDArray

from vampires_dpp.constants import SUBARU_LOC
from vampires_dpp.headers import fix_header, parallactic_angle
from vampires_dpp.image_processing import adaptive_sigma_clip_mask
from vampires_dpp.paths import get_paths
from vampires_dpp.util import load_fits, wrap_angle
from vampires_dpp.wcs import apply_wcs, get_coord_header

__all__ = ("apply_coordinate", "calibrate_file")


def apply_coordinate(image: NDArray, header, coord: SkyCoord | None = None):
    time_str = Time(header["MJD-STR"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
    time_end = Time(header["MJD-END"], format="mjd", scale="ut1", location=SUBARU_LOC)
    coord_now = get_coord_header(header, time) if coord is None else coord.apply_space_motion(time)
    for _time, _key in zip((time_str, time_end), ("STR", "END"), strict=True):
        if coord is None:
            _coord = get_coord_header(header, _time)
        else:
            _coord = coord.apply_space_motion(_time)
        pa = parallactic_angle(_time, _coord)
        header[f"PA-{_key}"] = pa, "[deg] parallactic angle of target"

    header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":"), header.comments["RA"]
    header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":"), header.comments["DEC"]
    pa = parallactic_angle(time, coord_now)
    header["PA"] = pa, "[deg] parallactic angle of target"
    derotang = wrap_angle(pa + header["PAOFFSET"])
    header["DEROTANG"] = derotang, "[deg] derotation angle for North up"
    return apply_wcs(image, header, angle=derotang)


def calibrate_file(
    filename: str,
    back_filename: str | None = None,
    flat_filename: str | None = None,
    transform_filename: str | None = None,
    force: bool = False,
    bpmask: bool = False,
    coord: SkyCoord | None = None,
    **kwargs,
) -> fits.HDUList:
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return fits.open(outpath)

    # load data and mask saturated pixels
    raw_cube, header = load_fits(path, header=True)
    header = fix_header(header)
    # mask values above saturation
    satlevel = header["FULLWELL"] / header["GAIN"]
    cube = np.where(raw_cube >= satlevel, np.nan, raw_cube.astype("f4")).byteswap().newbyteorder()
    # apply proper motion correction to coordinate
    header = apply_coordinate(cube, header, coord)
    cube_err = np.zeros_like(cube)
    # background subtraction
    if back_filename is not None:
        back_path = Path(back_filename)
        header["BACKFILE"] = back_path.name
        with fits.open(back_path) as hdul:
            assert hdul[0].header["U_CAMERA"] == header["U_CAMERA"]
            background = hdul[0].data.astype("f4")
            back_hdr = hdul[0].header
            back_err = hdul["ERR"].data.astype("f4")
            header["NOISEADU"] = back_hdr["NOISEADU"], back_hdr.comments["NOISEADU"]
            header["NOISE"] = back_hdr["NOISE"], back_hdr.comments["NOISE"]
        cube -= background
    else:
        back_err = 0
    cube_err = np.sqrt(np.maximum(cube / header["EFFGAIN"], 0) * header["ENF"] ** 2 + back_err**2)
    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        header["FLATFILE"] = flat_path.name
        with fits.open(flat_path) as hdul:
            assert hdul[0].header["U_CAMERA"] == header["U_CAMERA"]
            flat = hdul[0].data.astype("f4")
            flat_hdr = hdul[0].header
            flat[flat == 0] = np.nan
            flat_err = hdul["ERR"].data.astype("f4")
            if "NORMVAL" in flat_hdr:
                header["NORMVAL"] = flat_hdr["NORMVAL"], flat_hdr.comments["NORMVAL"]

        unnorm_cube = cube.copy()
        unnorm_cube[unnorm_cube == 0] = np.nan
        rel_err = cube_err / unnorm_cube
        rel_flat_err = flat_err / flat

        cube /= flat
        cube_err = np.abs(cube) * np.hypot(rel_err, rel_flat_err)
    # bad pixel correction
    if bpmask:
        mask = adaptive_sigma_clip_mask(cube)
        cube[mask] = np.nan
        cube_err[mask] = np.nan

    # flip cam 1 data on y-axis
    if header["U_CAMERA"] == 1:
        cube = np.flip(cube, axis=-2)
        cube_err = np.flip(cube_err, axis=-2)

    # clip fot float32 to limit data size
    prim_hdu = fits.PrimaryHDU(cube.astype("f4"), header=header)
    err_hdu = fits.ImageHDU(cube_err.astype("f4"), header=header, name="ERR")
    snr_hdu = fits.ImageHDU(prim_hdu.data / err_hdu.data, header=header, name="SNR")
    return fits.HDUList([prim_hdu, err_hdu, snr_hdu])
