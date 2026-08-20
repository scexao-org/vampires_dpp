# library functions for common calibration tasks like
# background subtraction, collapsing cubes
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from numpy.typing import NDArray

from vampires_dpp.constants import NBS_INSTALL_MJD, SUBARU_LOC
from vampires_dpp.headers import fix_header, parallactic_angle, sort_header
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
    force: bool = False,
    bpmask: bool = False,
    coord: SkyCoord | None = None,
    **kwargs,
) -> fits.HDUList:
    """Calibrate a raw VAMPIRES cube into a 2-HDU (data, ERR) HDUList.

    Operations are done in-place on the loaded cube wherever possible to keep peak
    RAM bounded. Original behaviour is preserved bit-exactly aside from harmless
    operation reordering inside the flat-correction block.
    """
    path, outpath = get_paths(filename, suffix="calib", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return fits.open(outpath)

    raw_cube, header = load_fits(path, header=True)
    header = fix_header(header)
    satlevel = header["FULLWELL"] / header["GAIN"]

    # Promote to native-endian float32 once (this is the only full-cube copy we make
    # of the raw data), then mask saturated pixels in-place. astype reads the
    # big-endian FITS data correctly and yields a native float32 copy; do NOT use
    # .view(newbyteorder("=")), which relabels the bytes without swapping them and
    # corrupts the values (e.g. flips large positives into large negatives).
    sat_mask = raw_cube >= satlevel
    cube = raw_cube.astype("f4")
    cube[sat_mask] = np.nan
    del raw_cube, sat_mask

    header = apply_coordinate(cube, header, coord)

    # background subtraction
    if back_filename is not None:
        back_path = Path(back_filename)
        header["BACKFILE"] = back_path.name
        with fits.open(back_path) as hdul:
            assert hdul[0].header["U_CAMERA"] == header["U_CAMERA"]
            back_hdr = hdul[0].header
            header["NOISEADU"] = back_hdr["NOISEADU"], back_hdr.comments["NOISEADU"]
            header["NOISE"] = back_hdr["NOISE"], back_hdr.comments["NOISE"]
            cube -= hdul[0].data
            back_err = hdul["ERR"].data.astype("f4")
    else:
        cube -= header["BIAS"]
        back_err = None

    # cube_err = sqrt(max(cube/EFFGAIN, 0) * ENF^2 + back_err^2)
    # build it in-place to avoid stacking several full-cube temporaries.
    cube_err = np.divide(cube, header["EFFGAIN"])
    np.maximum(cube_err, 0, out=cube_err)
    cube_err *= header["ENF"] ** 2
    if back_err is not None:
        cube_err += back_err**2
        del back_err
    np.sqrt(cube_err, out=cube_err)

    # flat correction
    if flat_filename is not None:
        flat_path = Path(flat_filename)
        header["FLATFILE"] = flat_path.name
        with fits.open(flat_path) as hdul:
            assert hdul[0].header["U_CAMERA"] == header["U_CAMERA"]
            flat_hdr = hdul[0].header
            flat = hdul[0].data.astype("f4")
            flat_err = hdul["ERR"].data.astype("f4")
            if "NORMVAL" in flat_hdr:
                header["NORMVAL"] = flat_hdr["NORMVAL"], flat_hdr.comments["NORMVAL"]
        flat[flat == 0] = np.nan

        # Propagate uncertainty without copying the cube. We need
        #     cube_err_new = |cube/flat| * hypot(cube_err/cube, flat_err/flat).
        # Mirror the original NaN behaviour: where the pre-flat cube is exactly 0,
        # the relative error is NaN (matching the old `unnorm_cube[==0] = NaN` step),
        # which then propagates through the final cube_err.
        zero_mask = cube == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            cube_err /= cube
        cube_err[zero_mask] = np.nan
        del zero_mask
        rel_flat_err = flat_err / flat
        del flat_err
        cube /= flat
        del flat
        np.hypot(cube_err, rel_flat_err, out=cube_err)
        del rel_flat_err
        cube_err *= np.abs(cube)

    # bad pixel correction
    if bpmask:
        mask = adaptive_sigma_clip_mask(cube)
        cube[mask] = np.nan
        cube_err[mask] = np.nan
        del mask

    # t_obs < 2025-10-01: flip cam 1 on y-axis; t_obs > 2025-10-01: flip cam 2 instead
    flip_idx = 1 if header["MJD"] < NBS_INSTALL_MJD else 2
    if header["U_CAMERA"] == flip_idx:
        cube = np.flip(cube, axis=-2)
        cube_err = np.flip(cube_err, axis=-2)

    # convert to e-/s (in-place)
    calib_fac = header["EFFGAIN"] / header["EXPTIME"]
    cube *= calib_fac
    cube_err *= calib_fac
    header["BUNIT"] = "e-/s"

    header = sort_header(header)
    # cube and cube_err are already native float32 — no redundant copy
    prim_hdu = fits.PrimaryHDU(cube, header=header)
    err_hdu = fits.ImageHDU(cube_err, header=header, name="ERR")
    return fits.HDUList([prim_hdu, err_hdu])
