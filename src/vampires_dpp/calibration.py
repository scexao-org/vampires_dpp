# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import ArrayLike

from vampires_dpp.headers import fix_header


def calibrate(
    data: ArrayLike,
    discard: int = 0,
    dark: Optional[ArrayLike] = None,
    flat: Optional[ArrayLike] = None,
    flip: bool = False,
    header=None,
):
    """
    Basic frame calibration.

    Will optionally do dark subtraction, flat correction, discard leading frames, and flip the axes for mirrored data.

    Parameters
    ----------
    data : ArrayLike
        3-D cube (t, y, x) of data
    discard : int, optional
        The amount of leading frames to discard (for data which has destructive readout frames), by default 0.
    dark : ArrayLike, optional
        If provided, will dark-subtract all frames by the provided 2-D master dark (y, x), by default None
    flat : ArrayLike, optional
        If provided, will flat-correct (after dark-subtraction) all frames by the provided 2-D master flat (y, x), by default None
    flip : bool, optional
        If True, will flip the y-axis of the data, for de-mirroring cam1 data, by default False.

    .. note:: Image flips are always done last, so that means the dark frames and flat frames should not be flipped ahead of time!

    Returns
    -------
    ArrayLike
        3-D calibrated data cube (t, y, x)
    """
    # discard frames
    # need to copy so inplace operations don't allocate new arrays
    output = data[discard:].copy()
    if dark is not None:
        output = output - dark
    if flat is not None:
        output = output - flat
    if flip:
        output = np.flip(output, axis=-2)
    return output, header


def calibrate_file(
    filename: str,
    outname=None,
    hdu: int = 0,
    dark: Optional[str] = None,
    flat: Optional[str] = None,
    skip=False,
    deinterleave=False,
    **kwargs,
):
    path = Path(filename)
    if outname is None:
        outname = path.with_name(f"{path.stem}_calib{path.suffix}")
    else:
        outname = Path(outname)
    if skip and outname.is_file():
        return outname

    data, hdr = fits.getdata(path, hdu, header=True)
    data = data.astype("f4")
    if dark is not None:
        dark = fits.getdata(dark).astype("f4")
    if flat is not None:
        flat = fits.getdata(flat).astype("f4")
    if flip == "auto":  # flip camera 1 (vcamim1)
        if "U_CAMERA" in hdr:
            flip = hdr["U_CAMERA"] == 1
        else:
            flip = "cam1" in path.stem
    if "U_FLCSTT" in hdr:
        discard = 0
    else:
        discard = 2
    processed = calibrate(data, dark=dark, flat=flat, flip=flip, discard=discard, **kwargs)
    if deinterleave:
        set1 = processed[::2]
        hdr["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        out1 = outname
        fits.writeto(outname, processed, hdr, overwrite=True)
        set2 = processed[1::2]
        hdr["U_FLCSTT"] = 1, "FLC state (1 or 2)"
        fits.writeto(outname, processed, hdr, overwrite=True)
    else:
        fits.writeto(outname, processed, hdr, overwrite=True)
    return outname


def deinterleave_file(filename: str, hdu: int = 0, skip=False, **kwargs):
    path = Path(filename)

    outname1 = path.with_name(f"{path.stem}_FLC1{path.suffix}")
    outname2 = path.with_name(f"{path.stem}_FLC2{path.suffix}")
    if skip and outname1.is_file() and outname2.is_file():
        return outname1, outname2

    data, hdr = fits.getdata(path, hdu, header=True)
    set1, set2 = deinterleave(data, **kwargs)
    hdr1 = hdr.copy()
    hdr1["U_FLCSTT"] = (1, "FLC state (1 or 2)")
    hdr2 = hdr.copy()
    hdr2["U_FLCSTT"] = (2, "FLC state (1 or 2)")

    fits.writeto(outname1, set1, hdr1, overwrite=True)
    fits.writeto(outname2, set2, hdr2, overwrite=True)
    return outname1, outname2


def make_dark_file(filename: str, output: Optional[str] = None, discard: int = 1, skip=False):
    _path = Path(filename)
    if output is not None:
        outname = output
    else:
        outname = _path.with_name(f"{_path.stem}_master_dark{_path.suffix}")
    if skip and outname.is_file():
        return outname
    cube, header = fits.getdata(_path, header=True)
    master_dark = np.median(cube.astype("f4")[discard:], axis=0, overwrite_input=True)
    fits.writeto(outname, master_dark, header=header, overwrite=True)
    return outname


def make_flat_file(
    filename: str,
    dark: Optional[str] = None,
    output: Optional[str] = None,
    discard: int = 1,
    skip=False,
):
    _path = Path(filename)
    if output is not None:
        outname = output
    else:
        outname = _path.with_name(f"{_path.stem}_master_flat{_path.suffix}")
    if skip and outname.is_file():
        return outname
    cube, header = fits.getdata(_path, header=True)
    cube = cube.astype("f4")[discard:]
    if dark is not None:
        header["VPP_DARK"] = (dark.name, "file used for dark subtraction")
        master_dark = fits.getdata(dark).astype("f4")
        cube = cube - master_dark
    master_flat = np.nanmedian(cube, axis=0, overwrite_input=True)
    master_flat = master_flat / np.nanmedian(master_flat)

    fits.writeto(outname, master_flat, header=header, overwrite=True)
    return outname


def filter_empty_frames(cube):
    output = np.array([frame for frame in cube if np.any(frame)])
    return output
