# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
from astropy.io import fits
import numpy as np
from pathlib import Path
from typing import Optional, Union
from numpy.typing import ArrayLike


def calibrate(
    data: ArrayLike,
    discard: int = 0,
    dark: Optional[ArrayLike] = None,
    flat: Optional[ArrayLike] = None,
    flip: bool = False,
):
    """
    Basic frame calibration.

    Will optionally do dark subtraction, flat correction, discard leading frames, and flip the axes for mirrored data.

    Parameters
    ----------
    data : ArrayLike
        3-D cube (t, y, x) of data
    discard : int, optional
        The amount of leading frames to discard (for data which has destructive readout frames), by default 0
    dark : ArrayLike, optional
        If provided, will dark-subtract all frames by the provided 2-D master dark (y, x), by default None
    flat : ArrayLike, optional
        If provided, will flat-correct (after dark-subtraction) all frames by the provided 2-D master flat (y, x), by default None
    flip : bool, optional
        If True, will flip the x-axis of the data, for de-mirroring cam2 data, by default False

    Returns
    -------
    ArrayLike
        3-D calibrated data cube (t, y, x)
    """
    # discard frames
    out = data.copy()[discard:]
    if dark is not None:
        out = out - dark
    if flat is not None:
        out = out / flat
    if flip:
        out = np.flip(out, axis=1)
    return out


def calibrate_file(
    filename: str,
    suffix="_calib",
    hdu: int = 0,
    dark: Optional[str] = None,
    flat: Optional[str] = None,
    flip: Union[str, bool] = "auto",
    **kwargs,
):
    path = Path(filename)
    outname = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    data, hdr = fits.getdata(path, hdu, header=True)
    if dark is not None:
        hdr["VPP_DARK"] = (dark.name, "file used for dark subtraction")
        dark = fits.getdata(dark)
    if flat is not None:
        hdr["VPP_FLAT"] = (flat.name, "file used for flat-fielding")
        flat = fits.getdata(flat)
    if flip == "auto":
        if "U_CAMERA" in hdr:
            flip = hdr["U_CAMERA"] == 2
        else:
            flip = "cam2" in path.stem
    processed = calibrate(data, dark=dark, flat=flat, flip=flip, **kwargs)
    fits.writeto(outname, processed, hdr, overwrite=True)


def deinterleave(data: ArrayLike):
    """
    Deinterleave data into two seperate FLC states

    Parameters
    ----------
    data : ArrayLike
        3-D data cube (t, y, x) from a single camera

    Returns
    -------
    (state1, state2) : Tuple[ArrayLike, ArrayLike]
        two 3-D data cubes (t, y, x), one for every other frame from the original cube
    """
    set1 = data[::2]
    set2 = data[1::2]
    return set1, set2


def deinterleave_file(filename: str, hdu: int = 0, **kwargs):
    path = Path(filename)
    data, hdr = fits.getdata(path, hdu, header=True)
    set1, set2 = deinterleave(data, **kwargs)
    hdr1 = hdr.copy()
    hdr1["U_FLCSTT"] = (1, "FLC state (1 or 2)")
    hdr2 = hdr.copy()
    hdr2["U_FLCSTT"] = (2, "FLC state (1 or 2)")
    fits.writeto(
        path.with_name(f"{path.stem}_FLC1{path.suffix}"), set1, hdr1, overwrite=True
    )
    fits.writeto(
        path.with_name(f"{path.stem}_FLC2{path.suffix}"), set2, hdr2, overwrite=True
    )


def make_dark_file(filename: str, output: Optional[str] = None, discard: int = 1):
    _path = Path(filename)
    cube, header = fits.getdata(_path, header=True)
    master_dark = np.median(cube[discard:], axis=0)
    if output is not None:
        outname = output
    else:
        outname = _path.with_name(f"{_path.stem}_master_dark{_path.suffix}")
    fits.writeto(outname, master_dark, header=header, overwrite=True)


def make_flat_file(
    filename: str,
    dark: Optional[str] = None,
    output: Optional[str] = None,
    discard: int = 1,
):
    _path = Path(filename)
    cube, header = fits.getdata(_path, header=True)
    cube = cube[discard:]
    if dark is not None:
        header["VPP_DARK"] = (dark.name, "file used for dark subtraction")
        master_dark = fits.getdata(dark)
        cube = cube - master_dark
    master_flat = np.median(cube, axis=0)
    master_flat /= np.median(master_flat)
    if output is not None:
        outname = output
    else:
        outname = _path.with_name(f"{_path.stem}_master_flat{_path.suffix}")
    fits.writeto(outname, master_flat, header=header, overwrite=True)
