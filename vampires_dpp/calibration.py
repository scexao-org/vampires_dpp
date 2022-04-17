# library functions for common calibration tasks like
# dark subtraction, collapsing cubes
from astropy.io import fits
import numpy as np
from pathlib import Path

def calibrate(data, discard=0, dark=None, flat=None, flip=False):
    """
    Basic frame calibration.

    Will optionally do dark subtraction, flat correction, discard leading frames, and flip the axes for mirrored data.

    Parameters
    ----------
    data : ndarray
        3-D cube (t, y, x) of data
    discard : int, optional
        The amount of leading frames to discard (for data which has destructive readout frames), by default 0
    dark : ndarray, optional
        If provided, will dark-subtract all frames by the provided 2-D master dark (y, x), by default None
    flat : ndarray, optional
        If provided, will flat-correct (after dark-subtraction) all frames by the provided 2-D master flat (y, x), by default None
    flip : bool, optional
        If True, will flip the x-axis of the data, for de-mirroring cam2 data, by default False

    Returns
    -------
    ndarray
        3-D calibrated data cube (t, y, x)
    """
    # discard frames
    out = data.copy()[discard:]
    if dark is not None:
        out -= dark
    if flat is not None:
        out /= flat
    if flip:
        out = np.flip(out, axis=1)
    return out

def calibrate_file(filename : str, suffix="_calib", hdu=0, **kwargs):
    path = Path(filename)
    outname = path.with_stem(f"{path.stem}{suffix}")
    with fits.open(path) as hdus:
        data = hdus[hdu].data
        hdr = hdus[hdu].header
        processed = calibrate(data, **kwargs)
        fits.writeto(outname, processed, hdr, overwrite=True)

def deinterleave(data):
    """
    Deinterleave data into two seperate FLC states

    Parameters
    ----------
    data : ndarray
        3-D data cube (t, y, x) from a single camera

    Returns
    -------
    (state1, state2)
        two 3-D data cubes (t, y, x), one for every other frame from the original cube
    """
    set1 = data[::2]    
    set2 = data[1::2]
    return set1, set2

def deinterleave_file(filename : str, hdu=0, **kwargs):
    path = Path(filename)
    with fits.open(path) as hdus:
        data = hdus[hdu].data
        hdr = hdus[hdu].header
        set1, set2 = deinterleave(data, **kwargs)
        hdr1 = hdr.copy()
        hdr1["U_FLCSTT"] = (1, "FLC state (1 or 2)")
        hdr2 = hdr.copy()
        hdr2["U_FLCSTT"] = (2, "FLC state (1 or 2)")
        fits.writeto(path.with_stem(f"{path.stem}_FLC1"), set1, hdr1, overwrite=True)
        fits.writeto(path.with_stem(f"{path.stem}_FLC2"), set2, hdr2, overwrite=True)