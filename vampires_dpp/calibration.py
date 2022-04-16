# library functions for common calibration tasks like
# dark subtraction, collapsing cubes

import numpy as np

def calibrate(data, discard=0, dark=None, flat=None, flip=False):
    """
    Basic frame calibration.

    Will optionally do dark subtraction, flat correction, discard leading frames, and flip the axes for mirrored data.

    Parameters
    ----------
    data : _type_
        _description_
    discard : int, optional
        _description_, by default 0
    dark : _type_, optional
        _description_, by default None
    flat : _type_, optional
        _description_, by default None
    flip : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
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

def deinterleave(data):
    """
    Deinterleave data into two seperate FLC states

    Parameters
    ----------
    data : ndarray
        3-D data cube (t, y, x) from a single camera

    Returns
    -------
    (state1, state2))
        two 3-D data cubes (t, y, x), one for every other frame from the original cube
    """
    set1 = data[::2]    
    set2 = data[1::2]
    return set1, set2
