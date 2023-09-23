import warnings
from pathlib import Path

import numpy as np
import tqdm.auto as tqdm
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.analysis import safe_aperture_sum
from vampires_dpp.image_processing import combine_frames_headers, derotate_frame
from vampires_dpp.image_registration import offset_centroid
from vampires_dpp.indexing import cutout_inds, frame_angles
from vampires_dpp.util import any_file_newer, average_angle
from vampires_dpp.wcs import apply_wcs

from .mueller_matrices import mueller_matrix_from_header

HWP_POS_STOKES = {0: "Q", 45: "-Q", 22.5: "U", 67.5: "-U"}


def measure_instpol(I: ArrayLike, X: ArrayLike, r=5, expected=0, window=30, **kwargs):
    """
    Use aperture photometry to estimate the instrument polarization.

    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube (4, y, x)
    r : float, optional
        Radius of circular aperture in pixels, by default 5
    expected : float, optional
        The expected fractional polarization, by default 0
    **kwargs

    Returns
    -------
    float
        The instrumental polarization coefficient
    """
    x = X / I
    inds = cutout_inds(x, window=window, **kwargs)
    cutout = x[inds]
    pX = safe_aperture_sum(cutout, r=r) / (np.pi * r**2)
    return pX - expected


def instpol_correct(stokes_cube: ArrayLike, pQ=0, pU=0):
    """
    Apply instrument polarization correction to stokes cube.

    Parameters
    ----------
    stokes_cube : ArrayLike
        (3, ...) array of stokes values
    pQ : float, optional
        I -> Q contribution, by default 0
    pU : float, optional
        I -> U contribution, by default 0

    Returns
    -------
    NDArray
        (3, ...) stokes cube with corrected parameters
    """
    return np.array(
        (
            stokes_cube[0],
            stokes_cube[1] - pQ * stokes_cube[0],
            stokes_cube[2] - pU * stokes_cube[0],
        )
    )


def radial_stokes(stokes_cube: ArrayLike, phi: float = 0) -> NDArray:
    r"""
    Calculate the radial Stokes parameters from the given Stokes cube (4, N, M)

    .. math::
        Q_\phi = -Q\cos(2\theta) - U\sin(2\theta) \\
        U_\phi = Q\sin(2\theta) - Q\cos(2\theta)


    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube, with dimensions (4, N, M)
    phi : float
        Radial angle offset in radians, by default 0

    Returns
    -------
    NDArray, NDArray
        Returns the tuple (Qphi, Uphi)
    """
    thetas = frame_angles(stokes_cube, conv="astro")

    cos2t = np.cos(2 * (thetas + phi))
    sin2t = np.sin(2 * (thetas + phi))
    Qphi = -stokes_cube[1] * cos2t - stokes_cube[2] * sin2t
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t

    return Qphi, Uphi


def rotate_stokes(stokes_cube, theta):
    out = stokes_cube.copy()
    sin2ts = np.sin(2 * theta)
    cos2ts = np.cos(2 * theta)
    out[1] = stokes_cube[1] * cos2ts - stokes_cube[2] * sin2ts
    out[2] = stokes_cube[1] * sin2ts + stokes_cube[2] * cos2ts
    return out


def collapse_stokes_cube(stokes_cube, header=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stokes_out = np.nanmedian(stokes_cube, axis=1, overwrite_input=True)
    # now that cube is derotated we can apply WCS
    if header is not None:
        apply_wcs(header)

    return stokes_out, header


def triplediff_average_angles(filenames):
    if len(filenames) % 16 != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )
    # make sure we get data in correct order using FITS headers
    derot_angles = np.asarray([fits.getval(f, "DEROTANG") for f in filenames])
    N_hwp_sets = len(filenames) // 16
    pas = np.zeros(N_hwp_sets, dtype="f4")
    for i in range(pas.shape[0]):
        ix = i * 16
        pas[i] = average_angle(derot_angles[ix : ix + 16])

    return pas


def write_stokes_products(stokes_cube, header=None, outname=None, force=False, phi=0):
    if outname is None:
        path = Path("stokes_cube.fits")
    else:
        path = Path(outname)

    if not force and path.is_file():
        return path

    pi = np.hypot(stokes_cube[2], stokes_cube[1])
    aolp = np.arctan2(stokes_cube[2], stokes_cube[1])
    Qphi, Uphi = radial_stokes(stokes_cube, phi=phi)

    if header is None:
        header = fits.Header()

    header["STOKES"] = "I,Q,U,Qphi,Uphi,LP_I,AoLP"
    if phi is not None:
        header["DPP_PHI"] = phi, "deg, angle of linear polarization offset"

    data = np.asarray((stokes_cube[0], stokes_cube[1], stokes_cube[2], Qphi, Uphi, pi, aolp))

    fits.writeto(path, data, header=header, overwrite=True)

    return path
