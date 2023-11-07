import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.analysis import safe_annulus_sum, safe_aperture_sum
from vampires_dpp.indexing import cutout_inds, frame_angles
from vampires_dpp.wcs import apply_wcs

from ..image_processing import combine_frames_headers
from .mueller_matrices import mueller_matrix_from_header

HWP_POS_STOKES = {0: "Q", 45: "-Q", 22.5: "U", 67.5: "-U"}


def measure_instpol(I: NDArray, X: NDArray, r=5, expected=0):
    """
    Use aperture photometry to estimate the instrument polarization.

    Parameters
    ----------
    stokes_cube : NDArray
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
    pX, _ = safe_aperture_sum(x, r=r)
    return pX / (np.pi * r**2) - expected


def measure_instpol_ann(I: NDArray, X: NDArray, Rin, Rout, expected=0):
    x = X / I
    pX, _ = safe_annulus_sum(x, Rin, Rout)
    return pX / (np.pi * (Rout**2 - Rin**2)) - expected


def instpol_correct(stokes_cube: NDArray, pQ=0, pU=0):
    """
    Apply instrument polarization correction to stokes cube.

    Parameters
    ----------
    stokes_cube : NDArray
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


def radial_stokes(stokes_cube: ArrayLike, stokes_err: Optional[ArrayLike] = None, phi: float = 0):
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
    Qphi = -cos2t * stokes_cube[1] - sin2t * stokes_cube[2]
    Uphi = sin2t * stokes_cube[1] - cos2t * stokes_cube[2]

    if stokes_err is not None:
        Qphi_err = np.hypot(cos2t * stokes_err[1], sin2t * stokes_err[2])
        Uphi_err = np.hypot(sin2t * stokes_err[1], cos2t * stokes_err[2])
    else:
        Qphi_err = Uphi_err = np.full_like(Qphi, np.nan)

    return Qphi, Uphi, Qphi_err, Uphi_err


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


def write_stokes_products(hdul, outname=None, force=False, phi=0):
    if outname is None:
        path = Path("stokes_cube.fits")
    else:
        path = Path(outname)

    if not force and path.is_file():
        return path

    nfields = hdul[0].shape[0]
    hdus = []
    for i in range(1, nfields + 1):
        data = hdul[i].data
        hdr = hdul[i].header
        err = hdul[f"{hdr['FIELD']}ERROR"].data

        pi = np.hypot(data[2], data[1])
        aolp = np.arctan2(data[2], data[1])
        Qphi, Uphi, Qphi_err, Uphi_err = radial_stokes(data, err, phi=phi)
        # error propagation
        pi_err = np.hypot(data[2] * err[2], data[1] * err[1]) / np.abs(pi)
        aolp_err = np.hypot(data[1] * err[2], data[2] * err[1]) / pi**2

        data = np.asarray((data[0], data[1], data[2], Qphi, Uphi, pi, aolp))
        data_err = np.asarray((err[0], err[1], err[2], Qphi_err, Uphi_err, pi_err, aolp_err))

        hdr["STOKES"] = "I,Q,U,Qphi,Uphi,LP_I,AoLP", "Stokes axis data type"
        if phi is not None:
            hdr["AOLPPHI"] = phi, "[deg] offset angle for Qphi and Uphi"

        hdu = fits.ImageHDU(data, hdr, name=hdr["FIELD"])
        hdu_err = fits.ImageHDU(data_err, hdr, name=f"{hdr['FIELD']}ERROR")
        hdus.append((hdu, hdu_err))

    prim_data = [hdu[0].data for hdu in hdus]
    prim_hdr = combine_frames_headers([hdu[0].header for hdu in hdus], wcs=True)
    prim_hdu = fits.PrimaryHDU(np.squeeze(np.array(prim_data)), header=prim_hdr)
    hdul_out = fits.HDUList(prim_hdu)
    # add data hdus
    for hdu in hdus:
        hdul_out.append(hdu[0])
    # add err hdus
    for hdu in hdus:
        hdul_out.append(hdu[1])

    hdul_out.writeto(path, overwrite=True)

    return path
