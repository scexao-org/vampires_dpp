import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray
from photutils.aperture import CircularAnnulus, CircularAperture

from vampires_dpp.combine_frames import combine_frames_headers
from vampires_dpp.headers import sort_header
from vampires_dpp.indexing import frame_angles, frame_center
from vampires_dpp.wcs import apply_wcs


def measure_instpol(I: NDArray, X: NDArray, r=5, expected=0) -> float:  # noqa: E741
    """Use aperture photometry to estimate the instrument polarization.

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
    cy, cx = frame_center(I)
    aper = CircularAperture((cx, cy), r)
    aper_mask = aper.to_mask()
    # take sum of each and reduce after (less noise than taking sum of divided image)
    pX = np.nanmedian(aper_mask.get_values(X))
    pI = np.nanmedian(aper_mask.get_values(I))
    return pX / pI - expected


def measure_instpol_ann(I: NDArray, X: NDArray, Rin, Rout, expected=0) -> float:  # noqa: E741
    # take sum of each and reduce after (less noise than taking sum of divided image)
    cy, cx = frame_center(I)
    aper = CircularAnnulus((cx, cy), Rin, Rout)
    aper_mask = aper.to_mask()
    # take sum of each and reduce after (less noise than taking sum of divided image)
    pX = np.nanmedian(aper_mask.get_values(X))
    pI = np.nanmedian(aper_mask.get_values(I))
    return pX / pI - expected


def radial_stokes(stokes_cube: ArrayLike, stokes_err: ArrayLike | None = None, phi: float = 0):
    r"""Calculate the radial Stokes parameters from the given Stokes cube (4, N, M)

    .. math::
        Q_\phi = -Q\cos(2\theta) - U\sin(2\theta) \\
        U_\phi = Q\sin(2\theta) - U\cos(2\theta)


    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube, with dimensions (4, N, M)
    phi : float
        Radial angle offset in radians, by default 0

    Returns
    -------
    NDArray, NDArray
        Returns the tuple (Qphi, Uphi, Qphi_err, Uphi_err)
    """
    thetas = frame_angles(stokes_cube, conv="astro")

    cos2t = np.cos(2 * (thetas + phi))
    sin2t = np.sin(2 * (thetas + phi))
    Qphi = -cos2t * stokes_cube[2] - sin2t * stokes_cube[3]
    Uphi = sin2t * stokes_cube[2] - cos2t * stokes_cube[3]

    if stokes_err is not None:
        Qphi_err = np.hypot(cos2t * stokes_err[2], sin2t * stokes_err[3])
        Uphi_err = np.hypot(sin2t * stokes_err[2], cos2t * stokes_err[3])
    else:
        Qphi_err = Uphi_err = np.full_like(Qphi, np.nan)

    return Qphi, Uphi, Qphi_err, Uphi_err


def rotate_stokes(stokes_cube, theta):
    out = stokes_cube.copy()
    theta_rad = np.deg2rad(theta)
    sin2ts = np.sin(2 * theta_rad)
    cos2ts = np.cos(2 * theta_rad)
    out[2] = stokes_cube[2] * cos2ts - stokes_cube[3] * sin2ts
    out[3] = stokes_cube[2] * sin2ts + stokes_cube[3] * cos2ts
    return out


def write_stokes_products(hdul, outname=None, force=False, phi=0, planetary=False):
    path = Path("stokes_cube.fits") if outname is None else Path(outname)

    if not force and path.is_file():
        return path

    output_data = []
    output_err = []
    output_hdrs = []
    stokes_data = hdul[0].data
    stokes_err = hdul["ERR"].data
    for i in range(stokes_data.shape[0]):
        data, err = stokes_products(stokes_data[i], stokes_err[i], phi=phi, planetary=planetary)

        hdr = hdul[2 + i].header
        hdr["CTYPE3"] = "STOKES"
        if planetary:
            stokes_keys = "I_Q", "I_U", "Q", "U", "Q_R", "U_R", "LP_I", "AOLP"
        else:
            stokes_keys = "I_Q", "I_U", "Q", "U", "Q_PHI", "U_PHI", "LP_I", "AOLP"

        hdr["STOKES"] = ",".join(stokes_keys), "Stokes axis data type"
        if phi != 0:
            hdr["AOLPPHI"] = phi, "[deg] offset angle for Qphi and Uphi"

        output_data.append(data)
        output_err.append(err)
        output_hdrs.append(hdr)

    prim_hdr = apply_wcs(stokes_data, combine_frames_headers(output_hdrs), angle=0)
    prim_hdr["NCOADD"] /= len(output_hdrs)
    prim_hdr["TINT"] /= len(output_hdrs)
    prim_hdr["CTYPE3"] = "STOKES"
    prim_hdr = sort_header(prim_hdr)
    prim_hdu = fits.PrimaryHDU(np.squeeze(output_data), header=prim_hdr)
    err_hdu = fits.ImageHDU(np.squeeze(output_err), header=prim_hdr, name="ERR")
    hdul_out = fits.HDUList([prim_hdu, err_hdu])
    hdul_out.extend([fits.ImageHDU(header=sort_header(hdr)) for hdr in output_hdrs])
    hdul_out.writeto(path, overwrite=True)

    return path


def stokes_products(stokes_frame, stokes_err, phi=0, planetary: bool = False):
    pi = np.hypot(stokes_frame[3], stokes_frame[2])
    aolp = 0.5 * np.arctan2(stokes_frame[3], stokes_frame[2])
    if planetary:
        Qphi, Uphi, Qphi_err, Uphi_err = radial_stokes(-stokes_frame, stokes_err, phi=phi)
    else:
        Qphi, Uphi, Qphi_err, Uphi_err = radial_stokes(stokes_frame, stokes_err, phi=phi)
    # error propagation
    # suppress all-divide error warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi_err = np.hypot(
            stokes_frame[3] * stokes_err[3], stokes_frame[2] * stokes_err[2]
        ) / np.abs(pi)
        aolp_err = (
            np.hypot(stokes_frame[2] * stokes_err[3], stokes_frame[3] * stokes_err[2]) / pi**2
        )

    data = np.asarray((*stokes_frame, Qphi, Uphi, pi, aolp))
    data_err = np.asarray((*stokes_err, Qphi_err, Uphi_err, pi_err, aolp_err))
    return data, data_err


def calculate_pol_efficiency(mmQ, mmU):
    poleff_Q = np.hypot(mmQ[1], mmU[1])
    poleff_U = np.hypot(mmQ[2], mmU[2])
    poleff_QU = np.sqrt(0.5 * (mmQ[1] + mmQ[2]) ** 2 + 0.5 * (mmU[1] + mmU[2]) ** 2)
    average_poleff = np.mean((poleff_Q, poleff_U, poleff_QU))
    return average_poleff
