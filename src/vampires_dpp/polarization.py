from astropy.io import fits
from scipy.optimize import minimize_scalar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from pathlib import Path
from photutils import CircularAperture, aperture_photometry

from .image_processing import frame_angles, frame_center, derotate_frame
from .satellite_spots import window_centers
from .mueller_matrices import (
    mueller_matrix_triplediff,
    mueller_matrix_model,
    mueller_matrix_calibration,
)


def instpol_correct(stokes_cube: ArrayLike, r=5, center=None) -> NDArray:
    """
    Use aperture photometry to estimate the instrument polarization and correct the Stokes parameters.

    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube (4, y, x)
    r : float, optional
        Radius of circular aperture in pixels, by default 5
    center : Tuple, optional
        Center of circular aperture (y, x). If None, will use the frame center. By default None

    Returns
    -------
    NDArray
        (4, y, x) Stokes cube with instrument polarization corrected
    """
    if center is None:
        center = frame_center(stokes_cube)
    I, Q, U, V = stokes_cube
    q = Q / I
    u = U / I

    ap = CircularAperture((center[1], center[0]), r)

    cQ = aperture_photometry(q, ap)["aperture_sum"][0] / ap.area
    cU = aperture_photometry(u, ap)["aperture_sum"][0] / ap.area

    S = np.array((I, Q - cQ * I, U - cU * I, V))
    return S


def instpol_correct_satellite_spots(
    stokes_cube: ArrayLike, r=5, center=None, **kwargs
) -> NDArray:
    """
    Use aperture photometry on satellite spots to estimate the instrument polarization and correct the Stokes parameters.

    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube (4, y, x)
    r : float, optional
        Radius of circular aperture in pixels, by default 5
    center : Tuple, optional
        Center of satellite spots (y, x). If None, will use the frame center. By default None
    radius : float
        Radius of satellite spots in pixels

    Returns
    -------
    NDArray
        (4, y, x) Stokes cube with instrument polarization corrected
    """
    if center is None:
        center = frame_center(stokes_cube)

    I, Q, U, V = stokes_cube
    q = Q / I
    u = U / I

    centers = window_centers(center, **kwargs)
    aps_centers = [(ctr[1], ctr[0]) for ctr in centers]
    aps = CircularAperture(aps_centers, r)
    cQ = np.median(aperture_photometry(q, aps)["aperture_sum"]) / aps.area
    cU = np.median(aperture_photometry(u, aps)["aperture_sum"]) / aps.area

    S = np.array((I, Q - cQ * I, U - cU * I, V))
    return S


def radial_stokes(stokes_cube: ArrayLike, phi: Optional[float] = None) -> NDArray:
    """
    Calculate the radial Stokes parameters from the given Stokes cube (4, N, M)

    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube, with dimensions (4, N, M)
    phi : float, optional
        Radial angle offset in radians. If None, will automatically optimize the angle with ``optimize_Uphi``, which minimizes the Uphi signal. By default None

    Returns
    -------
    NDArray, NDArray
        Returns the tuple (Qphi, Uphi)
    """
    thetas = frame_angles(stokes_cube)
    if phi is None:
        phi = optimize_Uphi(stokes_cube, thetas)

    cos2t = np.cos(2 * (thetas + phi))
    sin2t = np.sin(2 * (thetas + phi))
    Qphi = -stokes_cube[1] * cos2t - stokes_cube[2] * sin2t
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t

    return Qphi, Uphi


def optimize_Uphi(stokes_cube: ArrayLike, thetas: ArrayLike) -> float:
    loss = lambda X: Uphi_loss(X, stokes_cube, thetas)
    res = minimize_scalar(loss, bounds=(-np.pi / 2, np.pi / 2), method="bounded")
    return res.x


def Uphi_loss(X: float, stokes_cube: ArrayLike, thetas: ArrayLike) -> float:
    cos2t = np.cos(2 * (thetas + X))
    sin2t = np.sin(2 * (thetas + X))
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t
    l2norm = np.sum(Uphi**2)
    return l2norm


def polarization_calibration_triplediff(filenames):
    mueller_mats = []
    cube = []
    for filename in filenames:
        frame, header = fits.getdata(filename, header=True)

        rotangle = np.deg2rad(header["D_IMRPAD"] + 140.4)
        hwp_theta = np.deg2rad(header["U_HWPANG"])

        M = mueller_matrix_triplediff(
            camera=header["U_CAMERA"],
            flc_state=header["U_FLCSTT"],
            theta=rotangle,
            hwp_theta=hwp_theta,
        )
        # only keep the X -> I terms
        mueller_mats.append(M[0])
        cube.append(frame)

    mueller_mats = np.array(mueller_mats)
    cube = np.array(cube)

    return mueller_matrix_calibration(mueller_mats, cube)


def polarization_calibration_model(filenames):
    mueller_mats = []
    cube = []
    for filename in filenames:
        frame, header = fits.getdata(filename, header=True)

        pa = np.deg2rad(header["D_IMRPAD"] + 180 - header["D_IMRPAP"])
        altitude = np.deg2rad(header["ALTITUDE"])
        hwp_theta = np.deg2rad(header["U_HWPANG"])
        imr_theta = np.deg2rad(header["D_IMRANG"])
        qwp1 = np.deg2rad(header["U_QWP1"])
        qwp2 = np.deg2rad(header["U_QWP2"])

        M = mueller_matrix_model(
            camera=header["U_CAMERA"],
            filter=header["U_FILTER"],
            flc_state=header["U_FLCSTT"],
            qwp1=qwp1,
            qwp2=qwp2,
            imr_theta=imr_theta,
            hwp_theta=hwp_theta,
            pa=pa,
            altitude=altitude,
        )
        # only keep the X -> I terms
        mueller_mats.append(M[0])
        # derotate frame to N up
        cube.append(derotate_frame(frame, header["D_IMRPAD"] + 140.4))

    mueller_mats = np.array(mueller_mats)
    cube = np.array(cube)
    return mueller_matrix_calibration(mueller_mats, cube)


def polarization_calibration(filenames, method="model", output=None, skip=False):
    if output is None:
        indir = Path(filenames[0]).parent
        hdr = fits.getheader(filenames[0])
        name = hdr["OBJECT"]
        date = hdr["DATE-OBS"].replace("-", " ")
        output = indir / f"{name}_{date}_stokes.fits"
    else:
        output = Path(output)

    if skip and output.exists():
        stokes_cube = fits.getdata(output)
    elif method == "model":
        stokes_cube = polarization_calibration_model(filenames)
    elif method == "triplediff":
        stokes_cube = polarization_calibration_triplediff(filenames)
    else:
        raise ValueError(
            f'\'method\' must be either "model" or "triplediff" (got {method})'
        )

    return stokes_cube
