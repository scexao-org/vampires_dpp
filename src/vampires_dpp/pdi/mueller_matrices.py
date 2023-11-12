from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from vampires_dpp.constants import SUBARU_LOC

__all__ = ("hwp", "qwp", "waveplate", "rotator", "generic", "linear_polarizer", "wollaston")


def hwp(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal half-wave plate (HWP) with fast-axis oriented to the given angle, in radians.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the HWP

    Examples
    --------
    >>> hwp(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    >>> hwp(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0., -1.]])


    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2 - sin2t**2, 2 * cos2t * sin2t, 0),
            (0, 2 * cos2t * sin2t, sin2t**2 - cos2t**2, 0),
            (0, 0, 0, -1),
        )
    )
    return M


def qwp(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal quarter-wave plate (QWP) with fast-axis oriented to the given angle, in radians.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the QWP

    Examples
    --------
    >>> qwp(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])

    >>> qwp(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1., -0.,  0.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2, cos2t * sin2t, -sin2t),
            (0, cos2t * sin2t, sin2t**2, cos2t),
            (0, sin2t, -cos2t, 0),
        )
    )
    return M


def waveplate(theta=0, delta=0) -> NDArray:
    """Return the Mueller matrix for a waveplate with arbitrary phase retardance.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians. By default 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the waveplate

    Examples
    --------
    >>> waveplate(0, np.pi) # HWP
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0., -0., -1.]])

    >>> waveplate(np.deg2rad(45), np.pi/2) # QWP at 45°
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1., -0.,  0.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    cosd = np.cos(delta)
    sind = np.sin(delta)
    a = (1 - cosd) * sin2t * cos2t
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2 + cosd * sin2t**2, a, -sind * sin2t),
            (0, a, sin2t**2 + cosd * cos2t**2, sind * cos2t),
            (0, sind * sin2t, -sind * cos2t, cosd),
        )
    )
    return M


def generic(theta=0, epsilon=0, delta=0) -> NDArray:
    """Return a generic optic with diattenuation ``epsilon`` and phase retardance ``delta`` oriented at angle ``theta``.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians, by default 0
    epsilon : float, optional
        Diattenuation, by default 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the optic

    Examples
    --------
    >>> generic() # Identity
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0., -0.,  1.]])

    >>> generic(epsilon=0.01, delta=np.pi) # mirror with diatt.
    array([[ 1.     ,  0.01   ,  0.     ,  0.     ],
           [ 0.01   ,  1.     ,  0.     , -0.     ],
           [ 0.     ,  0.     , -0.99995,  0.     ],
           [ 0.     ,  0.     , -0.     , -0.99995]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    cosd = np.cos(delta)
    sind = np.sin(delta)
    fac = np.sqrt((1 - epsilon) * (1 + epsilon))
    M = np.array(
        (
            (1, epsilon * cos2t, epsilon * sin2t, 0),
            (
                epsilon * cos2t,
                cos2t**2 + sin2t**2 * fac * cosd,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                -fac * sind * sin2t,
            ),
            (
                epsilon * sin2t,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                sin2t**2 + cos2t**2 * fac * cosd,
                fac * sind * cos2t,
            ),
            (0, fac * sind * sin2t, -fac * sind * cos2t, fac * cosd),
        )
    )

    return M


def rotator(theta=0) -> NDArray:
    """Return the Mueller matrix for rotation clockwise about the optical axis.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix

    Examples
    --------
    >>> rotator(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0., -0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> rotator(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1)))
    return M


def linear_polarizer(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal linear polarizer oriented at the given angle.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the linear polarizer

    Examples
    --------
    >>> linear_polarizer(0)
    array([[0.5, 0.5, 0. , 0. ],
           [0.5, 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]])

    >>> linear_polarizer(np.deg2rad(45))
    array([[0.5, 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0. ],
           [0.5, 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0. ]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, cos2t, sin2t, 0),
            (cos2t, cos2t**2, cos2t * sin2t, 0),
            (sin2t, cos2t * sin2t, sin2t**2, 0),
            (0, 0, 0, 0),
        )
    )
    return 0.5 * M


def mirror() -> NDArray:
    """Return the Mueller matrix for an ideal mirror.

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the mirror

    Examples
    --------
    >>> mirror()
    array([[ 1,  0,  0,  0],
           [ 0,  1,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0,  0, -1]])
    """
    return hwp(theta=0)


def wollaston(ordinary: bool = True, eta=1) -> NDArray:
    """Return the Mueller matrix for a Wollaston prism or polarizing beamsplitter.

    Parameters
    ----------
    ordinary : bool, optional
        Return the ordinary beam's Mueller matrix, by default True
    eta : float, optional
        For imperfect beamsplitters, the diattenuation of the optic, by default 1

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the selected output beam

    Examples
    --------
    >>> wollaston()
    array([[0.5, 0.5, 0. , 0. ],
           [0.5, 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]])

    >>> wollaston(False, eta=0.8)
    array([[ 0.5, -0.4,  0. ,  0. ],
           [-0.4,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0.3,  0. ],
           [ 0. ,  0. ,  0. ,  0.3]])
    """
    eta = eta if ordinary else -eta

    radicand = (1 - eta) * (1 + eta)
    M = np.array(
        ((1, eta, 0, 0), (eta, 1, 0, 0), (0, 0, np.sqrt(radicand), 0), (0, 0, 0, np.sqrt(radicand)))
    )
    return 0.5 * M


def instrumental(pQ=0, pU=0, pV=0):
    M = np.eye(4)
    M[1:, 0] = (pQ, pU, pV)
    return M


# filter: hwp_phi, imr_phi, flc1_phi, flc2_phi, flc1_theta, flc2_theta, dp
CAL_DICT = {
    "775-50": {
        "hwp_delta": 2 * np.pi * 0.463,
        "hwp_offset": np.deg2rad(-7.145),
        "imr_delta": 2 * np.pi * 0.5,
        "imr_offset": np.deg2rad(12.106),
        "optics_delta": 2 * np.pi * -0.209,
        "optics_diatt": 0.004,
        "optics_theta": np.deg2rad(-23.116),
        "flc_delta": 2 * np.pi * 0.389,
        "flc_offset": np.deg2rad(-15.277),
        "pbs_throughput": 0.548,
    },
    "750-50": {  # rebecca's new coeffs 08/29
        "hwp_delta": 2 * np.pi * 0.48,
        "hwp_offset": np.deg2rad(-2.062),
        "imr_delta": 2 * np.pi * 0.479,
        "imr_offset": np.deg2rad(0.174),
        "optics_delta": 2 * np.pi * -0.157,
        "optics_diatt": 0.001,
        "optics_theta": np.deg2rad(-26.953),
        "flc_delta": 2 * np.pi * 0.24,
        "flc_offset": np.deg2rad(-1.523),
        "pbs_throughput": 0.489,
    },
    "725-50": {
        "hwp_delta": 2 * np.pi * 0.465,
        "hwp_offset": np.deg2rad(-3.279),
        "imr_delta": 2 * np.pi * 0.446,
        "imr_offset": np.deg2rad(1.696),
        "optics_delta": 2 * np.pi * -0.098,
        "optics_diatt": 0.011,
        "optics_theta": np.deg2rad(-30.441),
        "flc_delta": 2 * np.pi * 0.285,
        "flc_offset": np.deg2rad(5.027),
        "pbs_throughput": 0.446,
    },
    "675-50": {
        "hwp_delta": 2 * np.pi * 0.451,
        "hwp_offset": np.deg2rad(0.996),
        "imr_delta": 2 * np.pi * 0.321,
        "imr_offset": np.deg2rad(2.769),
        "optics_delta": 2 * np.pi * -0.254,
        "optics_diatt": 0.05,
        "optics_theta": np.deg2rad(-17.143),
        "flc_delta": 2 * np.pi * 0.237,
        "flc_offset": np.deg2rad(-12.957),
        "pbs_throughput": 0.416,
    },
    "625-50": {
        "hwp_delta": 2 * np.pi * 0.433,
        "hwp_offset": np.deg2rad(1.083),
        "imr_delta": 2 * np.pi * 0.225,
        "imr_offset": np.deg2rad(-0.336),
        "optics_delta": 2 * np.pi * 0.002,
        "optics_diatt": 0.015,
        "optics_theta": np.deg2rad(55.617),
        "flc_delta": 2 * np.pi * 0.289,
        "flc_offset": np.deg2rad(-4.313),
        "pbs_throughput": 0.426,
    },
    "ideal": {
        "hwp_delta": np.pi,
        "hwp_offset": 0,
        "imr_delta": np.pi,
        "imr_offset": 0,
        "optics_delta": 0,
        "optics_diatt": 0,
        "optics_theta": 0,
        "flc_delta": np.pi,
        "flc_offset": 0,
        "pbs_throughput": 1,
    },
}


def mueller_matrix_from_header(header, adi_sync=True, ideal=False):
    filt = header["FILTER01"]
    if ideal:
        filt_dict = CAL_DICT["ideal"]
    elif filt in CAL_DICT:
        filt_dict = CAL_DICT[filt]
    else:
        msg = f"Could not find Mueller matrix coefficients for {filt} filter"
        raise RuntimeError(msg)

    pa_theta = np.deg2rad(header["PA"])
    alt = np.deg2rad(header["ALTITUDE"])
    az = np.deg2rad(header["AZIMUTH"] - 180)
    if adi_sync:
        lat = SUBARU_LOC.lat.rad
        hwp_offset = (
            0.5 * np.arctan2(np.sin(az), np.sin(alt) * np.cos(az) + np.cos(alt) * np.tan(lat)) + alt
        )
    else:
        hwp_offset = 0

    hwp_theta = np.deg2rad(header["RET-ANG1"]) + filt_dict["hwp_offset"] + hwp_offset
    imr_theta = np.deg2rad(header["D_IMRANG"]) + filt_dict["imr_offset"]
    flc_ang = 0 if header["U_FLC"] == "A" else np.pi / 4
    flc_theta = flc_ang + filt_dict["flc_offset"]
    beam = header["U_CAMERA"] == 1  # true if ordinary
    if "U_MBI" in header:
        flc_theta *= -1
        beam = not beam

    M = np.linalg.multi_dot(
        (
            wollaston(beam),
            waveplate(flc_theta, filt_dict["flc_delta"]),
            generic(
                filt_dict["optics_theta"], filt_dict["optics_diatt"], filt_dict["optics_delta"]
            ),
            waveplate(imr_theta, filt_dict["imr_delta"]),
            waveplate(hwp_theta, filt_dict["hwp_delta"]),
            rotator(-alt),
            mirror(),
            rotator(pa_theta),
        )
    )

    return M


def mueller_matrix_from_file(filename, outpath, force=False, **kwargs):
    if (
        not force
        and Path(outpath).exists()
        and Path(filename).stat().st_mtime < Path(outpath).stat().st_mtime
    ):
        return outpath

    headers = []
    mms = []
    with fits.open(filename) as hdul:
        for hdu in hdul[1:]:
            headers.append(hdu.header)
            mms.append(mueller_matrix_from_header(hdu.header, **kwargs).astype("f4"))
        prim_hdu = fits.PrimaryHDU(np.array(mms), hdul[0].header)
    hdus = (fits.ImageHDU(cube, hdr) for cube, hdr in zip(mms, headers))
    hdul = fits.HDUList([prim_hdu, *hdus])
    hdul.writeto(outpath, overwrite=True)
    return outpath
