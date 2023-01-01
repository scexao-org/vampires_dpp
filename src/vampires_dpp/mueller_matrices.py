from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.constants import PUPIL_OFFSET

__all__ = (
    "hwp",
    "qwp",
    "waveplate",
    "rotator",
    "generic",
    "linear_polarizer",
    "wollaston",
)


def hwp(theta=0) -> NDArray:
    """
    Return the Mueller matrix for an ideal half-wave plate (HWP) with fast-axis oriented to the given angle, in radians.

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

    >>> hwp(np.radians(45))
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
    """
    Return the Mueller matrix for an ideal quarter-wave plate (QWP) with fast-axis oriented to the given angle, in radians.

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

    >>> qwp(np.radians(45))
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
    """
    Return the Mueller matrix for a waveplate with arbitrary phase retardance.

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

    >>> waveplate(np.radians(45), np.pi/2) # QWP at 45°
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
    """
    Return a generic optic with diattenuation ``epsilon`` and phase retardance ``delta`` oriented at angle ``theta``.

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
    """
    Return the Mueller matrix for rotation clockwise about the optical axis.

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

    >>> rotator(np.radians(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        ((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1))
    )
    return M


def linear_polarizer(theta=0) -> NDArray:
    """
    Return the Mueller matrix for an ideal linear polarizer oriented at the given angle.

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

    >>> linear_polarizer(np.radians(45))
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


def mirror(theta=0) -> NDArray:
    """
    Return the Mueller matrix for an ideal mirror.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the mirror)

    Examples
    --------
    >>> mirror()
    array([[ 1,  0,  0,  0],
           [ 0,  1,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0,  0, -1]])
    """
    M = hwp(theta=theta)
    return M


def wollaston(ordinary: bool = True, eta=1) -> NDArray:
    """
    Return the Mueller matrix for a Wollaston prism or polarizing beamsplitter.

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
    if ordinary:
        eta = eta
    else:
        eta = -eta

    M = np.array(
        (
            (1, eta, 0, 0),
            (eta, 1, 0, 0),
            (0, 0, np.sqrt(1 - eta**2), 0),
            (0, 0, 0, np.sqrt(1 - eta**2)),
        )
    )
    return 0.5 * M


def instrumental(cQ=0, cU=0, cV=0):
    M = np.eye(4)
    M[1:, 0] = (cQ, cU, cV)
    return M


def mueller_matrix_triplediff(camera, flc_state, theta, hwp_theta):
    flc_theta = 0 if flc_state == 1 else np.pi / 4
    M = np.linalg.multi_dot(
        (
            wollaston(camera == 1),  # Polarizing beamsplitter
            hwp(theta=flc_theta),  # FLC
            rotator(-theta),
            mirror(),
            hwp(theta=hwp_theta),  # HWP angle
        )
    )
    if camera == 1:
        M *= 0.65
    return M


# filter: hwp_phi, imr_phi, flc1_phi, flc2_phi, flc1_theta, flc2_theta, dp
CAL_DICT = {
    "775-50": {
        "hwp_delta": 2.90571,
        "imr_delta": 3.15131,
        "flc_delta": (1.63261, 2.43131),
        "flc_theta": (-0.33141, 0.84601),
        "throughput": (1, 0.90641),
    },
    "750-50": {
        "hwp_delta": 3.02061,
        "imr_delta": 3.01341,
        "flc_delta": (5.22121, 2.44421),
        "flc_theta": (-0.39021, 0.91411),
        "pbs_throughput": (1, 1.01511),
    },
    "725-50": {
        "hwp_delta": 3.32281,
        "imr_delta": 3.46251,
        "flc_delta": (0.74391, 3.85451),
        "flc_theta": (-0.80311, 0.89701),
        "pbs_throughput": (1, 1.07621),
    },
    "675-50": {
        "hwp_delta": 3.42041,
        "imr_delta": 4.27231,
        "flc_delta": (1.05581, 3.60451),
        "flc_theta": (-0.12211, 1.00821),
        "pbs_throughput": (1, 1.12611),
    },
    "625-50": {
        "hwp_delta": 2.69971,
        "imr_delta": 1.38721,
        "flc_delta": (-57.52101, 2.57911),
        "flc_theta": (-0.32851, 0.84681),
        "pbs_throughput": (1, 1.16191),
    },
}


def mueller_matrix_model(
    camera,
    filter,
    flc_state,
    qwp1,
    qwp2,
    imr_theta,
    hwp_theta,
    pa,
    altitude,
    pupil_offset=PUPIL_OFFSET,
):
    cam_idx = 0 if camera == 1 else 1
    flc_ang = 0 if flc_state == 1 else np.pi / 4
    cals = CAL_DICT[filter]
    M = np.linalg.multi_dot(
        (
            wollaston(camera == 1),  # Polarizing beamsplitter
            hwp(theta=flc_ang),  # FLC
            # mirror(),
            # rotator(theta=-pupil_offset),
            # mirror(),
            qwp(theta=qwp2),  # QWP 2
            qwp(theta=qwp1),  # QWP 1
            waveplate(theta=imr_theta, delta=cals["imr_delta"]),  # AO 188 K-mirror
            waveplate(theta=hwp_theta, delta=cals["hwp_delta"]),  # AO 188 HWP,
            rotator(theta=altitude),
            mirror(),
            rotator(theta=pa),
        )
    )
    if camera == 1:
        M *= 0.65
    return M
