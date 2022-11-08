import numpy as np
from numpy.typing import NDArray


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


def mirror() -> NDArray:
    """
    Return the Mueller matrix for an ideal mirror.

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
    M = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)))
    return M


def wollaston(ordinary: bool = True, throughput=1) -> NDArray:
    """
    Return the Mueller matrix for a Wollaston prism or polarizing beamsplitter.

    Parameters
    ----------
    ordinary : bool, optional
        Return the ordinary beam's Mueller matrix, by default True
    throughput : float, optional
        For imperfect beamsplitters, the throughput of the matrix, by default 1

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

    >>> wollaston(False, throughput=1.2)
    array([[ 0.6, -0.6,  0. ,  0. ],
           [-0.6,  0.6, -0. ,  0. ],
           [ 0. , -0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ]])
    """
    if ordinary:
        M = linear_polarizer(0)
    else:
        M = linear_polarizer(np.pi / 2)

    return throughput * M
