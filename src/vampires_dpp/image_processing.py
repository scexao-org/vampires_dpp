import bottleneck as bn
import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.indexing import cutout_inds, frame_center, frame_radii


def shift_frame(data: ArrayLike, shift: tuple[float, float]) -> NDArray:
    """Shift a single frame by the given offset using Fourier-domain shifting.

    Supports sub-pixel shifts. Uses periodic boundary conditions.

    Parameters
    ----------
    data : ArrayLike
        2D frame to shift
    shift : tuple[float, float]
        Shift (dy, dx) in pixels

    Returns
    -------
    NDArray
        Shifted frame
    """
    data = np.asarray(data, dtype="f8")
    fy = np.fft.fftfreq(data.shape[-2])
    fx = np.fft.fftfreq(data.shape[-1])
    phase = np.exp(-2j * np.pi * (shift[0] * fy[:, None] + shift[1] * fx[None, :]))
    return np.real(np.fft.ifft2(np.fft.fft2(data) * phase))


def derotate_frame(
    data: ArrayLike, angle: float, center: list | tuple | None = None, **kwargs
) -> NDArray:
    """Rotate a single frame clockwise by the given angle in degrees.

    Parameters
    ----------
    data : ArrayLike
        2D frame to derotate
    angle : float
        Angle, in degrees
    center : Optional[list | tuple]
        Point defining the axis of rotation. If `None`, will use the frame center. Default is `None`.
    **kwargs
        Keyword arguments are passed to `warp_frame`

    Returns
    -------
    NDArray
        Derotated frame
    """
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    return warp_frame(data, M, **kwargs)


def warp_frame(data: ArrayLike, matrix, antialias: bool = False, **kwargs) -> NDArray:
    """Geometric frame warping using Lanczos4 interpolation with NaN padding by default.

    Parameters
    ----------
    data : ArrayLike
        2D image
    matrix : ArrayLike
        Geometric transformation matrix
    antialias : bool
        Apply Gaussian blur before warping to reduce aliasing when downsampling.
    **kwargs
        Keyword arguments passed to opencv (e.g. `borderValue`, `borderMode`).

    Returns
    -------
    NDArray
        Warped frame
    """
    default_kwargs = {
        "flags": cv2.INTER_LANCZOS4,
        "borderMode": cv2.BORDER_CONSTANT,
        "borderValue": np.nan,
    }
    default_kwargs.update(**kwargs)
    shape = (data.shape[1], data.shape[0])
    if antialias:
        sigma = 0.5 * (1 - 1 / np.min(matrix.diagonal()))
        data = cv2.GaussianBlur(data.astype("f4"), (3, 3), sigma)
    return cv2.warpAffine(data.astype("f4"), matrix.astype("f4"), shape, **default_kwargs)


def derotate_cube(data: ArrayLike, angles: ArrayLike | float, **kwargs) -> NDArray:
    """Derotate a cube clockwise frame-by-frame with the corresponding angle vector.

    Parameters
    ----------
    data : ArrayLike
        3D cube to derotate
    angles : ArrayLike | float
        If a vector, will derotate each frame by the corresponding angle. If a float, will derotate each frame by the same value.

    Returns
    -------
    NDArray
        Derotated cube
    """
    angles = np.asarray(angles)
    rotated = np.empty_like(data)
    if angles.size == 1:
        angles = np.full(rotated.shape[0], angles)
    for idx in range(rotated.shape[0]):
        rotated[idx] = derotate_frame(data[idx], angles[idx], **kwargs)
    return rotated


def shift_cube(cube: ArrayLike, shifts: ArrayLike) -> NDArray:
    """Translate each frame in a cube using vectorized Fourier-domain shifting.

    Processes the entire cube in a single FFT call with no Python loop.

    Parameters
    ----------
    cube : ArrayLike
        3D cube (nframes, ny, nx)
    shifts : ArrayLike
        Array of (dy, dx) pairs, one for each frame in the input cube

    Returns
    -------
    NDArray
        Shifted cube
    """
    cube = np.asarray(cube, dtype="f8")
    shifts = np.asarray(shifts)
    fy = np.fft.fftfreq(cube.shape[-2])
    fx = np.fft.fftfreq(cube.shape[-1])
    phase = np.exp(
        -2j
        * np.pi
        * (
            shifts[:, 0, None, None] * fy[None, :, None]
            + shifts[:, 1, None, None] * fx[None, None, :]
        )
    )
    return np.real(np.fft.ifft2(np.fft.fft2(cube) * phase))


def radial_profile_image(frame: NDArray, fwhm: float = 3) -> NDArray:
    rs = frame_radii(frame)
    r_bins = np.arange(int(rs.min()), int(rs.max()) + 1)
    profile = np.array(
        [np.nanmedian(frame[(rs >= r - fwhm / 2) & (rs < r + fwhm / 2)]) for r in r_bins]
    )
    r_idx = np.clip(np.round(rs).astype(int) - int(rs.min()), 0, len(profile) - 1)
    return profile[r_idx]


def pad_cube(cube, pad_width: int, header=None, **pad_kwargs):
    new_shape = (cube.shape[0], cube.shape[1] + 2 * pad_width, cube.shape[2] + 2 * pad_width)
    output = np.empty_like(cube, shape=new_shape)

    for idx in range(cube.shape[0]):
        output[idx] = np.pad(cube[idx], pad_width, constant_values=np.nan)
    return output, header


def crop_to_nans_inds(data: NDArray) -> NDArray:
    """
    Crop numpy array to min/max indices that have finite values. In other words,
    trims the edges off where everything is NaN.
    """
    is_finite = np.isfinite(data)
    ndim_range = range(data.ndim)
    axes = tuple(set(ndim_range) - set(ndim_range[-2:]))
    finite_x = np.where(np.any(is_finite, axis=axes))[0]
    finite_y = np.where(np.any(is_finite, axis=axes))[0]

    min_x, max_x = finite_x[0], finite_x[-1]
    min_y, max_y = finite_y[0], finite_y[-1]
    cy, cx = frame_center(data)
    radius = max(max_x - cx, cx - min_x, max_y - cy, cy - min_y)
    return cutout_inds(data, center=(cy, cx), window=int(radius * 2))


def adaptive_sigma_clip_mask(data: NDArray, sigma: float = 10, boxsize: int = 8) -> NDArray:
    """Compute a sigma-clip bad pixel mask using non-overlapping local blocks.

    Parameters
    ----------
    data : NDArray
        2D image
    sigma : float
        Sigma threshold for clipping
    boxsize : int
        Size of the local block for computing statistics

    Returns
    -------
    NDArray
        Boolean mask, True where pixels are clipped
    """
    ny, nx = data.shape
    pad_y = (-ny) % boxsize
    pad_x = (-nx) % boxsize
    padded = np.pad(data, ((0, pad_y), (0, pad_x)), constant_values=np.nan)
    ny_p, nx_p = padded.shape

    # reshape into (nblocks_y, nblocks_x, boxsize, boxsize)
    blocks = padded.reshape(ny_p // boxsize, boxsize, nx_p // boxsize, boxsize).transpose(
        0, 2, 1, 3
    )
    with np.errstate(all="ignore"):
        med = np.nanmedian(blocks, axis=(-2, -1), keepdims=True)
        std = np.nanstd(blocks, axis=(-2, -1), keepdims=True)
    mask = np.abs(blocks - med) > sigma * std
    return mask.transpose(0, 2, 1, 3).reshape(ny_p, nx_p)[:ny, :nx]


def create_footprint(cube, angles):
    mask = np.isfinite(cube)
    derot = derotate_cube(mask.astype(float), angles)
    return bn.nanmean(derot, axis=0)
