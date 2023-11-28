import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.stats import biweight_location
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.indexing import cutout_inds, frame_center, frame_radii
from vampires_dpp.organization import dict_from_header
from vampires_dpp.util import delta_angle, load_fits

from .paths import any_file_newer, get_paths


def shift_frame(data: ArrayLike, shift: list | tuple, **kwargs) -> NDArray:
    """Shifts a single frame by the given offset

    Parameters
    ----------
    data : ArrayLike
        2D frame to shift
    shift : list | Tuple
        Shift (dy, dx) in pixels
    **kwargs
        Keyword arguments are passed to `warp_frame`

    Returns
    -------
    NDArray
        Shifted frame
    """
    M = np.float32(((1, 0, shift[1]), (0, 1, shift[0])))
    return warp_frame(data, M, **kwargs)


def derotate_frame(
    data: ArrayLike, angle: float, center: list | tuple | None = None, **kwargs
) -> NDArray:
    """Derotates a single frame by the given angle.

    Parameters
    ----------
    data : ArrayLike
        2D frame to derotate
    angle : float
        Angle, in degrees
    center : Optional[list | Tuple]
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


def warp_frame(data: ArrayLike, matrix, **kwargs) -> NDArray:
    """Geometric frame warping. By default will use bicubic interpolation with `NaN` padding.

    Parameters
    ----------
    data : ArrayLike
        2D image
    matrix : ArrayLike
        Geometric transformation matrix
    **kwargs
        Keyword arguments are passed to opencv. Important keywords like `borderValue`, `borderMode`, and `flags` can customize the padding and interpolation behavior of the transformation.

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
    return cv2.warpAffine(data.astype("f4"), matrix, shape, **default_kwargs)


def derotate_cube(data: ArrayLike, angles: ArrayLike | float, **kwargs) -> NDArray:
    """Derotates a cube frame-by-frame with the corresponding derotation angle vector.

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
    # reverse user-given center because scikit-image
    # uses swapped axes for this parameter only
    angles = np.asarray(angles)
    rotated = np.empty_like(data)
    # if angles is a scalar, broadcoast along frame index
    if angles.size == 1:
        angles = np.full(rotated.shape[0], angles)
    for idx in range(rotated.shape[0]):
        rotated[idx] = derotate_frame(data[idx], angles[idx], **kwargs)
    return rotated


def shift_cube(cube: ArrayLike, shifts: ArrayLike, **kwargs) -> NDArray:
    """Translate each frame in a cube.

    Parameters
    ----------
    cube : ArrayLike
        3D cube
    shifts : ArrayLike
        Array of (dy, dx) pairs, one for each frame in the input cube

    Returns
    -------
    NDArray
        Shifted cube
    """
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = shift_frame(cube[i], shifts[i], **kwargs)
    return out


def weighted_collapse(data: ArrayLike, angles: ArrayLike, **kwargs) -> NDArray:
    """Do a variance-weighted simultaneous derotation and collapse of ADI data based on the algorithm presented in `Bottom 2017 <https://ui.adsabs.harvard.edu/abs/2017RNAAS...1...30B>`_.

    Parameters
    ----------
    data : ArrayLike
        3D cube
    angles : ArrayLike
        Vector of angles to derotate data cube by

    Returns
    -------
    NDArray
        2D derotated and collapsed frame
    """
    variance_frame = np.var(data, axis=0, keepdims=True)

    # if the variance is zero, return the mean
    if np.allclose(variance_frame, 0):
        derotated = derotate_cube(data, angles, **kwargs)
        return np.nanmean(derotated, 0)

    # expand the variance frame into a cube
    variance_cube = np.repeat(variance_frame, data.shape[0], axis=0)
    # derotate both signal and variance
    derotated_data = derotate_cube(data, angles, **kwargs)
    derotated_variance = derotate_cube(variance_cube, angles, **kwargs)
    derotated_variance[derotated_variance == 0] = np.inf
    # calculate weighted sum
    numer = np.nansum(derotated_data / derotated_variance, axis=0)
    denom = np.nansum(1 / derotated_variance, axis=0)
    weighted_frame = numer / denom
    return weighted_frame


def collapse_cube(
    cube: NDArray, method: str = "median", header: fits.Header | None = None, **kwargs
) -> tuple[NDArray, fits.Header | None]:
    """Collapse a cube along its time axis

    Parameters
    ----------
    cube : NDArray
        3D cube
    method : str, optional
        One of "median", "mean", "varmean", "biweight", by default "median"
    header : fits.Header, optional
        FITS header, which will be updated with metadata if provided. By default None

    Returns
    -------
    Tuple[NDArray, Optional[fits.Header]]
        Tuple of collapsed frame and updated header. If header is not provided, will be None.
    """
    # clean inputs
    match method.strip().lower():
        case "median":
            # suppress all-nan axis warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = np.nanmedian(cube, axis=0, overwrite_input=True)
        case "mean":
            # suppress all-nan axis warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = np.nanmean(cube, axis=0)
        case "varmean":
            # suppress all-nan axis warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weights = 1 / np.nanvar(cube, axis=(1, 2), keepdims=True)
                frame = np.sum(cube * weights, axis=0) / np.sum(weights)
        case "biweight":
            frame = biweight_location(cube, axis=0, c=6, skip_nans=True)

    if header is not None:
        header["COL_METH"] = method, "DPP cube collapse method"

    return frame, header


def collapse_cube_file(filename, force: bool = False, **kwargs) -> Path:
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = load_fits(path, header=True)
    frame, header = collapse_cube(cube, header=header, **kwargs)

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath


def combine_frames(frames, headers=None, **kwargs):
    cube = np.array(frames)

    if headers is not None:
        headers = combine_frames_headers(headers, **kwargs)

    return cube, headers


WCS_KEYS = {
    "WCSAXES",
    "CRPIX1",
    "CRPIX2",
    "CDELT1",
    "CDELT2",
    "CUNIT1",
    "CUNIT2",
    "CTYPE1",
    "CTYPE2",
    "CRVAL1",
    "CRVAL2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
}

RESERVED_KEYS = {
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "BSCALE",
    "BZERO",
    "BITPIX",
    "WAVEMIN",
    "WAVEMAX",
    "WAVEFWHM",
    "DLAMLAM",
} | WCS_KEYS


def combine_frames_headers(headers: Sequence[fits.Header], wcs=False):
    output_header = fits.Header()
    # let's make this easier with tables
    test_header = headers[0]
    table = pd.DataFrame([dict_from_header(header, fix=False) for header in headers])
    table.sort_values("MJD", inplace=True)
    # use a single header to get comments
    # which columns have only a single unique value?
    unique_values = table.apply(lambda col: col.unique())
    unique_mask = unique_values.apply(lambda values: len(values) == 1)
    unique_row = table.loc[0, unique_mask]
    for key, val in unique_row.items():
        output_header[key] = val, test_header.comments[key]

    # as a start, for everything else just median it
    for key in table.columns[~unique_mask]:
        if key in RESERVED_KEYS or table[key].dtype not in (int, float):
            continue
        try:
            # there is no way to check if comment exists a priori...
            comment = test_header.comments[key]
            is_err = "error" in comment
        except KeyError:
            comment = None
            is_err = False
        if is_err:
            stderr = np.sqrt(np.nanmean(table[key] ** 2) / len(table))
            output_header[key] = stderr * np.sqrt(np.pi / 2), comment
        else:
            output_header[key] = np.nanmedian(table[key]), comment

    ## everything below here has special rules for combinations
    # sum exposure times
    if "TINT" in table:
        output_header["TINT"] = (table["TINT"].sum(), "[s] total integrated exposure time")

    # get PA rotation
    if "PA" in table:
        output_header["PA-STR"] = table["PA-STR"].iloc[0], "[deg] par. angle at start"
        output_header["PA-END"] = table["PA-END"].iloc[-1], "[deg] par. angle at end"
        total_rot = delta_angle(output_header["PA-STR"], output_header["PA-END"])
        output_header["PA-ROT"] = total_rot, "[deg] total par. angle rotation"

    # average position using average angle formula
    ras = Angle(table["RA"], unit=u.hourangle)
    ave_ra = np.arctan2(np.sin(ras.rad).mean(), np.cos(ras.rad).mean())
    decs = Angle(table["DEC"], unit=u.deg)
    ave_dec = np.arctan2(np.sin(decs.rad).mean(), np.cos(decs.rad).mean())
    output_header["RA"] = (
        Angle(ave_ra * u.rad).to_string(unit=u.hourangle, sep=":"),
        test_header.comments["RA"],
    )
    output_header["DEC"] = (
        Angle(ave_dec * u.rad).to_string(unit=u.deg, sep=":"),
        test_header.comments["DEC"],
    )
    if wcs:
        # need to get average CRVALs and PCs
        output_header["CRVAL1"] = np.rad2deg(ave_ra)
        output_header["CRVAL2"] = np.rad2deg(ave_dec)
        output_header["PC1_1"] = table["PC1_1"].mean()
        output_header["PC1_2"] = table["PC1_2"].mean()
        output_header["PC2_1"] = table["PC2_1"].mean()
        output_header["PC2_2"] = table["PC2_2"].mean()
    else:
        wcskeys = filter(
            lambda k: any(wcsk.startswith(k) for wcsk in WCS_KEYS), output_header.keys()
        )
        for k in wcskeys:
            del output_header[k]

    return output_header


def combine_frames_files(filenames, output, *, force: bool = False, crop: bool = False, **kwargs):
    path = Path(output)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path

    frames = []
    headers = []
    for filename in filenames:
        # use memmap=False to avoid "too many files open" effects
        # another way would be to set ulimit -n <MAX_FILES>
        frame, header = load_fits(filename, header=True, memmap=False)
        frames.append(frame)
        headers.append(header)

    if crop:
        frames_arr = np.array(frames)
        inds = crop_to_nans_inds(frames_arr)
        frames = frames_arr[inds]
    cube, header = combine_frames(frames, headers, **kwargs)
    fits.writeto(path, cube, header=header, overwrite=True)
    return path


def collapse_frames(frames, headers=None, method="median", **kwargs):
    cube, header = combine_frames(frames, headers=headers, **kwargs)
    return collapse_cube(cube, method=method, header=header)


def collapse_frames_files(filenames, output, force=False, cubes=False, quiet=True, **kwargs):
    path = Path(output)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path

    frames = []
    headers = []
    _iter = tqdm.tqdm(filenames, "Collecting files") if not quiet else filenames
    for filename in _iter:
        # use memmap=False to avoid "too many files open" effects
        # another way would be to set ulimit -n <MAX_FILES>
        frame, header = load_fits(filename, header=True, memmap=False)
        if cubes:
            rand_idx = np.random.default_rng().integers(low=0, high=len(frame))
            frame = frame[rand_idx]
        frames.append(frame)
        headers.append(header)

    frame, header = collapse_frames(frames, headers=headers, **kwargs)
    fits.writeto(path, frame, header=header, overwrite=True)
    return path


def correct_distortion(
    frame: ArrayLike, angle: float = 0, scale: float = 1, header=None, center=None, **kwargs
):
    """Rotate and scale a single frame to match with a pinhole grid

    Parameters
    ----------
    frame : ArrayLike
        Input frame to transform
    angle : float, optional
        Clockwise angle to rotate frame by, by default 0
    scale : float, optional
        Amount of radial scaling, by default 1
    header : Header, optional
        FITS header to update, by default None
    center : (y, x), optional
        Frame center, if None will default to geometric frame center, by default None

    Returns
    -------
    corrected_frame, header
        The distortion-corrected frame and header. If `header` was not provided, this will be `None`.
    """
    if center is None:
        center = frame_center(frame)
    # if downsizing, low-pass filter to reduce moire effect
    if scale < 1:
        frame = cv2.GaussianBlur(frame, sigmaX=0.5 / scale)
    # scale and rotate frames with single transform
    M = cv2.getRotationMatrix2D(center[::-1], angle=angle, scale=scale)
    corr_frame = warp_frame(frame, M, **kwargs)
    # update header
    if header is not None:
        header["DPP_SCAL"] = scale, "scaling ratio for distortion correction"
        header["DPP_ANGL"] = angle, "deg, offset angle for distortion correction"
    return corr_frame, header


def correct_distortion_cube(
    cube: NDArray, angle: float = 0, scale: float = 1, header=None, center=None, **kwargs
):
    """Rotate and scale a all frames in a cube to match with a pinhole grid

    Parameters
    ----------
    cube : NDArray
        Input cube to transform
    angle : float, optional
        Clockwise angle to rotate frames by, by default 0
    scale : float, optional
        Amount of radial scaling, by default 1
    header : Header, optional
        FITS header to update, by default None
    center : (y, x), optional
        Frame center, if None will default to geometric frame center, by default None

    Returns
    -------
    corrected_cube, header
        The distortion-corrected cube and header. If `header` was not provided, this will be `None`.
    """
    if center is None:
        center = frame_center(cube)
    # scale and retate frames with single transformpty_like(cube)
    M = cv2.getRotationMatrix2D(center[::-1], angle=angle, scale=scale)
    corr_cube = np.empty_like(cube)
    for i in range(cube.shape[0]):
        # if downsizing, low-pass filter to reduce moire effect
        frame = cv2.GaussianBlur(cube[i], (0, 0), sigmaX=0.5 / scale) if scale < 1 else cube[i]
        corr_cube[i] = warp_frame(frame, M, **kwargs)
    # update header
    if header is not None:
        header["DPP_SCAL"] = scale, "scaling ratio for distortion correction"
        header["DPP_ANGL"] = angle, "deg, offset angle for distortion correction"
    return corr_cube, header


def make_diff_image(cam1_file, cam2_file, outname=None, force=False):
    if outname is not None:
        outname = Path(outname)
    else:
        stem = re.sub("_cam[12]", "", cam1_file.stem)
        outname = cam1_file.with_name(f"{stem}_diff.fits")

    if not force and outname.is_file() and not any_file_newer((cam1_file, cam2_file), outname):
        return outname

    cam1_frame, header = load_fits(cam1_file, header=True)
    cam2_frame, header2 = load_fits(cam2_file, header=True)

    if header["MJD"] != header2["MJD"]:
        msg = f"{cam1_file.name} has MJD {header['MJD']}\n{cam2_file.name} has MJD {header2['MJD']}"
        raise ValueError(msg)
    if "U_FLC" in header and header["U_FLC"] != header2["U_FLC"]:
        msg = f"{cam1_file.name} has FLC state {header['U_FLC']}\n{cam2_file.name} has FLC state {header2['U_FLC']}"
        raise ValueError(msg)

    diff = cam1_frame - cam2_frame
    summ = cam1_frame + cam2_frame

    stack = np.asarray((diff, summ))

    # prepare header
    del header["U_CAMERA"]

    fits.writeto(outname, stack, header=header, overwrite=True)
    return outname


def radial_profile_image(frame, fwhm=3):
    rs = frame_radii(frame)
    bins = np.arange(rs.min(), rs.max())
    output = np.zeros_like(frame)
    for r in bins:
        mask = (rs >= r - fwhm / 2) & (rs < r + fwhm / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            output[mask] = np.nanmedian(frame[mask])

    return output


def pad_cube(cube, pad_width: int, header=None, **pad_kwargs):
    new_shape = (cube.shape[0], cube.shape[1] + 2 * pad_width, cube.shape[2] + 2 * pad_width)
    output = np.empty_like(cube, shape=new_shape)

    for idx in range(cube.shape[0]):
        output[idx] = np.pad(cube[idx], pad_width, constant_values=np.nan)
    return output, header


def generate_footprint(cube):
    mask = np.isfinite(cube).astype(int)
    return np.mean(mask, axis=0)


def crop_to_nans_inds(data: NDArray) -> NDArray:
    """
    Crop numpy array to min/max indices that have finite values. In other words,
    trims the edges off where everything is NaN.
    """
    # determine first index that contains finite value
    is_finite = np.isfinite(data)
    ndim_range = range(data.ndim)
    # reduce over every axis except the image axes
    axes = tuple(set(ndim_range) - set(ndim_range[-2:]))
    finite_x = np.where(np.any(is_finite, axis=axes))[0]
    finite_y = np.where(np.any(is_finite, axis=axes))[0]

    min_x, max_x = finite_x[0], finite_x[-1]
    min_y, max_y = finite_y[0], finite_y[-1]
    cy, cx = frame_center(data)
    # don't just take min to max indices, calculate the radius
    # of each extreme to the center and keep everything centered
    radius = max(max_x - cx, cx - min_x, max_y - cy, cy - min_y)
    return cutout_inds(data, center=(cy, cx), window=int(radius * 2))
