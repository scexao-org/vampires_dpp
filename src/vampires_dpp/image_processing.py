from astropy.io import fits
import astropy.units as u
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.ndimage import fourier_shift
from numpy.typing import ArrayLike, NDArray
from typing import Union
import cv2
from astropy.coordinates import Angle

from vampires_dpp.headers import dict_from_header


def shift_frame_fft(data: ArrayLike, shift):
    data_freq = np.fft.fft2(data)
    filt = fourier_shift(data_freq, shift)
    shifted = np.real(np.fft.ifft2(filt))
    return shifted


def shift_frame(data: ArrayLike, shift, **kwargs):
    M = np.float32(((1, 0, shift[1]), (0, 1, shift[0])))
    return distort_frame(data, **kwargs)


def derotate_frame(data: ArrayLike, angle, center=None, **kwargs):
    """_summary_

    Parameters
    ----------
    data : ArrayLike
        _description_
    angle : _type_
        ANGLE CONVENTION IS CW, OPPOSITE OF ASTROMETRIC
    center : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    return distort_frame(data, M, **kwargs)


def warp_frame(data: ArrayLike, shift=0, angle=0, center=None, **kwargs):
    if center is None:
        center = frame_center(data)
    M = cv2.getRotationMatrix2D(center[::-1], -angle, 1)
    M[::-1, 2] += shift
    return distort_frame(data, M, **kwargs)


def distort_frame(data: ArrayLike, matrix, **kwargs):
    default_kwargs = {
        "flags": cv2.INTER_CUBIC,
        "borderMode": cv2.BORDER_CONSTANT,
        "borderValue": np.nan,
    }
    default_kwargs.update(**kwargs)
    shape = (data.shape[1], data.shape[0])
    return cv2.warpAffine(data.astype("f4"), matrix, shape, **default_kwargs)


def derotate_cube(data: ArrayLike, angles: Union[ArrayLike, float], **kwargs):
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


def collapse_file(filename, output=None, skip=False, **kwargs):
    if output is None:
        path = Path(filename)
        output = path.with_name(f"{path.stem}_collapsed{path.suffix}")
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    cube, header = fits.getdata(filename, header=True)
    frame = np.median(cube, 0, overwrite_input=True)

    fits.writeto(output, frame, header=header, overwrite=True)
    return output


def shift_cube(cube: ArrayLike, shifts: ArrayLike, **kwargs):
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = shift_frame(cube[i], shifts[i], **kwargs)
    return out


def weighted_collapse(data: ArrayLike, angles: ArrayLike, **kwargs):
    variance_frame = np.var(data, axis=0, keepdims=True)
    # if the variance is zero, return the mean
    if np.allclose(variance_frame, 0):
        derotated = derotate_cube(data, angles, **kwargs)
        return np.mean(derotated, 0)

    # expand the variance frame into a cube
    variance_cube = np.repeat(variance_frame, data.shape[0], axis=0)
    # derotate both signal and variance
    derotated_data = derotate_cube(data, angles, **kwargs)
    derotated_variance = derotate_cube(variance_cube, angles, **kwargs)
    # calculate weighted sum
    numer = np.sum(derotated_data / derotated_variance, 0)
    denom = np.sum(1 / derotated_variance, 0)
    weighted_frame = numer / denom
    return weighted_frame


def frame_center(image: ArrayLike):
    """
    Find the center of the frame or cube in pixel coordinates

    Parameters
    ----------
    image : ArrayLike
        N-D array with the final two dimensions as the (y, x) axes.

    Returns
    -------
    (cy, cx)
        A tuple of the image center in pixel coordinates
    """
    ny = image.shape[-2]
    nx = image.shape[-1]
    return (ny - 1) / 2, (nx - 1) / 2


def frame_radii(frame: ArrayLike, center=None) -> NDArray:
    """
    Return the radii of pixels around ``center`` in the image

    Parameters
    ----------
    frame : ArrayLike
        Input frame
    center : Tuple, optional
        The center to calculate radii from. If None, will default to the frame center. By default None

    Returns
    -------
    NDArray
        Matrix with frame radii
    """
    if center is None:
        center = frame_center(frame)
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    radii = np.hypot(Ys - center[0], Xs - center[1])
    return radii


def frame_angles(frame: ArrayLike, center=None):
    """
    Return the angles of pixels around ``center`` in the image

    Parameters
    ----------
    frame : ArrayLike
        Input frame
    center : Tuple, optional
        The center to calculate radii from. If None, will default to the frame center. By default None

    Returns
    -------
    NDArray
        Matrix with frame angles
    """
    if center is None:
        center = frame_center(frame)
    Ys, Xs = np.ogrid[0 : frame.shape[-2], 0 : frame.shape[-1]]
    # y flip + x flip
    thetas = np.arctan2(Xs - center[1], center[0] - Ys)
    return thetas


def collapse_cube(cube: ArrayLike, method: str = "median", header=None, **kwargs):
    # clean inputs
    method = method.strip().lower()

    if method == "median":
        frame = np.median(cube, axis=0, **kwargs)

    if header is not None:
        header = collapse_cube_header(header, method=method)

    return frame, header


def collapse_cube_header(header, method="median"):
    header["VPP_COLL"] = method, "VAMPIRES DPP cube collapse method"
    return header


def combine_frames(frames, headers=None, **kwargs):
    cube = np.array(frames)

    if headers is not None:
        headers = combine_frames_headers(headers, **kwargs)

    return cube, headers


def combine_frames_headers(headers, wcs=False):
    output_header = fits.Header()
    # let's make this easier with tables
    table = pd.DataFrame([dict_from_header(header) for header in headers])

    # which columns have only a single unique value?
    unique_values = table.apply(lambda col: col.unique())
    unique_mask = unique_values.apply(lambda values: len(values) == 1)
    unique_row = table.loc[0, unique_mask]
    for key, val in unique_row.items():
        output_header[key] = val

    # average position using average angle formula
    ras = Angle(table["RA"], unit=u.hourangle)
    ave_ra = np.arctan2(np.sin(ras.rad).mean(), np.cos(ras.rad).mean())
    decs = Angle(table["DEC"], unit=u.deg)
    ave_dec = np.arctan2(np.sin(decs.rad).mean(), np.cos(decs.rad).mean())
    output_header["RA"] = Angle(ave_ra * u.rad).to_string(unit=u.hourangle, sep=":")
    output_header["DEC"] = Angle(ave_dec * u.rad).to_string(unit=u.deg, sep=":")
    if wcs:
        # need to get average CRVALs and PCs
        output_header["CRVAL1"] = np.rad2deg(ave_ra)
        output_header["CRVAL2"] = np.rad2deg(ave_dec)
        output_header["PC1_1"] = table["PC1_1"].mean()
        output_header["PC1_2"] = table["PC1_2"].mean()
        output_header["PC2_1"] = table["PC2_1"].mean()
        output_header["PC2_2"] = table["PC2_2"].mean()
    else:
        for k in (
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
        ):
            del table[k]

    return output_header


def combine_frames_files(filenames, output, skip=False, **kwargs):
    path = Path(output)
    if skip and path.is_file():
        return path

    frames = []
    headers = []
    for filename in filenames:
        frame, header = fits.getdata(filename, header=True)
        frames.append(frame)
        headers.append(header)

    cube, header = combine_frames(frames, headers, **kwargs)

    fits.writeto(path, cube, header=header, overwrite=True)
    return path
