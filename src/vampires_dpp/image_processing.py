import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import astropy.units as u
import cv2
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.stats import biweight_location
from numpy.typing import ArrayLike, NDArray

from vampires_dpp.indexing import frame_center, frame_radii
from vampires_dpp.organization import dict_from_header, header_table
from vampires_dpp.util import any_file_newer, get_paths


def shift_frame(data: ArrayLike, shift: list | Tuple, **kwargs) -> NDArray:
    """
    Shifts a single frame by the given offset

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
    data: ArrayLike, angle: float, center: Optional[list | Tuple] = None, **kwargs
) -> NDArray:
    """
    Derotates a single frame by the given angle.

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
    """
    Geometric frame warping. By default will use bicubic interpolation with `NaN` padding.

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
    """
    Derotates a cube frame-by-frame with the corresponding derotation angle vector.

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
    """
    Translate each frame in a cube.

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
    """
    Do a variance-weighted simultaneous derotation and collapse of ADI data based on the algorithm presented in `Bottom 2017 <https://ui.adsabs.harvard.edu/abs/2017RNAAS...1...30B>`_.

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
    cube: ArrayLike, method: str = "median", header: Optional[fits.Header] = None, **kwargs
) -> Tuple[NDArray, Optional[fits.Header]]:
    """
    Collapse a cube along its time axis

    Parameters
    ----------
    cube : ArrayLike
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

    cube, header = fits.getdata(
        path,
        header=True,
    )
    frame, header = collapse_cube(cube, header=header, **kwargs)

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath


def combine_frames(frames, headers=None, **kwargs):
    cube = np.array(frames)

    if headers is not None:
        headers = combine_frames_headers(headers, **kwargs)

    return cube, headers


def delta_angle(alpha, beta):
    alphar, betar = np.deg2rad(alpha), np.deg2rad(beta)
    dy = np.sin(alphar) - np.sin(betar)
    dx = np.cos(alphar) - np.cos(betar)
    return np.abs(np.rad2deg(np.arctan2(dy, dx)))


def combine_frames_headers(headers, wcs=False):
    output_header = fits.Header()
    # let's make this easier with tables
    table = pd.DataFrame([dict_from_header(header) for header in headers])
    # use a single header to get comments
    test_header = headers[0]
    # which columns have only a single unique value?
    unique_values = table.apply(lambda col: col.unique())
    unique_mask = unique_values.apply(lambda values: len(values) == 1)
    unique_row = table.loc[0, unique_mask]
    for key, val in unique_row.items():
        output_header[key] = val, test_header.comments[key]

    # sum exposure times
    output_header["TINT"] = table["TINT"].sum(), "[s] total integrated exposure time"
    # median PSF models
    if "MODEL" in table.keys():
        output_header["MOD_AMP"] = table["MOD_AMP"].mean(), "[adu] PSF model amplitude"
        output_header["MOD_X"] = table["MOD_X"].mean(), "[px] PSF model x"
        output_header["MOD_Y"] = table["MOD_Y"].mean(), "[px] PSF model y"
    if "PHOTFLUX" in table.keys():
        output_header["PHOTFLUX"] = table["PHOTFLUX"].mean(), "[adu] Aperture photometry flux"
    if "MEDFLUX" in table.keys():
        output_header["MEDFLUX"] = table["MEDFLUX"].mean(), "[adu] Median frame flux"
    if "SUMFLUX" in table.keys():
        output_header["SUMFLUX"] = table["SUMFLUX"].mean(), "[adu] Total frame flux"
    if "PEAKFLUX" in table.keys():
        output_header["PEAKFLUX"] = table["PEAKFLUX"].mean(), "[adu] Peak frame flux"
    # get PA rotation
    if "PA" in table.keys():
        output_header["PA-STR"] = table["PA-STR"].iloc[0], "[deg] par. angle at start"
        output_header["PA-END"] = table["PA-END"].iloc[-1], "[deg] par. angle at end"
        total_rot = delta_angle(output_header["PA-END"], output_header["PA-STR"])
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
            if k in output_header:
                del output_header[k]

    return output_header


def combine_frames_files(filenames, output, force=False, **kwargs):
    path = Path(output)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path

    frames = []
    headers = []
    for filename in filenames:
        # use memmap=False to avoid "too many files open" effects
        # another way would be to set ulimit -n <MAX_FILES>
        frame, header = fits.getdata(filename, header=True, memmap=False)
        frames.append(frame)
        headers.append(header)

    cube, header = combine_frames(frames, headers, **kwargs)

    fits.writeto(path, cube, header=header, overwrite=True)
    return path


def collapse_frames(frames, headers=None, method="median", **kwargs):
    cube, header = combine_frames(frames, headers=headers, **kwargs)
    return collapse_cube(cube, method=method, header=header)


def collapse_frames_files(filenames, output, force=False, **kwargs):
    path = Path(output)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path

    frames = []
    headers = []
    for filename in filenames:
        # use memmap=False to avoid "too many files open" effects
        # another way would be to set ulimit -n <MAX_FILES>
        frame, header = fits.getdata(filename, header=True, memmap=False)
        frames.append(frame)
        headers.append(header)

    frame, header = collapse_frames(frames, headers=headers, **kwargs)
    fits.writeto(path, frame, header=header, overwrite=True)
    return path


def correct_distortion(
    frame: ArrayLike,
    angle: float = 0,
    scale: float = 1,
    header=None,
    center=None,
    **kwargs,
):
    """
    Rotate and scale a single frame to match with a pinhole grid

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
    cube: ArrayLike,
    angle: float = 0,
    scale: float = 1,
    header=None,
    center=None,
    **kwargs,
):
    """
    Rotate and scale a all frames in a cube to match with a pinhole grid

    Parameters
    ----------
    cube : ArrayLike
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
        if scale < 1:
            frame = cv2.GaussianBlur(cube[i], (0, 0), sigmaX=0.5 / scale)
        else:
            frame = cube[i]
        corr_cube[i] = warp_frame(frame, M, **kwargs)
    # update header
    if header is not None:
        header["DPP_SCAL"] = scale, "scaling ratio for distortion correction"
        header["DPP_ANGL"] = angle, "deg, offset angle for distortion correction"
    return corr_cube, header


class FileSet:
    def __init__(self, filenames, ext=None) -> None:
        self.paths = {}
        self.headers = {}
        # get the headers
        for path in map(Path, filenames):
            if ext is None:
                ext = 1 if ".fits.fz" in path.name else 0
            with fits.open(path) as hdus:
                prim_hdu = hdus[ext]
                header = prim_hdu.header
            cam = header["U_CAMERA"]
            flc = header.get("U_FLCSTT", None)
            key = (cam, flc)
            self.paths[key] = path
            self.headers[key] = header

        # try and understand what we just got
        N = len(self.paths)
        if N not in (1, 2, 4):
            if N > 4:
                raise ValueError(f"Too many input files, should be 4 at max, got {N}")
            if N == 3:
                missing = set((1, 1), (1, 2), (2, 1), (2, 2)) - set(self.paths.keys())
                print(
                    f"Expected set of 4 files, one for each camera and FLC state combination. Missing camera {missing[0][0]} FLC state {missing[0][1]}"
                )

    @property
    def keys(self):
        return self.paths.keys()

    @property
    def cam1_paths(self):
        return [self.paths[k] for k in self.cam1_keys]

    @property
    def cam2_paths(self):
        return [self.paths[k] for k in self.cam2_keys]

    @property
    def cam1_keys(self):
        return filter(lambda k: k[0] == 1, self.keys)

    @property
    def cam2_keys(self):
        return filter(lambda k: k[0] == 2, self.keys)


def make_file_sets(filenames):
    # group files by identifiers that are unique to a simultaneous capture
    table = header_table(filenames)
    groups = table.groupby("MJD")
    return [FileSet(grp["path"]) for _, grp in groups]


def make_diff_image(cam1_file, cam2_file, outname=None, force=False):
    if outname is not None:
        outname = Path(outname)
    else:
        stem = re.sub("_cam[12]", "", cam1_file.stem)
        outname = cam1_file.with_name(f"{stem}_diff.fits")

    if not force and outname.is_file() and not any_file_newer((cam1_file, cam2_file), outname):
        return outname

    cam1_frame, header = fits.getdata(
        cam1_file,
        header=True,
    )
    cam2_frame, header2 = fits.getdata(
        cam2_file,
        header=True,
    )

    if header["MJD"] != header2["MJD"]:
        msg = f"{cam1_file.name} has MJD {header['MJD']}\n{cam2_file.name} has MJD {header2['MJD']}"
        raise ValueError(msg)
    if "U_FLCSTT" in header:
        if header["U_FLCSTT"] != header2["U_FLCSTT"]:
            msg = f"{cam1_file.name} has FLC state {header['U_FLCSTT']}\n{cam2_file.name} has FLC state {header2['U_FLCSTT']}"
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
