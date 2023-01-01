from astropy.io import fits
from scipy.optimize import minimize_scalar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple, Sequence
from pathlib import Path
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import tqdm.auto as tqdm
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs

from .constants import PUPIL_OFFSET
from .image_processing import (
    frame_angles,
    frame_center,
    derotate_frame,
    weighted_collapse,
    combine_frames_headers,
)
from .indexing import window_slices
from .mueller_matrices import mueller_matrix_model, mueller_matrix_triplediff
from .image_registration import offset_centroid
from .headers import observation_table
from .util import average_angle


def measure_instpol(I: ArrayLike, X: ArrayLike, r=5, center=None) -> Tuple:
    """
    Use aperture photometry to estimate the instrument polarization.

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
    Tuple
        (cQ, cU) tuple of instrument polarization coefficients
    """
    if center is None:
        center = frame_center(I)

    x = X / I

    ap = CircularAperture((center[1], center[0]), r)

    cX = aperture_photometry(x, ap)["aperture_sum"][0] / ap.area

    return cX


def measure_instpol_satellite_spots(stokes_cube: ArrayLike, r=5, **kwargs) -> Tuple:
    """
    Use aperture photometry on satellite spots to estimate the instrument polarization.

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
    Tuple
        (cQ, cU) tuple of instrument polarization coefficients
    """
    q = stokes_cube[1] / stokes_cube[0]
    u = stokes_cube[2] / stokes_cube[0]

    slices = window_slices(stokes_cube[0], **kwargs)
    # refine satellite spot apertures onto centroids
    # TODO may be biased by central halo?
    aps_centers = [offset_centroid(stokes_cube[0], sl) for sl in slices]

    # do background-subtracted photometry
    aps = CircularAperture(aps_centers, r)
    anns = CircularAnnulus(aps_centers, 2 * r, 3 * r)

    cQ = np.mean(background_subtracted_photometry(q, aps, anns)) / aps.area
    cU = np.mean(background_subtracted_photometry(u, aps, anns)) / aps.area

    return cQ, cU


def instpol_correct(stokes_cube: ArrayLike, cQ=0, cU=0, cV=0):
    """
    Apply instrument polarization correction to stokes cube.

    Parameters
    ----------
    stokes_cube : ArrayLike
        (4, ...) array of stokes values
    cQ : float, optional
        I -> Q contribution, by default 0
    cU : float, optional
        I -> U contribution, by default 0
    cV : float, optional
        I -> V contribution, by default 0

    Returns
    -------
    NDArray
        (4, ...) stokes cube with corrected parameters
    """
    return np.array(
        (
            stokes_cube[0],
            stokes_cube[1] - cQ * stokes_cube[0],
            stokes_cube[2] - cU * stokes_cube[0],
            stokes_cube[3] - cV * stokes_cube[0],
        )
    )


def background_subtracted_photometry(frame, aps, anns):
    ap_sums = aperture_photometry(frame, aps)["aperture_sum"]
    ann_sums = aperture_photometry(frame, anns)["aperture_sum"]
    return ap_sums - aps.area / anns.area * ann_sums


def radial_stokes(stokes_cube: ArrayLike, phi: Optional[float] = None) -> NDArray:
    r"""
    Calculate the radial Stokes parameters from the given Stokes cube (4, N, M)

    ..math::
        Q_\phi = -Q\cos(2\theta) - U\sin(2\theta) \\
        U_\phi = Q\sin(2\theta) - Q\cos(2\theta)
        

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


def rotate_stokes(stokes_cube, angles):
    out = stokes_cube.copy()
    thetas = np.deg2rad(angles)[None, :, None, None]
    sin2ts = np.sin(2 * thetas)
    cos2ts = np.cos(2 * thetas)
    out[1] = stokes_cube[1] * cos2ts - stokes_cube[2] * sin2ts
    out[2] = stokes_cube[1] * sin2ts + stokes_cube[2] * cos2ts
    return out


def collapse_stokes_cube(stokes_cube, pa):
    out = np.empty(
        (stokes_cube.shape[0], stokes_cube.shape[-2], stokes_cube.shape[-1]),
        stokes_cube.dtype,
    )
    # derotate polarization frame
    # stokes_cube_derot = rotate_stokes(stokes_cube, pa)
    for s in range(stokes_cube.shape[0]):
        out[s] = weighted_collapse(stokes_cube[s], pa)

    return out


def polarization_calibration_triplediff_naive(filenames: Sequence[str]) -> NDArray:
    """
    Return a Stokes cube using the _bona fide_ triple differential method. This method will split the input data into sets of 16 frames- 2 for each camera, 2 for each FLC state, and 4 for each HWP angle.

    .. admonition:: Pupil-tracking mode
        :class: warning
        For each of these 16 image sets, it is important to consider the apparant sky rotation when in pupil-tracking mode (which is the default for most VAMPIRES observations). With this naive triple-differential subtraction, if there is significant sky motion, the output Stokes frame will be smeared.

        The parallactic angles for each set of 16 frames should be averaged (``average_angle``) and stored to construct the final derotation angle vector

    Parameters
    ----------
    filenames : Sequence[str]
        List of input filenames to construct Stokes frames from

    Raises
    ------
    ValueError:
        If the input filenames are not a clean multiple of 16. To ensure you have proper 16 frame sets, use ``flc_inds`` with a sorted observation table.

    Returns
    -------
    NDArray
        (4, t, y, x) Stokes cube from all 16 frame sets.
    """
    if len(filenames) % 16 != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )

    # make sure we get data in correct order using FITS headers
    # tbl = observation_table(filenames).sort_values(
    #     ["DATE", "U_PLSTIT", "U_FLCSTT", "U_CAMERA"]
    # )

    # once more check again that we have proper HWP sets
    # hwpangs = tbl["U_HWPANG"].values.reshape((-1, 4, 4)).mean(axis=(0, 2))
    # print(hwpangs)
    # if hwpangs[0] != 0 or hwpangs[1] != 45 or hwpangs[2] != 22.5 or hwpangs[3] != 67.5:
    #     raise ValueError(
    #         "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
    #     )

    # now do triple-differential calibration

    N_hwp_sets = len(filenames) // 16
    with fits.open(filenames.iloc[0]) as hdus:
        stokes_cube = np.zeros(shape=(4, N_hwp_sets, *hdus[0].shape), dtype="f4")
    angles = triplediff_average_angles(filenames)
    iter = tqdm.trange(N_hwp_sets, desc="Triple-differential calibration")
    for i in iter:
        ix = i * 16  # offset index
        cube = np.asarray([fits.getdata(f) for f in filenames.iloc[ix : ix + 16]])
        # (cam1 - cam2) - (cam1 - cam2)
        pQ0 = cube[0] - cube[2]
        mQ0 = cube[1] - cube[3]
        Q0 = 0.5 * (pQ0 - mQ0)

        pQ1 = cube[4] - cube[6]
        mQ1 = cube[5] - cube[7]
        Q1 = 0.5 * (pQ1 - mQ1)

        # (cam1 - cam2) - (cam1 - cam2)
        pU0 = cube[8] - cube[10]
        mU0 = cube[9] - cube[11]
        U0 = 0.5 * (pU0 - mU0)

        pU1 = cube[12] - cube[14]
        mU1 = cube[13] - cube[15]
        U1 = 0.5 * (pU1 - mU1)

        # factor of 2 because intensity is cut in half by beamsplitter
        stokes_cube[0, i] = 2 * np.mean(cube, axis=0)
        # Q = 0.5 * (Q+ - Q-)
        stokes_cube[1, i] = 0.5 * (Q0 - Q1)
        # U = 0.5 * (U+ - U-)
        stokes_cube[2, i] = 0.5 * (U0 - U1)

    return collapse_stokes_cube(stokes_cube, angles)


def triplediff_average_angles(filenames):
    if len(filenames) % 16 != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )
    # make sure we get data in correct order using FITS headers
    tbl = observation_table(filenames).sort_values(
        ["DATE", "U_PLSTIT", "U_FLCSTT", "U_CAMERA"]
    )

    # once more check again that we have proper HWP sets
    hwpangs = tbl["U_HWPANG"].values.reshape((-1, 4, 4)).mean(axis=(0, 2))
    if hwpangs[0] != 0 or hwpangs[1] != 45 or hwpangs[2] != 22.5 or hwpangs[3] != 67.5:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )

    N_hwp_sets = len(tbl) // 16
    pas = np.zeros(N_hwp_sets, dtype="f4")
    for i in range(pas.shape[0]):
        ix = i * 16
        pas[i] = average_angle(tbl["D_IMRPAD"].iloc[ix : ix + 16] + PUPIL_OFFSET)

    return pas


def polarization_calibration_triplediff(filenames: Sequence[str]) -> NDArray:
    headers = [fits.getheader(f) for f in filenames]
    mueller_mats = np.empty((len(headers), 4), dtype="f4")
    for i in range(mueller_mats.shape[0]):
        header = headers[i]
        derot_angle = np.deg2rad(header["D_IMRPAD"] + PUPIL_OFFSET)
        hwp_theta = np.deg2rad(header["U_HWPANG"])

        M = mueller_matrix_triplediff(
            camera=header["U_CAMERA"],
            flc_state=header["U_FLCSTT"],
            theta=derot_angle,
            hwp_theta=hwp_theta,
        )
        # only keep the X -> I terms
        mueller_mats[i] = M[0]

    return mueller_mats


def polarization_calibration_model(filenames):
    headers = [fits.getheader(f) for f in filenames]
    mueller_mats = np.empty((len(headers), 4), dtype="f4")
    for i in range(mueller_mats.shape[0]):
        header = headers[i]
        pa = np.deg2rad(header["D_IMRPAD"] + header["LONPOLE"] - header["D_IMRPAP"])
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
            pupil_offset=np.deg2rad(PUPIL_OFFSET),
        )
        # only keep the X -> I terms
        mueller_mats[i] = M[0]

    return mueller_mats


def mueller_mats_files(filenames, method="mueller", output=None, skip=False):
    if output is None:
        indir = Path(filenames[0]).parent
        output = indir / f"mueller_mats.fits"
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    elif method == "mueller":
        mueller_mats = polarization_calibration_model(filenames)
    elif method == "triplediff":
        mueller_mats = polarization_calibration_triplediff(filenames)
    else:
        raise ValueError(
            f'\'method\' must be either "model" or "triplediff" (got {method})'
        )

    hdu = fits.PrimaryHDU(mueller_mats)
    hdu.header["METHOD"] = method
    hdu.writeto(output, overwrite=True)

    return output


def mueller_matrix_calibration_files(
    filenames, mueller_matrix_file=None, output=None, skip=False
):
    if output is None:
        indir = Path(filenames[0]).parent
        output = indir / f"stokes_cube.fits"
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    if mueller_matrix_file is None:
        mueller_matrix_file = mueller_mats_files(filenames)

    mueller_mats, muller_mat_hdr = fits.getdata(mueller_matrix_file, header=True)

    cubes = []
    headers = []
    for f in filenames:
        with fits.open(f) as hdus:
            cubes.append(hdus[0].data.copy())
            headers.append(hdus[0].header)
            # have to manually close mmap file descriptor
            # https://docs.astropy.org/en/stable/io/fits/index.html#working-with-large-files
            del hdus[0].data
    cube = np.asarray(cubes)
    stokes_cube = mueller_matrix_calibration(mueller_mats, cube)
    header = combine_frames_headers(headers, wcs=True)

    # add in stokes WCS information
    wcs = WCS(header)
    wcs = add_stokes_axis_to_wcs(wcs, 2)

    header.update(wcs.to_header())
    header["VPP_PDI"] = (
        muller_mat_hdr["METHOD"],
        "VAMPIRES DPP polarization calibration method",
    )

    fits.writeto(output, stokes_cube, header=header, overwrite=True)
    return output


# def mueller_matrix_calibration(mueller_matrices: ArrayLike, cube: ArrayLike) -> NDArray:
#     stokes_cube = np.empty((4, cube.shape[0], cube.shape[1], cube.shape[2]))
#     # for each frame, compute stokes frames
#     for i in range(cube.shape[0]):
#         M = mueller_matrices[i]
#         frame = cube[i].ravel()
#         S = np.linalg.lstsq(np.atleast_2d(M), np.atleast_2d(frame), rcond=None)[0]
#         stokes_cube[:, i] = S.reshape((4, cube.shape[1], cube.shape[2]))
#     return stokes_cube


def mueller_matrix_calibration(mueller_matrices: ArrayLike, cube: ArrayLike) -> NDArray:
    stokes_cube = np.zeros((mueller_matrices.shape[-1], cube.shape[-2], cube.shape[-1]))
    # go pixel-by-pixel
    for i in range(cube.shape[-2]):
        for j in range(cube.shape[-1]):
            stokes_cube[:, i, j] = np.linalg.lstsq(
                mueller_matrices, cube[:, i, j], rcond=None
            )[0]

    return stokes_cube
