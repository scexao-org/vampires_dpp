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
    frame_radii,
    derotate_cube,
    derotate_frame,
    weighted_collapse,
    combine_frames_headers,
)
from .indexing import window_slices
from .mueller_matrices import mueller_matrix_model, mueller_matrix_triplediff
from .image_registration import offset_centroid
from .headers import observation_table
from .util import average_angle
from .wcs import apply_wcs


def measure_instpol(I: ArrayLike, X: ArrayLike, r=5, center=None, expected=0):
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
    expected : float, optional
        The expected fractional polarization, by default 0

    Returns
    -------
    float
        The instrumental polarization coefficient
    """
    if center is None:
        center = frame_center(I)

    x = X / I

    rs = frame_radii(x)

    weights = np.sqrt(np.abs(I))
    # only keep values inside aperture
    weights[rs > r] = 0

    pX = np.nansum(x * weights) / np.nansum(weights)
    return pX - expected


def measure_instpol_satellite_spots(I: ArrayLike, X: ArrayLike, r=5, expected=0, **kwargs):
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
    expected : float, optional
        The expected fractional polarization, by default 0

    Returns
    -------
    float
        The instrumental polarization coefficient
    """
    x = X / I

    slices = window_slices(x, **kwargs)
    # refine satellite spot apertures onto centroids
    aps_centers = [offset_centroid(I, sl) for sl in slices]

    # TODO may be biased by central halo?
    # measure IP from aperture photometry
    aps = CircularAperture(aps_centers, r)
    fluxes = aperture_photometry(x, aps)["aperture_sum"]
    cX = np.mean(fluxes) / aps.area

    return cX - expected


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


def radial_stokes(stokes_cube: ArrayLike, phi: Optional[float] = None, **kwargs) -> NDArray:
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
        phi = optimize_Uphi(stokes_cube, thetas, **kwargs)

    cos2t = np.cos(2 * (thetas + phi))
    sin2t = np.sin(2 * (thetas + phi))
    Qphi = -stokes_cube[1] * cos2t - stokes_cube[2] * sin2t
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t

    return Qphi, Uphi


def optimize_Uphi(stokes_cube: ArrayLike, thetas: ArrayLike, r=8) -> float:
    cy, cx = frame_center(stokes_cube)
    rs = frame_radii(stokes_cube)
    mask = rs <= r
    masked_stokes_cube = stokes_cube[..., mask]
    masked_thetas = thetas[..., mask]

    loss = lambda X: Uphi_loss(X, masked_stokes_cube, masked_thetas, r=r)
    res = minimize_scalar(loss, bounds=(-np.pi / 2, np.pi / 2), method="bounded")
    return res.x


def Uphi_loss(X: float, stokes_cube: ArrayLike, thetas: ArrayLike, r) -> float:
    cos2t = np.cos(2 * (thetas + X))
    sin2t = np.sin(2 * (thetas + X))
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t
    l2norm = np.nansum(Uphi**2)
    return l2norm


def collapse_stokes_cube(stokes_cube, pa, header=None):
    stokes_out = np.empty_like(stokes_cube, shape=(stokes_cube.shape[0], *stokes_cube.shape[-2:]))
    for s in range(stokes_cube.shape[0]):
        derot = derotate_cube(stokes_cube[s], pa)
        stokes_out[s] = np.nanmedian(derot, axis=0)

    # now that cube is derotated we can apply WCS
    if header is not None:
        apply_wcs(header, pupil_offset=None)

    return stokes_out, header


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
    tbl = observation_table(filenames)

    # once more check again that we have proper HWP sets
    hwpangs = tbl["U_HWPANG"].values.reshape((-1, 4, 4)).mean(axis=(0, 2))
    if hwpangs[0] != 0 or hwpangs[1] != 45 or hwpangs[2] != 22.5 or hwpangs[3] != 67.5:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )

    # now do triple-differential calibration
    # only load 16 files at a time to avoid running out of memory on large datasets
    N_hwp_sets = len(filenames) // 16
    with fits.open(filenames.iloc[0]) as hdus:
        stokes_cube = np.zeros(shape=(4, N_hwp_sets, *hdus[0].shape), dtype=hdus[0].data.dtype)
    angles = triplediff_average_angles(filenames)
    iter = tqdm.trange(N_hwp_sets, desc="Triple-differential calibration")
    for i in iter:
        # prepare input frames
        ix = i * 16  # offset index
        frame_dict = {}
        for file in filenames.iloc[ix : ix + 16]:
            frame, hdr = fits.getdata(file, header=True)
            key = hdr["U_HWPANG"], hdr["U_FLCSTT"], hdr["U_CAMERA"]
            frame_dict[key] = frame
        # make difference images
        for hwp_ang in (0.0, 45.0, 22.5, 67.5):
            # single diff: cam1 - cam2
            pX = frame_dict[(hwp_ang, 1, 1)] - frame_dict[(hwp_ang, 1, 2)]
            mX = frame_dict[(hwp_ang, 2, 1)] - frame_dict[(hwp_ang, 2, 2)]
            # double diff: flc1 - flc2
            X = 0.5 * (pX - mX)
            # triple diff: hwp1 - hwp2
            if hwp_ang == 0.0:  # Q
                stokes_cube[1, i] += 0.5 * X
            elif hwp_ang == 45.0:  # -Q
                stokes_cube[1, i] -= 0.5 * X
            elif hwp_ang == 22.5:  # U
                stokes_cube[2, i] += 0.5 * X
            elif hwp_ang == 67.5:  # -U
                stokes_cube[2, i] -= 0.5 * X
        # factor of 2 because intensity is cut in half by beamsplitter
        stokes_cube[0, i] = 2 * np.mean(list(frame_dict.values()), axis=0)

    headers = [fits.getheader(f) for f in filenames]
    stokes_hdr = combine_frames_headers(headers)

    return stokes_cube, stokes_hdr, angles


def triplediff_average_angles(filenames):
    if len(filenames) % 16 != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )
    # make sure we get data in correct order using FITS headers
    tbl = observation_table(filenames).sort_values(["DATE", "U_PLSTIT", "U_FLCSTT", "U_CAMERA"])

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
        raise ValueError(f'\'method\' must be either "model" or "triplediff" (got {method})')

    hdu = fits.PrimaryHDU(mueller_mats)
    hdu.header["METHOD"] = method
    hdu.writeto(output, overwrite=True)

    return output


def mueller_matrix_calibration_files(filenames, mueller_matrix_file=None, output=None, skip=False):
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
            stokes_cube[:, i, j] = np.linalg.lstsq(mueller_matrices, cube[:, i, j], rcond=None)[0]

    return stokes_cube


def write_stokes_products(stokes_cube, header=None, outname=None, skip=False, phi=None):
    if outname is None:
        path = Path("stokes_cube.fits")
    else:
        path = Path(outname)

    if skip and path.is_file():
        return path

    pi = np.hypot(stokes_cube[2], stokes_cube[1])
    aolp = np.arctan2(stokes_cube[2], stokes_cube[1])
    Qphi, Uphi = radial_stokes(stokes_cube, phi=phi)

    if header is None:
        header = fits.Header()

    header["STOKES"] = "I,Q,U,Qphi,Uphi,LP_I,AoLP"
    if phi is not None:
        header["VPP_PHI"] = phi, "deg, angle of linear polarization offset"

    data = np.asarray((stokes_cube[0], stokes_cube[1], stokes_cube[2], Qphi, Uphi, pi, aolp))

    fits.writeto(path, data, header=header, overwrite=True)

    return path
