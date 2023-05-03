from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import tqdm.auto as tqdm
from astropy.io import fits
from numpy.typing import ArrayLike, NDArray
from photutils import aperture_photometry
from scipy.optimize import minimize_scalar

from vampires_dpp.constants import PUPIL_OFFSET
from vampires_dpp.image_processing import combine_frames_headers, derotate_cube
from vampires_dpp.image_registration import offset_centroid
from vampires_dpp.indexing import cutout_slice, frame_angles, frame_radii, window_slices
from vampires_dpp.mueller_matrices import mueller_matrix_model
from vampires_dpp.util import any_file_newer, average_angle
from vampires_dpp.wcs import apply_wcs

HWP_POS_STOKES = {0: "Q", 45: "-Q", 22.5: "U", 67.5: "-U"}


def measure_instpol(I: ArrayLike, X: ArrayLike, r=5, expected=0, window=30, **kwargs):
    """
    Use aperture photometry to estimate the instrument polarization.

    Parameters
    ----------
    stokes_cube : ArrayLike
        Input Stokes cube (4, y, x)
    r : float, optional
        Radius of circular aperture in pixels, by default 5
    expected : float, optional
        The expected fractional polarization, by default 0
    **kwargs

    Returns
    -------
    float
        The instrumental polarization coefficient
    """
    x = X / I
    sl = cutout_slice(x, window=window, **kwargs)
    cutout = x[sl[0], sl[1]]
    pX = safe_aperture_sum(cutout, r=r)
    return pX - expected


def measure_instpol_satellite_spots(
    I: ArrayLike, X: ArrayLike, r=5, expected=0, window=30, **kwargs
):
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

    slices = window_slices(x, window=window, **kwargs)
    # refine satellite spot apertures onto centroids
    aps_centers = [offset_centroid(I, sl) for sl in slices]

    # TODO may be biased by central halo?
    # measure IP from aperture photometry
    fluxes = []
    for ap_center in aps_centers:
        # use refined window
        sl = cutout_slice(x, window=window, center=ap_center)
        cutout = x[sl[0], sl[1]]
        fluxes.append(safe_aperture_sum(cutout, r=r))
    return np.mean(fluxes) - expected


def safe_aperture_sum(frame, r):
    weights = np.ones_like(frame)
    rs = frame_radii(frame)
    # only keep values inside aperture
    weights[rs > r] = 0
    return np.nansum(frame * weights) / np.nansum(weights)


def instpol_correct(stokes_cube: ArrayLike, pQ=0, pU=0):
    """
    Apply instrument polarization correction to stokes cube.

    Parameters
    ----------
    stokes_cube : ArrayLike
        (3, ...) array of stokes values
    pQ : float, optional
        I -> Q contribution, by default 0
    pU : float, optional
        I -> U contribution, by default 0

    Returns
    -------
    NDArray
        (3, ...) stokes cube with corrected parameters
    """
    return np.array(
        (
            stokes_cube[0],
            stokes_cube[1] - pQ * stokes_cube[0],
            stokes_cube[2] - pU * stokes_cube[0],
        )
    )


def background_subtracted_photometry(frame, aps, anns):
    ap_sums = aperture_photometry(frame, aps)["aperture_sum"]
    ann_sums = aperture_photometry(frame, anns)["aperture_sum"]
    return ap_sums - aps.area / anns.area * ann_sums


def radial_stokes(stokes_cube: ArrayLike, phi: Optional[float] = None, **kwargs) -> NDArray:
    r"""
    Calculate the radial Stokes parameters from the given Stokes cube (4, N, M)

    .. math::
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
    thetas = frame_angles(stokes_cube, conv="astro")
    if phi is None:
        phi = optimize_Uphi(stokes_cube, thetas, **kwargs)

    cos2t = np.cos(2 * (thetas + phi))
    sin2t = np.sin(2 * (thetas + phi))
    Qphi = -stokes_cube[1] * cos2t - stokes_cube[2] * sin2t
    Uphi = stokes_cube[1] * sin2t - stokes_cube[2] * cos2t

    return Qphi, Uphi


def optimize_Uphi(stokes_cube: ArrayLike, thetas: ArrayLike, r=8) -> float:
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


def rotate_stokes(stokes_cube, theta):
    out = stokes_cube.copy()
    sin2ts = np.sin(2 * theta)
    cos2ts = np.cos(2 * theta)
    out[1] = stokes_cube[1] * cos2ts - stokes_cube[2] * sin2ts
    out[2] = stokes_cube[1] * sin2ts + stokes_cube[2] * cos2ts
    return out


def collapse_stokes_cube(stokes_cube, pa, adi_sync=True, header=None):
    stokes_out = np.empty_like(stokes_cube, shape=(stokes_cube.shape[0], *stokes_cube.shape[-2:]))
    # derotate stokes vectors
    stokes_cube_derot = np.empty_like(stokes_cube)
    for i in range(stokes_cube.shape[1]):
        derot_angle = PUPIL_OFFSET
        if not adi_sync:
            derot_angle += pa[i]
        stokes_cube_derot[:, i] = rotate_stokes(stokes_cube[:, i], np.deg2rad(derot_angle))

    for s in range(stokes_cube_derot.shape[0]):
        derot = derotate_cube(stokes_cube_derot[s], pa)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stokes_out[s] = np.nanmedian(derot, axis=0, overwrite_input=True)
    # now that cube is derotated we can apply WCS
    if header is not None:
        apply_wcs(header)

    return stokes_out, header


def polarization_calibration_triplediff(
    filenames: Sequence[str], outname, force=False, N_per_hwp=1
) -> NDArray:
    """
    Return a Stokes cube using the *bona fide* triple differential method. This method will split the input data into sets of 16 frames- 2 for each camera, 2 for each FLC state, and 4 for each HWP angle.

    .. admonition:: Pupil-tracking mode
        :class: tip

        For each of these 16 image sets, it is important to consider the apparant sky rotation when in pupil-tracking mode (which is the default for most VAMPIRES observations). With this naive triple-differential subtraction, if there is significant sky motion, the output Stokes frame will be smeared.

        The parallactic angles for each set of 16 frames should be averaged (``average_angle``) and stored to construct the final derotation angle vector

    Parameters
    ----------
    filenames : Sequence[str]
        list of input filenames to construct Stokes frames from

    Raises
    ------
    ValueError:
        If the input filenames are not a clean multiple of 16. To ensure you have proper 16 frame sets, use ``pol_inds`` with a sorted observation table.

    Returns
    -------
    NDArray
        (4, t, y, x) Stokes cube from all 16 frame sets.
    """
    if len(filenames) % 16 * N_per_hwp != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of {16 * N_per_hwp} frames for each HWP cycle"
        )
    # now do triple-differential calibration
    # only load 8 files at a time to avoid running out of memory on large datasets
    N_hwp_sets = len(filenames) // 16
    with fits.open(filenames.iloc[0]) as hdus:
        stokes_cube = np.zeros(shape=(4, N_hwp_sets, *hdus[0].shape[-2:]), dtype=hdus[0].data.dtype)
    iter = tqdm.trange(N_hwp_sets, desc="Triple-differential calibration")
    for i in iter:
        # prepare input frames
        ix = i * 16 * N_per_hwp  # offset index
        frame_dict = {}
        for file in filenames.iloc[ix : ix + 16 * N_per_hwp]:
            frame, hdr = fits.getdata(
                file,
                header=True,
            )
            key = hdr["U_HWPANG"], hdr["U_FLCSTT"], hdr["U_CAMERA"]
            frame_dict[key] = frame

        ## make difference images
        # single diff (cams)
        pQ0 = frame_dict[(0, 1, 1)] - frame_dict[(0, 1, 2)]
        pIQ0 = frame_dict[(0, 1, 1)] + frame_dict[(0, 1, 2)]
        pQ1 = frame_dict[(0, 2, 1)] - frame_dict[(0, 2, 2)]
        pIQ1 = frame_dict[(0, 2, 1)] + frame_dict[(0, 2, 2)]
        # double difference (FLC1 - FLC2)
        pQ = 0.5 * (pQ0 - pQ1)
        pIQ = 0.5 * (pIQ0 + pIQ1)

        mQ0 = frame_dict[(45, 1, 1)] - frame_dict[(45, 1, 2)]
        mIQ0 = frame_dict[(45, 1, 1)] + frame_dict[(45, 1, 2)]
        mQ1 = frame_dict[(45, 2, 1)] - frame_dict[(45, 2, 2)]
        mIQ1 = frame_dict[(45, 2, 1)] + frame_dict[(45, 2, 2)]

        mQ = 0.5 * (mQ0 - mQ1)
        mIQ = 0.5 * (mIQ0 + mIQ1)

        pU0 = frame_dict[(22.5, 1, 1)] - frame_dict[(22.5, 1, 2)]
        pIU0 = frame_dict[(22.5, 1, 1)] + frame_dict[(22.5, 1, 2)]
        pU1 = frame_dict[(22.5, 2, 1)] - frame_dict[(22.5, 2, 2)]
        pIU1 = frame_dict[(22.5, 2, 1)] + frame_dict[(22.5, 2, 2)]

        pU = 0.5 * (pU0 - pU1)
        pIU = 0.5 * (pIU0 + pIU1)

        mU0 = frame_dict[(67.5, 1, 1)] - frame_dict[(67.5, 1, 2)]
        mIU0 = frame_dict[(67.5, 1, 1)] + frame_dict[(67.5, 1, 2)]
        mU1 = frame_dict[(67.5, 2, 1)] - frame_dict[(67.5, 2, 2)]
        mIU1 = frame_dict[(67.5, 2, 1)] + frame_dict[(67.5, 2, 2)]

        mU = 0.5 * (mU0 - mU1)
        mIU = 0.5 * (mIU0 + mIU1)

        # triple difference (HWP1 - HWP2)
        Q = 0.5 * (pQ - mQ)
        IQ = 0.5 * (pIQ + mIQ)
        U = 0.5 * (pU - mU)
        IU = 0.5 * (pIU + mIU)
        I = 0.5 * (IQ + IU)

        stokes_cube[0, i : i + N_per_hwp] = I
        stokes_cube[1, i : i + N_per_hwp] = Q
        stokes_cube[2, i : i + N_per_hwp] = U

    headers = [fits.getheader(f) for f in filenames]
    stokes_hdr = combine_frames_headers(headers)

    return write_stokes_products(stokes_cube, stokes_hdr, outname=outname, force=force)


def triplediff_average_angles(filenames):
    if len(filenames) % 16 != 0:
        raise ValueError(
            "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )
    # make sure we get data in correct order using FITS headers
    derot_angles = np.asarray([fits.getval(f, "PARANG") for f in filenames])
    N_hwp_sets = len(filenames) // 16
    pas = np.zeros(N_hwp_sets, dtype="f4")
    for i in range(pas.shape[0]):
        ix = i * 16
        pas[i] = average_angle(derot_angles[ix : ix + 16])

    return pas


def pol_inds(hwp_angs: ArrayLike, n=4, order="QQUU"):
    """
    Find consistent runs of FLC and HWP states.

    A consistent run will have either 2 or 4 files per HWP state, and will have exactly 4 HWP states per cycle. Sometimes when VAMPIRES is syncing with CHARIS a HWP state will get skipped, creating partial HWP cycles. This function will return the indices which create consistent HWP cycles from the given list of FLC states, which should already be sorted by time.

    Parameters
    ----------
    hwp_angs : ArrayLike
        The HWP states to sort through
    n : int, optional
        The number of files per HWP state, either 2 or 4. By default 4

    Returns
    -------
    inds :
        The indices for which `hwp_angs` forms consistent HWP cycles
    """
    states = np.asarray(hwp_angs)
    N_cycle = n * 4
    if order == "QQUU":
        ang_list = np.repeat([0, 45, 22.5, 67.5], n)
    elif order == "QUQU":
        ang_list = np.repeat([0, 22.5, 45, 67.5], n)
    inds = []
    idx = 0
    while idx <= len(hwp_angs) - N_cycle:
        if np.all(states[idx : idx + N_cycle] == ang_list):
            inds.extend(range(idx, idx + N_cycle))
            idx += N_cycle
        else:
            idx += 1

    return inds


def polarization_calibration_model(filename):
    header = fits.getheader(filename)
    pa = np.deg2rad(header["PARANG"])
    altitude = np.deg2rad(header["ALTITUDE"])
    hwp_theta = np.deg2rad(header["U_HWPANG"])
    imr_theta = np.deg2rad(header["D_IMRANG"])
    # qwp are oriented with 0 on vertical axis
    qwp1 = np.pi / 2 - np.deg2rad(header["U_QWP1"])
    qwp2 = np.pi / 2 - np.deg2rad(header["U_QWP2"])

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
    return M


def mueller_mats_file(filename, output=None, force: bool = False):
    if output is None:
        indir = Path(filename).parent
        output = indir / f"mueller_mats.fits"
    else:
        output = Path(output)

    if not force and output.is_file() and Path(filename).stat().st_mtime < output.stat().st_mtime:
        return output

    mueller_mat = polarization_calibration_model(filename)

    hdu = fits.PrimaryHDU(mueller_mat)
    hdu.header["INPUT"] = filename.absolute(), "FITS diff frame"
    hdu.writeto(
        output,
        overwrite=True,
    )

    return output


def mueller_matrix_calibration(mueller_matrices: ArrayLike, cube: ArrayLike) -> NDArray:
    stokes_cube = np.zeros((mueller_matrices.shape[-1], cube.shape[-2], cube.shape[-1]))
    # go pixel-by-pixel
    for i in range(cube.shape[-2]):
        for j in range(cube.shape[-1]):
            stokes_cube[:, i, j] = np.linalg.lstsq(mueller_matrices, cube[:, i, j], rcond=None)[0]

    return stokes_cube


def write_stokes_products(stokes_cube, header=None, outname=None, force=False, phi=0):
    if outname is None:
        path = Path("stokes_cube.fits")
    else:
        path = Path(outname)

    if not force and path.is_file():
        return path

    pi = np.hypot(stokes_cube[2], stokes_cube[1])
    aolp = np.arctan2(stokes_cube[2], stokes_cube[1])
    Qphi, Uphi = radial_stokes(stokes_cube, phi=phi)

    if header is None:
        header = fits.Header()

    header["STOKES"] = "I,Q,U,Qphi,Uphi,LP_I,AoLP"
    if phi is not None:
        header["DPP_PHI"] = phi, "deg, angle of linear polarization offset"

    data = np.asarray((stokes_cube[0], stokes_cube[1], stokes_cube[2], Qphi, Uphi, pi, aolp))

    fits.writeto(path, data, header=header, overwrite=True)

    return path
