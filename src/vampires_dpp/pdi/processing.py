import itertools
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tqdm.auto as tqdm
from astropy.io import fits
from astropy.time import Time
from numpy.typing import NDArray

from vampires_dpp.image_processing import combine_frames_headers, derotate_frame

from ..util import any_file_newer
from .mueller_matrices import mueller_matrix_from_header
from .utils import instpol_correct, measure_instpol, write_stokes_products


def polarization_calibration_triplediff(
    filenames: Sequence[str],
    mm_filenames: Optional[Sequence[str]] = None,
    adi_sync=True,
    mm_correct=True,
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
    if len(filenames) % 16 != 0:
        raise ValueError(
            f"Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        )
    # now do triple-differential calibration
    frame_dict = {}
    for file in filenames:
        frame, hdr = fits.getdata(
            file,
            header=True,
        )
        # derotate frame - necessary for crosstalk correction
        frame_derot = derotate_frame(frame, hdr["DEROTANG"])
        # store into dictionaries
        key = hdr["U_HWPANG"], hdr["U_FLCSTT"], hdr["U_CAMERA"]
        frame_dict[key] = frame_derot

    I, Q, U = triple_diff_dict(frame_dict)

    stokes_cube = np.array((I, Q, U))

    headers = [fits.getheader(f) for f in filenames]
    stokes_hdr = combine_frames_headers(headers)
    # reduce exptime by 2 because cam1 and cam2 are simultaneous
    stokes_hdr["TINT"] /= 2

    return fits.PrimaryHDU(stokes_cube, stokes_hdr)


def make_triplediff_dict(filenames):
    output = {}
    for file in filenames:
        data, hdr = fits.getdata(
            file,
            header=True,
        )
        # get row vector for mueller matrix of each file
        # store into dictionaries
        key = hdr["U_HWPANG"], hdr["U_FLCSTT"], hdr["U_CAMERA"]
        output[key] = data
    return output


def mueller_matrix_correct(stokes_cube, filenames, header=None):
    for file in filenames:
        fits.getdata(file)


def triple_diff_dict(input_dict):
    ## make difference images
    # single diff (cams)
    pQ0 = input_dict[(0, 1, 1)] - input_dict[(0, 1, 2)]
    pIQ0 = input_dict[(0, 1, 1)] + input_dict[(0, 1, 2)]
    pQ1 = input_dict[(0, 2, 1)] - input_dict[(0, 2, 2)]
    pIQ1 = input_dict[(0, 2, 1)] + input_dict[(0, 2, 2)]
    # double difference (FLC1 - FLC2)
    pQ = 0.5 * (pQ0 - pQ1)
    pIQ = 0.5 * (pIQ0 + pIQ1)

    mQ0 = input_dict[(45, 1, 1)] - input_dict[(45, 1, 2)]
    mIQ0 = input_dict[(45, 1, 1)] + input_dict[(45, 1, 2)]
    mQ1 = input_dict[(45, 2, 1)] - input_dict[(45, 2, 2)]
    mIQ1 = input_dict[(45, 2, 1)] + input_dict[(45, 2, 2)]

    mQ = 0.5 * (mQ0 - mQ1)
    mIQ = 0.5 * (mIQ0 + mIQ1)

    pU0 = input_dict[(22.5, 1, 1)] - input_dict[(22.5, 1, 2)]
    pIU0 = input_dict[(22.5, 1, 1)] + input_dict[(22.5, 1, 2)]
    pU1 = input_dict[(22.5, 2, 1)] - input_dict[(22.5, 2, 2)]
    pIU1 = input_dict[(22.5, 2, 1)] + input_dict[(22.5, 2, 2)]

    pU = 0.5 * (pU0 - pU1)
    pIU = 0.5 * (pIU0 + pIU1)

    mU0 = input_dict[(67.5, 1, 1)] - input_dict[(67.5, 1, 2)]
    mIU0 = input_dict[(67.5, 1, 1)] + input_dict[(67.5, 1, 2)]
    mU1 = input_dict[(67.5, 2, 1)] - input_dict[(67.5, 2, 2)]
    mIU1 = input_dict[(67.5, 2, 1)] + input_dict[(67.5, 2, 2)]

    mU = 0.5 * (mU0 - mU1)
    mIU = 0.5 * (mIU0 + mIU1)

    # triple difference (HWP1 - HWP2)
    Q = 0.5 * (pQ - mQ)
    IQ = 0.5 * (pIQ + mIQ)
    U = 0.5 * (pU - mU)
    IU = 0.5 * (pIU + mIU)
    I = 0.5 * (IQ + IU)

    return I, Q, U


def mueller_matrix_calibration(mueller_matrices: NDArray, cube: NDArray) -> NDArray:
    stokes_cube = np.zeros_like(cube, shape=(mueller_matrices.shape[-1], *cube.shape[-2:]))
    # go pixel-by-pixel
    for i in range(cube.shape[-2]):
        for j in range(cube.shape[-1]):
            stokes_cube[:, i, j] = np.linalg.lstsq(mueller_matrices, cube[:, i, j], rcond=None)[0]

    return stokes_cube[:3]


def polarization_calibration_leastsq(filenames, mm_filenames, outname, force=False):
    path = Path(outname)
    if not force and path.is_file() and not any_file_newer(filenames, path):
        return path
    # step 1: load each file and header- derotate on the way
    frames = []
    headers = []
    mueller_mats = []
    for file, mm_file in tqdm.tqdm(
        zip(filenames, mm_filenames), total=len(filenames), desc="Least-squares calibration"
    ):
        frame, hdr = fits.getdata(file, header=True, memmap=False)
        # rotate to N up E left
        frame_derot = derotate_frame(frame, hdr["DEROTANG"])
        # get row vector for mueller matrix of each file
        mueller_mat = fits.getdata(mm_file, memmap=False)[0]

        frames.append(frame_derot)
        headers.append(hdr)
        mueller_mats.append(mueller_mat)

    mueller_mat_cube = np.array(mueller_mats)
    cube = np.array(frames)
    stokes_hdr = combine_frames_headers(headers, wcs=True)
    stokes_cube = mueller_matrix_calibration(mueller_mat_cube, cube)
    return write_stokes_products(stokes_cube, stokes_hdr, outname=path, force=True)


TRIPLEDIFF_SETS = set(itertools.product((0, 45, 22.5, 67.5), (1, 2), (1, 2)))


def get_triplediff_set(table, row):
    time_arr = Time(table["MJD"], format="mjd")
    row_time = Time(row["MJD"], format="mjd")
    deltatime = np.abs(row_time - time_arr)
    row_key = (row["U_HWPANG"], row["U_FLCSTT"], row["U_CAMERA"])
    remaining_keys = TRIPLEDIFF_SETS - set(row_key)
    output_set = {row_key: row["path"]}
    for key in remaining_keys:
        mask = (
            (table["U_HWPANG"] == key[0])
            & (table["U_FLCSTT"] == key[1])
            & (table["U_CAMERA"] == key[2])
        )
        idx = deltatime[mask].argmin()

        output_set[key] = table.loc[mask, "path"].iloc[idx]

    return output_set


def make_stokes_image(
    path_set,
    outpath: Path,
    mm_paths=None,
    method="triplediff",
    mm_correct=True,
    ip_correct=True,
    ip_radius=8,
    ip_method="photometry",
    force=False,
):
    if not force and outpath.exists() and not any_file_newer(path_set, outpath):
        return fits.getdata(outpath, header=True)

    # create stokes cube

    if method == "triplediff":
        stokes_hdu = polarization_calibration_triplediff(path_set)
        if mm_correct:
            mm_dict = make_triplediff_dict(mm_paths)
    else:
        raise ValueError(f"Unrecognized method {method}")

    stokes_data = stokes_hdu.data
    stokes_header = stokes_hdu.header
    I, Q, U = stokes_data
    # mm correct
    if mm_correct:
        _, mmQ, mmU = triple_diff_dict(mm_dict)

        # correct IP
        Q -= mmQ[0, 0] * I
        U -= mmU[0, 0] * I

        # correct cross-talk
        Sarr = np.array((Q.ravel(), U.ravel()))
        Marr = np.array((mmQ[0, 1:3], mmU[0, 1:3]))
        res = np.linalg.lstsq(Marr.T, Sarr, rcond=None)[0]
        Q = res[0].reshape(I.shape[-2:])
        U = res[1].reshape(I.shape[-2:])
        stokes_data = np.array((I, Q, U))
    # second order IP correction
    if ip_correct:
        stokes_data, stokes_header = polarization_ip_correct(
            stokes_data, phot_rad=ip_radius, header=stokes_header
        )

    fits.writeto(outpath, stokes_data, header=stokes_header, overwrite=True)
    return stokes_data, stokes_header


def polarization_ip_correct(stokes_data, phot_rad, header=None):
    cQ = measure_instpol(stokes_data[0], stokes_data[1], r=phot_rad)
    cU = measure_instpol(stokes_data[0], stokes_data[2], r=phot_rad)

    stokes_data[:3] = instpol_correct(stokes_data[:3], cQ, cU)

    if header is not None:
        header["IP_PHOTQ"] = cQ, "I -> Q IP correction value"
        header["IP_PHOTU"] = cU, "I -> U IP correction value"
    return stokes_data, header
