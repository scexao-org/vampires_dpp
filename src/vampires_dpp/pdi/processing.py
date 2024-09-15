import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import bottleneck as bn
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from astropy.io import fits
from numpy.typing import NDArray

from vampires_dpp.combine_frames import combine_frames_headers
from vampires_dpp.headers import sort_header
from vampires_dpp.image_processing import derotate_cube, derotate_frame
from vampires_dpp.paths import any_file_newer
from vampires_dpp.util import create_or_append, load_fits
from vampires_dpp.wcs import apply_wcs

from .utils import (
    calculate_pol_efficiency,
    measure_instpol,
    measure_instpol_ann,
    rotate_stokes,
    write_stokes_products,
)


def polarization_calibration_triplediff(filenames: Sequence[str], derotate: bool = True):
    """Return a Stokes cube using the *bona fide* triple differential method. This method will split the input data into sets of 16 frames- 2 for each camera, 2 for each FLC state, and 4 for each HWP angle.

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
        msg = "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        raise ValueError(msg)
    # now do triple-differential calibration
    cube_dict = {}
    header_dict = {}
    cube_errs = {}
    headers: dict[str, fits.Header] = {}
    for file in filenames:
        with fits.open(file) as hdul:
            cube = hdul[0].data
            cube_err = hdul["ERR"].data
            prim_hdr = hdul[0].header
            create_or_append(headers, "PRIMARY", prim_hdr)
            for hdu in hdul[2:]:
                hdr = apply_wcs(cube, hdu.header, angle=0)
                create_or_append(headers, hdr["FIELD"], hdr)
        # derotate frame - necessary for crosstalk correction
        if derotate:
            cube = derotate_cube(cube, prim_hdr["DEROTANG"])
            cube_err = derotate_cube(cube_err, prim_hdr["DEROTANG"])
            prim_hdr = apply_wcs(cube, prim_hdr, angle=0)
        # store into dictionaries
        key = prim_hdr["RET-ANG1"], prim_hdr["U_FLC"], prim_hdr["U_CAMERA"]
        header_dict[key] = prim_hdr
        cube_dict[key] = cube
        cube_errs[key] = cube_err
    assert len(cube_dict) == 16

    stokes_cube = triple_diff_dict(cube_dict)
    # swap stokes and field axes so field is first
    stokes_cube = np.swapaxes(stokes_cube, 0, 1)
    stokes_err = np.sqrt(np.sum(np.power(list(cube_errs.values()), 2), axis=0)) / 16
    stokes_err = np.swapaxes((stokes_err, stokes_err, stokes_err, stokes_err), 0, 1)

    stokes_hdrs: dict[str, fits.Header] = {}
    for key, hdrs in headers.items():
        stokes_hdr = apply_wcs(stokes_cube, combine_frames_headers(hdrs), angle=0)
        # reduce exptime by 2 because cam1 and cam2 are simultaneous
        if "TINT" and "NCOADD" in stokes_hdr:
            stokes_hdr["NCOADD"] /= 2
            stokes_hdr["TINT"] /= 2
        stokes_hdrs[key] = stokes_hdr

    # reform hdulist
    prim_hdr = stokes_hdrs.pop("PRIMARY")
    prim_hdu = fits.PrimaryHDU(stokes_cube, header=prim_hdr)
    err_hdu = fits.ImageHDU(stokes_err, header=prim_hdr, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    hdul.extend([fits.ImageHDU(header=hdr, name=key) for key, hdr in stokes_hdrs.items()])
    return hdul


def polarization_calibration_triplediff_cube(filenames: Sequence[str], derotate: bool = True):
    if len(filenames) % 16 != 0:
        msg = "Cannot do triple-differential calibration without exact sets of 16 frames for each HWP cycle"
        raise ValueError(msg)
    # now do triple-differential calibration
    cube_dict = {}
    header_dict = {}
    cube_errs = {}
    headers: dict[str, fits.Header] = {}
    for file in filenames:
        with fits.open(file) as hdul:
            cube = hdul[0].data
            cube_err = hdul["ERR"].data
            prim_hdr = hdul[0].header
            create_or_append(headers, "PRIMARY", prim_hdr)
            for hdu in hdul[2:]:
                hdr = apply_wcs(cube, hdu.header, angle=0)
                create_or_append(headers, hdr["FIELD"], hdr)
        # derotate frame - necessary for crosstalk correction
        if derotate:
            for wl_idx in range(cube.shape[1]):
                cube[:, wl_idx] = derotate_cube(cube[:, wl_idx], prim_hdr["DEROTANG"])
                cube_err[:, wl_idx] = derotate_cube(cube_err[:, wl_idx], prim_hdr["DEROTANG"])
        prim_hdr = apply_wcs(cube, prim_hdr, angle=0)
        # store into dictionaries
        key = prim_hdr["RET-ANG1"], prim_hdr["U_FLC"], prim_hdr["U_CAMERA"]
        header_dict[key] = prim_hdr
        cube_dict[key] = cube
        cube_errs[key] = cube_err
    assert len(cube_dict) == 16

    # make sure all arrays are same length
    min_length = min(c.shape[0] for c in cube_dict.values())
    for key in cube_dict:
        cube_dict[key] = cube_dict[key][:min_length]

    stokes_cube = triple_diff_dict(cube_dict)
    # collapse time axis, now
    stokes_cube = bn.nanmedian(stokes_cube, axis=1)
    # swap stokes and field axes so field is first
    stokes_cube = np.swapaxes(stokes_cube, 0, 1)
    stokes_err = np.zeros_like(stokes_cube)  # TODO
    # stokes_err = np.sqrt(np.sum(np.power(list(cube_errs.values()), 2), axis=0)) / 16
    # stokes_err = np.swapaxes((stokes_err, stokes_err, stokes_err, stokes_err), 0, 1)

    stokes_hdrs: dict[str, fits.Header] = {}
    for key, hdrs in headers.items():
        stokes_hdr = apply_wcs(stokes_cube, combine_frames_headers(hdrs), angle=0)
        # reduce exptime by 2 because cam1 and cam2 are simultaneous
        if "TINT" and "NCOADD" in stokes_hdr:
            stokes_hdr["NCOADD"] /= 2
            stokes_hdr["TINT"] /= 2
        stokes_hdrs[key] = stokes_hdr

    # reform hdulist
    prim_hdr = stokes_hdrs.pop("PRIMARY")
    prim_hdu = fits.PrimaryHDU(stokes_cube, header=prim_hdr)
    err_hdu = fits.ImageHDU(stokes_err, header=prim_hdr, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    hdul.extend([fits.ImageHDU(header=hdr, name=key) for key, hdr in stokes_hdrs.items()])
    return hdul


def polarization_calibration_doublediff(filenames: Sequence[str], derotate: bool = True):
    if len(filenames) % 8 != 0:
        msg = "Cannot do double-differential calibration without exact sets of 8 frames for each HWP cycle"
        raise ValueError(msg)
    # now do double-differential calibration
    cube_dict = {}
    header_dict = {}
    cube_errs = {}
    headers: dict[str, fits.Header] = {}
    for file in filenames:
        with fits.open(file) as hdul:
            cube = hdul[0].data
            cube_err = hdul["ERR"].data
            prim_hdr = hdul[0].header
            create_or_append(headers, "PRIMARY", prim_hdr)
            for hdu in hdul[2:]:
                hdr = apply_wcs(cube, hdu.header, angle=0)
                create_or_append(headers, hdr["FIELD"], hdr)
        # derotate frame - necessary for crosstalk correction
        if derotate:
            cube = derotate_cube(cube, prim_hdr["DEROTANG"])
            cube_err = derotate_cube(cube_err, prim_hdr["DEROTANG"])
        # store into dictionaries
        prim_hdr = apply_wcs(cube, prim_hdr, angle=0)
        key = prim_hdr["RET-ANG1"], prim_hdr["U_CAMERA"]
        header_dict[key] = prim_hdr
        cube_dict[key] = cube
        cube_errs[key] = cube_err

    stokes_cube = double_diff_dict(cube_dict)
    # swap stokes and field axes so field is first
    stokes_cube = np.swapaxes(stokes_cube, 0, 1)
    stokes_err = np.sqrt(np.sum(np.power(list(cube_errs.values()), 2), axis=0)) / 8
    stokes_err = np.swapaxes((stokes_err, stokes_err, stokes_err, stokes_err), 0, 1)

    stokes_hdrs: dict[str, fits.Header] = {}
    for key, hdrs in headers.items():
        stokes_hdr = apply_wcs(stokes_cube, combine_frames_headers(hdrs), angle=0)
        # reduce exptime by 2 because cam1 and cam2 are simultaneous
        if "TINT" in stokes_hdr:
            stokes_hdr["TINT"] /= 2
        if "NCOADD" in stokes_hdr:
            stokes_hdr["NCOADD"] /= 2
        stokes_hdrs[key] = stokes_hdr

    # reform hdulist
    prim_hdr = stokes_hdrs.pop("PRIMARY")
    prim_hdu = fits.PrimaryHDU(stokes_cube, header=prim_hdr)
    err_hdu = fits.ImageHDU(stokes_err, header=prim_hdr, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    hdul.extend([fits.ImageHDU(header=hdr, name=key) for key, hdr in stokes_hdrs.items()])
    return hdul


def polarization_calibration_doublediff_cube(filenames: Sequence[str], derotate: bool = True):
    if len(filenames) % 8 != 0:
        msg = "Cannot do double-differential calibration without exact sets of 8 frames for each HWP cycle"
        raise ValueError(msg)
    # now do double-differential calibration
    cube_dict = {}
    header_dict = {}
    cube_errs = {}
    headers: dict[str, fits.Header] = {}
    for file in filenames:
        with fits.open(file) as hdul:
            cube = hdul[0].data
            cube_err = hdul["ERR"].data
            prim_hdr = hdul[0].header
            create_or_append(headers, "PRIMARY", prim_hdr)
            for hdu in hdul[2:]:
                hdr = apply_wcs(cube, hdu.header, angle=0)
                create_or_append(headers, hdr["FIELD"], hdr)
        # derotate frame - necessary for crosstalk correction
        if derotate:
            for wl_idx in range(cube.shape[1]):
                cube[:, wl_idx] = derotate_cube(cube[:, wl_idx], prim_hdr["DEROTANG"])
                cube_err[:, wl_idx] = derotate_cube(cube_err[:, wl_idx], prim_hdr["DEROTANG"])
        # store into dictionaries
        prim_hdr = apply_wcs(cube, prim_hdr, angle=0)
        key = prim_hdr["RET-ANG1"], prim_hdr["U_CAMERA"]
        header_dict[key] = prim_hdr
        cube_dict[key] = cube
        cube_errs[key] = cube_err

    # make sure all arrays are same length
    min_length = min(c.shape[0] for c in cube_dict.values())
    for key in cube_dict:
        cube_dict[key] = cube_dict[key][:min_length]

    stokes_cube = double_diff_dict(cube_dict)
    # collapse time axis, now
    stokes_cube = bn.nanmedian(stokes_cube, axis=1)
    # swap stokes and field axes so field is first
    stokes_cube = np.swapaxes(stokes_cube, 0, 1)
    stokes_err = np.zeros_like(stokes_cube)  # TODO

    stokes_hdrs: dict[str, fits.Header] = {}
    for key, hdrs in headers.items():
        stokes_hdr = apply_wcs(stokes_cube, combine_frames_headers(hdrs), angle=0)
        # reduce exptime by 2 because cam1 and cam2 are simultaneous
        if "TINT" in stokes_hdr:
            stokes_hdr["TINT"] /= 2
        if "NCOADD" in stokes_hdr:
            stokes_hdr["NCOADD"] /= 2
        stokes_hdrs[key] = stokes_hdr

    # reform hdulist
    prim_hdr = stokes_hdrs.pop("PRIMARY")
    prim_hdu = fits.PrimaryHDU(stokes_cube, header=prim_hdr)
    err_hdu = fits.ImageHDU(stokes_err, header=prim_hdr, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    hdul.extend([fits.ImageHDU(header=hdr, name=key) for key, hdr in stokes_hdrs.items()])
    return hdul


def make_triplediff_dict(filenames):
    output = {}
    for file in filenames:
        data, hdr = load_fits(file, header=True)
        # get row vector for mueller matrix of each file
        # store into dictionaries
        key = hdr["RET-ANG1"], hdr["U_FLC"], hdr["U_CAMERA"]
        output[key] = data
    return output


def make_doublediff_dict(filenames):
    output = {}
    for file in filenames:
        data, hdr = load_fits(file, header=True)
        # get row vector for mueller matrix of each file
        # store into dictionaries
        key = hdr["RET-ANG1"], hdr["U_CAMERA"]
        output[key] = data
    return output


T = TypeVar("T")


def triple_diff_dict(input_dict: dict[tuple[float, str, int], T]) -> tuple[T, T, T, T]:
    ## make difference images
    # single diff (cams)
    pQ0 = 0.5 * (input_dict[(0.0, "A", 1)] - input_dict[(0.0, "A", 2)])
    pIQ0 = 0.5 * (input_dict[(0.0, "A", 1)] + input_dict[(0.0, "A", 2)])
    pQ1 = 0.5 * (input_dict[(0.0, "B", 1)] - input_dict[(0.0, "B", 2)])
    pIQ1 = 0.5 * (input_dict[(0.0, "B", 1)] + input_dict[(0.0, "B", 2)])

    mQ0 = 0.5 * (input_dict[(45.0, "A", 1)] - input_dict[(45.0, "A", 2)])
    mIQ0 = 0.5 * (input_dict[(45.0, "A", 1)] + input_dict[(45.0, "A", 2)])
    mQ1 = 0.5 * (input_dict[(45.0, "B", 1)] - input_dict[(45.0, "B", 2)])
    mIQ1 = 0.5 * (input_dict[(45.0, "B", 1)] + input_dict[(45.0, "B", 2)])

    pU0 = 0.5 * (input_dict[(22.5, "A", 1)] - input_dict[(22.5, "A", 2)])
    pIU0 = 0.5 * (input_dict[(22.5, "A", 1)] + input_dict[(22.5, "A", 2)])
    pU1 = 0.5 * (input_dict[(22.5, "B", 1)] - input_dict[(22.5, "B", 2)])
    pIU1 = 0.5 * (input_dict[(22.5, "B", 1)] + input_dict[(22.5, "B", 2)])

    mU0 = 0.5 * (input_dict[(67.5, "A", 1)] - input_dict[(67.5, "A", 2)])
    mIU0 = 0.5 * (input_dict[(67.5, "A", 1)] + input_dict[(67.5, "A", 2)])
    mU1 = 0.5 * (input_dict[(67.5, "B", 1)] - input_dict[(67.5, "B", 2)])
    mIU1 = 0.5 * (input_dict[(67.5, "B", 1)] + input_dict[(67.5, "B", 2)])

    # double difference (FLC1 - FLC2)
    pQ = 0.5 * (pQ0 - pQ1)
    pIQ = 0.5 * (pIQ0 + pIQ1)

    mQ = 0.5 * (mQ0 - mQ1)
    mIQ = 0.5 * (mIQ0 + mIQ1)

    pU = 0.5 * (pU0 - pU1)
    pIU = 0.5 * (pIU0 + pIU1)

    mU = 0.5 * (mU0 - mU1)
    mIU = 0.5 * (mIU0 + mIU1)

    # triple difference (HWP1 - HWP2)
    Q = 0.5 * (pQ - mQ)
    IQ = 0.5 * (pIQ + mIQ)
    U = 0.5 * (pU - mU)
    IU = 0.5 * (pIU + mIU)

    return IQ, IU, Q, U


def double_diff_dict(input_dict: dict[tuple[float, str, int], T]) -> tuple[T, T, T, T]:
    ## make difference images
    # single diff (cams)
    pQ = 0.5 * (input_dict[(0.0, 1)] - input_dict[(0.0, 2)])
    pIQ = 0.5 * (input_dict[(0.0, 1)] + input_dict[(0.0, 2)])

    mQ = 0.5 * (input_dict[(45.0, 1)] - input_dict[(45.0, 2)])
    mIQ = 0.5 * (input_dict[(45.0, 1)] + input_dict[(45.0, 2)])

    pU = 0.5 * (input_dict[(22.5, 1)] - input_dict[(22.5, 2)])
    pIU = 0.5 * (input_dict[(22.5, 1)] + input_dict[(22.5, 2)])

    mU = 0.5 * (input_dict[(67.5, 1)] - input_dict[(67.5, 2)])
    mIU = 0.5 * (input_dict[(67.5, 1)] + input_dict[(67.5, 2)])

    # double difference (HWP1 - HWP2)
    Q = 0.5 * (pQ - mQ)
    IQ = 0.5 * (pIQ + mIQ)
    U = 0.5 * (pU - mU)
    IU = 0.5 * (pIU + mIU)

    return IQ, IU, Q, U


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
    cubes = []
    headers = []
    mueller_mats = []
    for file, mm_file in tqdm.tqdm(
        zip(filenames, mm_filenames, strict=True),
        total=len(filenames),
        desc="Least-squares calibration",
    ):
        cube, hdr = load_fits(file, header=True, memmap=False)
        # rotate to N up E left
        cube_derot = derotate_frame(cube, hdr["DEROTANG"])
        # get row vector for mueller matrix of each file
        mueller_mat = load_fits(mm_file, memmap=False)[0]

        cubes.append(cube_derot)
        headers.append(hdr)
        mueller_mats.append(mueller_mat)

    mueller_mat_cube = np.array(mueller_mats)
    super_cube = np.array(cubes)
    stokes_hdr = combine_frames_headers(headers, wcs=True)
    stokes_final = []
    for cube in super_cube:
        stokes_cube = mueller_matrix_calibration(mueller_mat_cube, cube)
        stokes_final.append(stokes_cube)
    stokes_final = np.array(stokes_final)
    return write_stokes_products(stokes_final, stokes_hdr, outname=path, force=True)


DOUBLEDIFF_SETS = set(itertools.product((0.0, 45.0, 22.5, 67.5), (1, 2)))
TRIPLEDIFF_SETS = set(itertools.product((0.0, 45.0, 22.5, 67.5), ("A", "B"), (1, 2)))


def get_triplediff_set(table) -> dict | None:
    columns = ["path", "UT", "MJD", "PA", "DEROTANG", "RET-ANG1", "U_FLC", "U_CAMERA"]
    work_table = table[columns].copy()
    stokes_idx = 0
    output_tables = []
    bad_matches = []
    while len(work_table) >= 16:
        row_set = work_table.iloc[:16]
        for key in TRIPLEDIFF_SETS:
            subset = row_set.query(
                f"`RET-ANG1` == {key[0]} and U_FLC == '{key[1]}' and U_CAMERA == {key[2]}"
            )
            if len(subset) == 0:
                # found a missing key!
                row_set = None
                break

        # if there's a missing key let's try sliding the window down to the next timestamp
        if row_set is None:
            indices_to_drop = work_table["UT"] == work_table["UT"].iloc[0]
            bad_matches.append(work_table.loc[indices_to_drop])
            work_table = work_table.loc[~indices_to_drop]
        else:
            current_table = pd.DataFrame(row_set)
            current_table["STOKES_IDX"] = stokes_idx
            stokes_idx += 1
            output_tables.append(current_table)
            work_table.drop(work_table.index[:16], axis=0, inplace=True)

    # address bad matches with second search using queries instead of structured indices
    if len(bad_matches) > 0:
        bad_table = pd.concat(bad_matches)
        work_table = table[columns].copy()
        while len(bad_table) > 0:
            row = bad_table.iloc[0]
            row_key = (row["RET-ANG1"], row["U_FLC"], int(row["U_CAMERA"]))
            work_table["DELTA_PA"] = np.abs(work_table["PA"] - row["PA"])
            work_table["DELTA_TIME"] = np.abs(work_table["MJD"] - row["MJD"])

            output_set = [row]
            remaining_keys = TRIPLEDIFF_SETS - set((row_key,))
            for key in remaining_keys:
                subset = work_table.query(
                    f"`RET-ANG1` == {key[0]} and U_FLC == '{key[1]}' and U_CAMERA == {key[2]} and DELTA_PA < 4"
                )

                if len(subset) > 0:
                    idx = subset["DELTA_TIME"].argmin()
                    output_set.append(subset.iloc[idx])
                else:
                    output_set = None
                    break

            if output_set is None:
                # pop this row from working table, and any temporal matches
                indices_to_drop = bad_table["UT"] == row["UT"]
                current_table = bad_table.loc[indices_to_drop].copy()
                current_table["STOKES_IDX"] = -1
                bad_table = bad_table.loc[~indices_to_drop]
            else:
                current_table = pd.DataFrame(output_set).drop(["DELTA_PA", "DELTA_TIME"], axis=1)
                current_table["STOKES_IDX"] = stokes_idx
                stokes_idx += 1
                # pop full output set
                for row in current_table.itertuples():
                    indices_to_drop = bad_table["path"] == row.path
                    bad_table = bad_table.loc[~indices_to_drop]

            output_tables.append(current_table)
    if len(output_tables) > 0:
        final_table = pd.concat(output_tables)
    else:
        msg = "Could not generate any valid Stokes sets."
        raise RuntimeError(msg)

    final_table.sort_values(["MJD", "U_FLC", "U_CAMERA"], inplace=True)
    final_table["STOKES_IDX"] = reindex_stokes_index(final_table["STOKES_IDX"])

    return final_table


def get_doublediff_set(table) -> dict | None:
    columns = ["path", "UT", "MJD", "PA", "DEROTANG", "RET-ANG1", "U_CAMERA"]
    work_table = table[columns].copy()
    stokes_idx = 0
    output_tables = []
    bad_matches = []
    while len(work_table) >= 8:
        row_set = work_table.iloc[:8]
        for key in DOUBLEDIFF_SETS:
            subset = row_set.query(f"`RET-ANG1` == {key[0]} and U_CAMERA == {key[1]}")
            if len(subset) == 0:
                # found a missing key!
                row_set = None
                break

        # if there's a missing key let's try sliding the window down to the next timestamp
        if row_set is None:
            indices_to_drop = work_table["UT"] == work_table["UT"].iloc[0]
            bad_matches.append(work_table.loc[indices_to_drop])
            work_table = work_table.loc[~indices_to_drop]
        else:
            current_table = pd.DataFrame(row_set)
            current_table["STOKES_IDX"] = stokes_idx
            stokes_idx += 1
            output_tables.append(current_table)
            work_table.drop(work_table.index[:8], axis=0, inplace=True)

    # address bad matches with second search using queries instead of structured indices
    if len(bad_matches) > 0:
        bad_table = pd.concat(bad_matches)
        work_table = table[columns].copy()
        while len(bad_table) > 0:
            row = bad_table.iloc[0]
            row_key = (row["RET-ANG1"], int(row["U_CAMERA"]))
            work_table["DELTA_PA"] = np.abs(work_table["PA"] - row["PA"])
            work_table["DELTA_TIME"] = np.abs(work_table["MJD"] - row["MJD"])

            output_set = [row]
            remaining_keys = DOUBLEDIFF_SETS - set((row_key,))
            for key in remaining_keys:
                subset = work_table.query(
                    f"`RET-ANG1` == {key[0]} and U_CAMERA == {key[1]} and DELTA_PA < 4"
                )

                if len(subset) > 0:
                    idx = subset["DELTA_TIME"].argmin()
                    output_set.append(subset.iloc[idx])
                else:
                    output_set = None
                    break

            if output_set is None:
                # pop this row from working table, and any temporal matches
                indices_to_drop = bad_table["UT"] == row["UT"]
                current_table = bad_table.loc[indices_to_drop].copy()
                current_table["STOKES_IDX"] = -1
                bad_table = bad_table.loc[~indices_to_drop]
            else:
                current_table = pd.DataFrame(output_set).drop(["DELTA_PA", "DELTA_TIME"], axis=1)
                current_table["STOKES_IDX"] = stokes_idx
                stokes_idx += 1
                # pop full output set
                for row in current_table.itertuples():
                    indices_to_drop = bad_table["path"] == row.path
                    bad_table = bad_table.loc[~indices_to_drop]

            output_tables.append(current_table)

    final_table = pd.concat(output_tables)

    final_table.sort_values(["MJD", "U_CAMERA"], inplace=True)
    final_table["STOKES_IDX"] = reindex_stokes_index(final_table["STOKES_IDX"])

    return final_table


def reindex_stokes_index(stokes_idxs: pd.Series) -> list:
    # sorted unique elements
    unique_sorted = stokes_idxs[stokes_idxs != -1].unique()
    # map indices to their index in the unique sorted list
    mapping = {num: i for i, num in enumerate(unique_sorted)}
    # except for -1, which we propagate without keeping it in our unique last
    mapping[-1] = -1
    mapped_list = [mapping[num] for num in stokes_idxs.values]
    # make sure we haven't borked it
    assert min(mapped_list) == stokes_idxs.min()
    assert max(mapped_list) == stokes_idxs.max()
    return mapped_list


def make_stokes_image(
    path_set,
    outpath: Path,
    mm_paths=None,
    method="triplediff",
    coadded: bool = True,
    derotate: bool = True,
    mm_correct: bool = True,
    ip_correct: bool = True,
    hwp_adi_sync: bool = True,
    ip_radius=8,
    ip_radius2=8,
    ip_method="photometry",
    force: bool = False,
):
    if not force and outpath.exists() and not any_file_newer(path_set, outpath):
        return outpath

    if mm_correct and not derotate:
        msg = "You cannot apply MM correction without derotating data, please check configuration"
        raise ValueError(msg)

    # create stokes cube
    if method == "triplediff":
        if coadded:
            stokes_hdul = polarization_calibration_triplediff(path_set, derotate=derotate)
        else:
            stokes_hdul = polarization_calibration_triplediff_cube(path_set, derotate=derotate)
        if mm_correct:
            mm_dict = make_triplediff_dict(mm_paths)
            _, _, mmQs, mmUs = triple_diff_dict(mm_dict)
    elif method == "doublediff":
        if coadded:
            stokes_hdul = polarization_calibration_doublediff(path_set, derotate=derotate)
        else:
            stokes_hdul = polarization_calibration_doublediff_cube(path_set, derotate=derotate)
        if mm_correct:
            mm_dict = make_doublediff_dict(mm_paths)
            _, _, mmQs, mmUs = double_diff_dict(mm_dict)
    else:
        msg = f"Unrecognized method {method}"
        raise ValueError(msg)

    stokes_data = stokes_hdul[0].data
    stokes_err = stokes_hdul["ERR"].data
    stokes_outdata = np.empty_like(stokes_data)
    stokes_outerr = np.empty_like(stokes_err)
    prim_hdr = stokes_hdul[0].header
    headers = [stokes_hdul[i].header for i in range(2, len(stokes_hdul))]
    mms = []
    for i in range(stokes_data.shape[0]):
        stokes_frame = stokes_data[i]
        stokes_frame_err = stokes_err[i]
        stokes_header = headers[i]
        field = stokes_header["FIELD"]
        # mm correct
        if mm_correct:
            # get first row of diffed Mueller-matrix
            # note: multiplying by two because of extra 0.5 in _diff_dict method
            mmQ = 2 * mmQs[i, 0]
            mmU = 2 * mmUs[i, 0]
            mms.append(np.array((mmQ, mmU)))
            IQ, IU, Q, U = stokes_frame  # noqa: E741
            IQ_err, IU_err, Q_err, U_err = stokes_frame_err
            # correct IP
            Q -= mmQ[0] * IQ
            U -= mmU[0] * IU
            Q_err = np.hypot(Q_err, mmQ[0] * IQ_err)
            U_err = np.hypot(U_err, mmU[0] * IU_err)

            # correct cross-talk
            Sarr = np.array((Q.ravel(), U.ravel()))
            Marr = np.array((mmQ[1:3], mmU[1:3]))
            res = np.linalg.lstsq(Marr, Sarr, rcond=None)[0]
            Q = res[0].reshape(Q.shape[-2:])
            U = res[1].reshape(U.shape[-2:])
            stokes_frame = np.array((IQ, IU, Q, U))
            stokes_frame_err = np.array((IQ_err, IU_err, Q_err, U_err))
            stokes_header[f"hierarch DPP PDI MM IP_PQ {field}"] = (
                mmQ[0],
                "DPP IP (I -> Q) from Mueller matrix",
            )
            stokes_header[f"hierarch DPP PDI MM IP_PU {field}"] = (
                mmU[0],
                "DPP IP (I -> U) from Mueller matrix",
            )
            average_poleff = calculate_pol_efficiency(mmQ, mmU)
            stokes_header[f"hierarch DPP PDI MM POLEFF {field}"] = (
                average_poleff,
                "DPP polarimetric efficiency from Mueller matrix",
            )
        elif derotate and not hwp_adi_sync:
            # if HWP ADI sync is off but we don't do Mueller correction
            # we need to manually rotate stokes values by the derotation angle
            angle = stokes_header["DEROTANG"]
            stokes_frame = rotate_stokes(stokes_frame, angle)
            stokes_frame_err = rotate_stokes(stokes_frame_err, angle)

        # second order IP correction
        if ip_correct:
            stokes_frame, stokes_header = polarization_ip_correct(
                stokes_frame,
                phot_rad=(ip_radius, ip_radius2),
                method=ip_method,
                header=stokes_header,
            )
            pQ, pU = (
                stokes_header[f"hierarch DPP PDI IP_PQ {field}"],
                stokes_header[f"hierarch DPP PDI IP_PU {field}"],
            )
            stokes_frame_err[2] = np.hypot(stokes_frame_err[2], pQ * stokes_frame_err[0])
            stokes_frame_err[3] = np.hypot(stokes_frame_err[3], pU * stokes_frame_err[1])

        stokes_header["CTYPE3"] = "STOKES"
        stokes_header["STOKES"] = "I_Q,I_U,Q,U", "Stokes axis data type"
        stokes_outdata[i] = stokes_frame
        stokes_outerr[i] = stokes_frame_err
        headers[i] = stokes_header
    # have to awkwardly combine since there's no NAXIS keywords
    prim_hdr = apply_wcs(stokes_outdata, combine_frames_headers(headers), angle=0)
    if "NCOADD" in prim_hdr:
        prim_hdr["NCOADD"] /= len(headers)
    if "TINT" in prim_hdr:
        prim_hdr["TINT"] /= len(headers)
    prim_hdr = sort_header(prim_hdr)
    prim_hdu = fits.PrimaryHDU(stokes_outdata, header=prim_hdr)
    err_hdu = fits.ImageHDU(stokes_err, header=prim_hdr, name="ERR")
    hdul = fits.HDUList([prim_hdu, err_hdu])
    hdul.extend([fits.ImageHDU(header=sort_header(hdr), name=hdr["FIELD"]) for hdr in headers])
    if mm_correct:
        for hdr, mm in zip(headers, mms, strict=True):
            hdul.append(fits.ImageHDU(mm, header=sort_header(hdr), name=hdr["FIELD"] + "MM"))
    hdul.writeto(outpath, overwrite=True)
    return outpath


def polarization_ip_correct(stokes_data, phot_rad, method, header=None):
    if method == "aperture":
        pQ = measure_instpol(stokes_data[0], stokes_data[2], r=phot_rad[0])
        pU = measure_instpol(stokes_data[1], stokes_data[3], r=phot_rad[0])
    elif method == "annulus":
        pQ = measure_instpol_ann(stokes_data[0], stokes_data[2], Rin=phot_rad[0], Rout=phot_rad[1])
        pU = measure_instpol_ann(stokes_data[1], stokes_data[3], Rin=phot_rad[0], Rout=phot_rad[1])

    stokes_data[2] -= pQ * stokes_data[0]
    stokes_data[3] -= pU * stokes_data[1]

    if header is not None:
        field = header["FIELD"]
        header[f"hierarch DPP PDI IP_PQ {field}"] = pQ, "I -> Q IP correction value"
        header[f"hierarch DPP PDI IP_PU {field}"] = pU, "I -> U IP correction value"
        header[f"hierarch DPP PDI IP_POL {field}"] = np.hypot(pU, pQ), "Residual IP DoLP"
        header[f"hierarch DPP PDI IP_ANG {field}"] = (
            0.5 * np.rad2deg(np.arctan2(pU, pQ)),
            "[deg] Residual IP AoLP",
        )
        header[f"hierarch DPP PDI IP_METH {field}"] = method, "IP measurement method"
    return stokes_data, header
