import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.io import fits

from vampires_dpp.headers import sort_header
from vampires_dpp.organization import dict_from_header
from vampires_dpp.util import delta_angle, hst_from_ut_time, iso_time_stats, load_fits

from .image_processing import crop_to_nans_inds
from .paths import any_file_newer


def generate_frame_combinations(
    table: pd.DataFrame, method: Literal["cube", "pdi"], save_intermediate: bool = False
):
    work_table = table.sort_values("MJD")
    if method == "cube":
        output_table = work_table.copy()
        output_table["GROUP_KEY"] = [f"{idx:04d}" for idx in range(len(work_table))]
        return output_table
    # have to force U_FLC to be a string because otherwise we'll never
    # have matching indices with NaN, due to NaN equality being NaN
    work_table.loc[work_table["U_FLC"].isna(), "U_FLC"] = "NA"
    tables = []
    for key, group in work_table.groupby(["U_CAMERA", "U_FLC"]):
        framelist = generate_framelist_for_hwp_angles(group, key)
        tables.append(framelist)
    return pd.concat(tables)


def generate_framelist_for_hwp_angles(table: pd.DataFrame, key: tuple[int, str]) -> pd.DataFrame:
    ## find consistent runs of HWP angles
    work_table = table.copy()
    file_index = 0
    frameinds = [file_index]
    for i in range(1, len(work_table)):
        prev_row = work_table.iloc[i - 1]
        cur_row = work_table.iloc[i]
        if prev_row["RET-ANG1"] != cur_row["RET-ANG1"]:
            file_index += 1
        frameinds.append(file_index)
    work_table["GROUP_KEY"] = [f"{idx:03d}_cam{key[0]:.0f}_FLC{key[1]}" for idx in frameinds]
    return work_table


# def generate_framelist_for_num_frames(table: pd.DataFrame, num_frames: int):
#     ## find consistent runs of HWP angles
#     columns = ["path", "NAXIS3", "UT", "MJD"]
#     work_table = table[columns].copy()
#     framelist = []
#     file_index = 0
#     while len(work_table) > 0:
#         cur_row = work_table.iloc[0]
#         total_frames = cur_row["NAXIS3"]
#         if total_frames > num_frames:
#             max_idx = num_frames - total_frames
#         else:
#             max_idx = cur_row["NAXIS3"]
#         framelist.append({
#             **cur_row[columns],
#             "indices": f"0:{max_idx}",
#             "GROUP_IDX": file_index
#         })
#         last_index_to_remove = 0
#         for i in range(1, len(work_table)):
#             next_row = work_table.iloc[i]
#             if next_row["NAXIS3"] + total_frames < num_frames:
#                 max_idx = next_row["NAXIS3"]
#             max_idx = num_frames - total_frames - next_row["NAXIS3"]
#                 break
#             framelist.append({
#                 **next_row[columns],
#                 "indices": f"0:{max_idx}",
#                 "GROUP_IDX": file_index
#             })
#             last_index_to_remove += 1
#         work_table.drop(work_table.index[:last_index_to_remove], axis=0, inplace=True)
#         file_index += 1
#     return pd.DataFrame(framelist)


def _merge_two_hdul(hdul1, hdul2):
    hdul_out = hdul1.copy()
    for idx in range(len(hdul_out)):
        data1 = hdul1[idx].data
        hdr1 = hdul1[idx].header
        data2 = hdul2[idx].data
        hdr2 = hdul2[idx].header
        hdul_out[idx].data = np.vstack((data1, data2))
        hdul_out[idx].header = combine_frames_headers((hdr1, hdr2), wcs=True)
    return hdul_out


def combine_hduls(hduls: list[fits.HDUList]):
    return functools.reduce(_merge_two_hdul, hduls)


def combine_frames(frames, headers=None, **kwargs):
    cube = np.array(frames)

    if headers is not None:
        headers = combine_frames_headers(headers, **kwargs)

    return cube, headers


WCS_KEYS: Final = {
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

RESERVED_KEYS: Final = {
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
        output_header["TINT"] = table["TINT"].sum(), test_header.comments["TINT"]

    if "NCOADD" in table:
        output_header["NCOADD"] = table["NCOADD"].sum(), test_header.comments["NCOADD"]

    # get PA rotation
    if "PA" in table:
        output_header["PA-STR"] = table["PA-STR"].iloc[0], "[deg] parallactic angle at start"
        output_header["PA-END"] = table["PA-END"].iloc[-1], "[deg] parallactic angle at end"
        total_rot = delta_angle(output_header["PA-STR"], output_header["PA-END"])
        output_header["PA-ROT"] = total_rot, "[deg] total parallactic angle rotation"

    if "DEROTANG" in table:
        angs = Angle(table["DEROTANG"], unit=u.deg)
        ave_ang = np.arctan2(np.sin(angs.rad).mean(), np.cos(angs.rad).mean())
        output_header["DEROTANG"] = np.rad2deg(ave_ang), test_header.comments["DEROTANG"]

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
    # deal with time
    ut_str = ut_end = None
    for _, hdr in table.iterrows():
        ut_stats = iso_time_stats(hdr["DATE-OBS"], hdr["UT-STR"], hdr["UT-END"])
        ut_str = ut_stats[0] if ut_str is None else min(ut_stats[0], ut_str)
        ut_end = ut_stats[-1] if ut_end is None else max(ut_stats[-1], ut_end)
    ut_typ = ut_str + (ut_end - ut_str) / 2

    output_header["UT-STR"] = ut_str.iso.split()[-1], test_header.comments["UT-STR"]
    output_header["UT-END"] = ut_end.iso.split()[-1], test_header.comments["UT-END"]
    output_header["UT"] = ut_typ.iso.split()[-1], test_header.comments["UT"]
    output_header["DATE-OBS"] = ut_typ.iso.split()[0], test_header.comments["DATE-OBS"]

    hst_str = hst_from_ut_time(ut_str)
    hst_typ = hst_from_ut_time(ut_typ)
    hst_end = hst_from_ut_time(ut_end)

    output_header["HST-STR"] = hst_str.iso.split()[-1], test_header.comments["HST-STR"]
    output_header["HST-END"] = hst_end.iso.split()[-1], test_header.comments["HST-END"]
    output_header["HST"] = hst_typ.iso.split()[-1], test_header.comments["HST"]

    output_header["MJD-STR"] = ut_str.mjd, test_header.comments["MJD-STR"]
    output_header["MJD-END"] = ut_end.mjd, test_header.comments["MJD-END"]
    output_header["MJD"] = ut_typ.mjd, test_header.comments["MJD"]

    # WCS
    if wcs:
        # need to get average CRVALs and PCs
        output_header["CRVAL1"] = np.rad2deg(ave_ra), test_header.comments["CRVAL1"]
        output_header["CRVAL2"] = np.rad2deg(ave_dec), test_header.comments["CRVAL2"]
        output_header["PC1_1"] = table["PC1_1"].mean(), test_header.comments["PC1_1"]
        output_header["PC1_2"] = table["PC1_2"].mean(), test_header.comments["PC1_2"]
        output_header["PC2_1"] = table["PC2_1"].mean(), test_header.comments["PC2_1"]
        output_header["PC2_2"] = table["PC2_2"].mean(), test_header.comments["PC2_2"]
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
    pairs = sorted(zip(frames, headers, strict=True), key=lambda t: t[1]["MJD"])
    frames = []
    headers = []
    for frame, header in pairs:
        frames.append(frame)
        headers.append(header)
    cube, header = combine_frames(frames, headers, **kwargs)
    fits.writeto(path, cube, header=sort_header(header), overwrite=True)
    return path
