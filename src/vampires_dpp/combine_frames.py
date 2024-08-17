import functools
from typing import Literal

import numpy as np
import pandas as pd
from astropy.io import fits

from .image_processing import combine_frames_headers


def generate_frame_combinations(
    table: pd.DataFrame, method: Literal["cube", "pdi"], save_intermediate: bool = False
):
    work_table = table.sort_values("MJD")
    # have to force U_FLC to be a string because otherwise we'll never
    # have matching indices with NaN, due to NaN equality being NaN
    work_table.loc[work_table["U_FLC"].isna(), "U_FLC"] = "NA"
    tables = []
    for _, group in work_table.groupby(["U_CAMERA", "U_FLC"]):
        match method:
            case "pdi":
                framelist = generate_framelist_for_hwp_angles(group)
            case "cube":
                framelist = group.copy()
                framelist["GROUP_IDX"] = range(len(group))
            case _:
                msg = f"Unrecognized combination method {method}"
                raise ValueError(msg)
        tables.append(framelist)
    return pd.concat(tables)


def generate_framelist_for_hwp_angles(table: pd.DataFrame) -> pd.DataFrame:
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
    work_table["GROUP_IDX"] = frameinds
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
        hdul_out[idx].data = np.stack((data1, data2), axis=0)
        hdul_out[idx].header = combine_frames_headers((hdr1, hdr2), wcs=True)
    return hdul_out


def combine_hduls(hduls: list[fits.HDUList]):
    return functools.reduce(_merge_two_hdul, hduls)
