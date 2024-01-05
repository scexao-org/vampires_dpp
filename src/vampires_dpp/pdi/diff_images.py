from pathlib import Path

import numpy as np
from astropy.io import fits
from reproject import reproject_interp

from vampires_dpp.headers import sort_header
from vampires_dpp.image_processing import combine_frames_headers
from vampires_dpp.paths import any_file_newer
from vampires_dpp.wcs import apply_wcs


def get_singlediff_sets(table):
    path_sets = []
    for _k, group in table.sort_values(["MJD", "U_CAMERA", "U_FLC"]).groupby("MJD"):
        if len(group) == 2:
            path_sets.append(group["path"])
        elif len(group) == 4:  # if FLC deinterleaved, create two different single diff images
            for _, grp in group.groupby("U_FLC"):
                path_sets.append(grp["path"])
    return path_sets


def get_doublediff_sets(table):
    path_sets = []
    for _, group in table.sort_values(["MJD", "U_CAMERA", "U_FLC"]).groupby("MJD"):
        if len(group) < 4:
            continue
        path_sets.append(group["path"])
    return path_sets


def singlediff_images(paths, outpath: Path, force: bool = False) -> Path:
    if not force and outpath.exists() and not any_file_newer(paths, outpath):
        return outpath
    data = {}
    errs = {}
    prim_hdrs = {}
    hdrs = {}
    for path in paths:
        with fits.open(path) as hdul:
            prim_hdr = hdul[0].header
            key = prim_hdr["U_CAMERA"]
            prim_hdrs[key] = prim_hdr
            data[key] = hdul[0].data
            errs[key] = hdul["ERR"].data
            hdrs[key] = [hdul[i].header for i in range(3, len(hdul))]
    if len(data) < 2:
        return None
    # reproject cam2 onto cam1
    reproject_interp(
        (np.nan_to_num(data[2]), prim_hdrs[2]),
        prim_hdrs[1],
        return_footprint=False,
        output_array=data[2],
        order="bicubic",
    )
    reproject_interp(
        (np.nan_to_num(errs[2]), prim_hdrs[2]),
        prim_hdrs[1],
        return_footprint=False,
        output_array=errs[2],
        order="bicubic",
    )
    single_diff = data[1] - data[2]
    single_sum = data[1] + data[2]
    single_err = np.hypot(errs[1], errs[2])
    comb_hdrs = []
    for i in range(single_diff.shape[0]):
        headers = (hdrs[1][i], hdrs[2][i])
        hdr = combine_frames_headers(headers)
        if "NCOADD" in hdr:
            hdr["NCOADD"] /= 2
        if "TINT" in hdr:
            hdr["TINT"] /= 2
        comb_hdrs.append(hdr)
    prim_hdr = combine_frames_headers(list(prim_hdrs.values()))
    if "NCOADD" in prim_hdr:
        prim_hdr["NCOADD"] /= 2
    if "TINT" in prim_hdr:
        prim_hdr["TINT"] /= 2
    prim_hdr = apply_wcs(single_diff, prim_hdr, angle=prim_hdr["DEROTANG"])
    prim_hdr = sort_header(prim_hdr)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(single_diff, header=prim_hdr),
            fits.ImageHDU(single_sum, header=prim_hdr, name="SUM"),
            fits.ImageHDU(single_err, header=prim_hdr, name="ERR"),
        ]
    )
    hdul.extend([fits.ImageHDU(header=sort_header(hdr)) for hdr in comb_hdrs])

    hdul.writeto(outpath, overwrite=True)
    return outpath


def doublediff_images(paths, outpath: Path, force: bool = False) -> Path:
    if not force and outpath.exists() and not any_file_newer(paths, outpath):
        return outpath
    data = {}
    errs = {}
    prim_hdrs = {}
    hdrs = {}
    for path in paths:
        with fits.open(path) as hdul:
            prim_hdr = hdul[0].header
            key = (prim_hdr["U_CAMERA"], prim_hdr["U_FLC"])
            prim_hdrs[key] = prim_hdr
            data[key] = hdul[0].data
            errs[key] = hdul["ERR"].data
            hdrs[key] = [hdul[i].header for i in range(3, len(hdul))]

    if len(data) < 4:
        return None
    # reproject cam2 onto cam1
    for key in ("A", "B"):
        reproject_interp(
            (np.nan_to_num(data[2, key]), prim_hdrs[2, key]),
            prim_hdrs[1, key],
            return_footprint=False,
            output_array=data[2, key],
            order="bicubic",
        )
        reproject_interp(
            (np.nan_to_num(errs[2, key]), prim_hdrs[2, key]),
            prim_hdrs[1, key],
            return_footprint=False,
            output_array=errs[2, key],
            order="bicubic",
        )

    diff_A = data[1, "A"] - data[2, "A"]
    diff_B = data[1, "B"] - data[2, "B"]
    double_diff = 0.5 * (diff_A - diff_B)
    double_sum = 2 * np.mean(list(data.values()), axis=0)
    double_err = 0.5 * np.sqrt(np.sum(np.power(list(errs.values()), 2), axis=0))
    prim_hdr = combine_frames_headers(list(prim_hdrs.values()))
    if "NCOADD" in prim_hdr:
        prim_hdr["NCOADD"] /= 4
    if "TINT" in prim_hdr:
        prim_hdr["TINT"] /= 4
    prim_hdr = apply_wcs(double_diff, prim_hdr, angle=prim_hdr["DEROTANG"])
    prim_hdr = sort_header(prim_hdr)
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(double_diff, header=prim_hdr),
            fits.ImageHDU(double_sum, header=prim_hdr, name="SUM"),
            fits.ImageHDU(double_err, header=prim_hdr, name="ERR"),
        ]
    )
    for i in range(double_diff.shape[0]):
        headers = [hdr[i] for hdr in hdrs.values()]
        hdr = sort_header(combine_frames_headers(headers))
        hdul.append(fits.ImageHDU(header=hdr))

    hdul.writeto(outpath, overwrite=True)
    return outpath
