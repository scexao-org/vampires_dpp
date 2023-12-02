from pathlib import Path

import numpy as np
from astropy.io import fits

from vampires_dpp.image_processing import combine_frames_headers
from vampires_dpp.paths import any_file_newer


def get_singlediff_sets(table):
    path_sets = []
    for _, group in table.sort_values(["MJD", "U_CAMERA"]).groupby("MJD"):
        if len(group) < 2:
            continue
        path_sets.append(group["path"])
    return path_sets


def get_doublediff_sets(table):
    path_sets = []
    for _, group in table.sort_values(["MJD", "U_CAMERA", "U_FLC"]).groupby("MJD"):
        if len(group) < 4:
            continue
        path_sets.append(group["path"])
    return path_sets


def singlediff_images(paths, outpath: Path, force: bool = False) -> fits.HDUList:
    if not force and outpath.exists() and not any_file_newer(paths, outpath):
        return fits.open(outpath)
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
            _hdrs = []
            _errs = []
            for i in range(1, len(hdul), 2):
                extname = hdul[i].name
                _hdrs.append(hdul[i].header)
                _errs.append(hdul[f"{extname}ERR"].data)
            errs[key] = np.array(_errs)
            hdrs[key] = _hdrs

    single_diff = data[1] - data[2]
    single_sum = data[1] + data[2]
    single_err = np.hypot(errs[1], errs[2])
    comb_hdrs = []
    for i in range(single_diff.shape[0]):
        headers = (hdrs[1][i], hdrs[2][i])
        hdr = combine_frames_headers(headers)
        comb_hdrs.append(hdr)
    prim_hdr = combine_frames_headers(list(prim_hdrs.values()))
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(single_diff, header=prim_hdr),
            fits.ImageHDU(single_sum, header=prim_hdr, name="SUM"),
            fits.ImageHDU(single_err, header=prim_hdr, name="ERR"),
        ]
    )
    for i in range(single_diff.shape[0]):
        hdul.append(fits.ImageHDU(header=comb_hdrs[i]))

    hdul.writeto(outpath, overwrite=True)
    return hdul


def doublediff_images(paths, outpath: Path, force: bool = False) -> fits.HDUList:
    if not force and outpath.exists() and not any_file_newer(paths, outpath):
        return fits.open(outpath)
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
            _hdrs = []
            _errs = []
            for i in range(1, len(hdul), 2):
                extname = hdul[i].name
                _hdrs.append(hdul[i].header)
                _errs.append(hdul[f"{extname}ERR"].data)
            errs[key] = np.array(_errs)
            hdrs[key] = _hdrs

    double_diff = 0.5 * ((data[1, "A"] - data[2, "A"]) - (data[1, "B"] - data[2, "B"]))
    double_sum = 0.5 * ((data[1, "A"] + data[2, "A"]) + (data[1, "B"] + data[2, "B"]))
    double_err = np.sqrt(
        0.5 * (errs[1, "A"] ** 2 + errs[2, "A"] ** 2 + errs[1, "B"] ** 2 + errs[2, "B"] ** 2)
    )
    comb_hdrs = []
    for i in range(double_diff.shape[0]):
        headers = (hdrs[1, "A"][i], hdrs[2, "A"][i], hdrs[1, "B"][i], hdrs[2, "B"][i])
        hdr = combine_frames_headers(headers)
        comb_hdrs.append(hdr)
    prim_hdr = combine_frames_headers(list(prim_hdrs.values()))
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(double_diff, header=prim_hdr),
            fits.ImageHDU(double_sum, header=prim_hdr, name="SUM"),
            fits.ImageHDU(double_err, header=prim_hdr, name="ERR"),
        ]
    )
    for i in range(double_diff.shape[0]):
        hdul.append(fits.ImageHDU(header=comb_hdrs[i]))

    hdul.writeto(outpath, overwrite=True)
    return hdul
