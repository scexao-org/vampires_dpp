from pathlib import Path

import numpy as np
from astropy.io import fits

from vampires_dpp.indexing import cutout_slice, lamd_to_pixel, window_slices
from vampires_dpp.util import get_paths


def measure_metric(cube, metric="l2norm", center=None, window=None, **kwargs):
    if window is not None:
        inds = cutout_slice(cube[0], center=center, window=window)
        view = cube[..., inds[0], inds[1]]
    else:
        view = cube

    match metric:
        case "peak" | "max":
            values = np.max(view, axis=(-2, -1))
        case "l2norm":
            values = np.mean(view**2, axis=(-2, -1))
        case "normvar":
            var = np.var(view, axis=(-2, -1))
            mean = np.mean(view, axis=(-2, -1))
            values = var / mean
    return values


def measure_satellite_spot_metrics(cube, metric="l2norm", **kwargs):
    slices = window_slices(cube[0], **kwargs)
    values = np.zeros_like(cube, shape=(cube.shape[0]))
    for sl in slices:
        view = cube[..., sl[0], sl[1]]
        match metric:
            case "peak" | "max":
                values += np.max(view, axis=(-2, -1))
            case "l2norm":
                values += np.mean(view**2, axis=(-2, -1))
            case "normvar":
                var = np.var(view, axis=(-2, -1))
                mean = np.mean(view, axis=(-2, -1))
                values += var / mean
    return values / len(slices)


def measure_metric_file(
    filename,
    metric="l2norm",
    coronagraphic=False,
    force=False,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="metric", filetype=".csv", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(path, header=True, output_verify="silentfix")
    if coronagraphic:
        kwargs["radius"] = lamd_to_pixel(kwargs["radius"], header["U_FILTER"])
        metrics = measure_satellite_spot_metrics(cube, metric=metric, **kwargs)
    else:
        metrics = measure_metric(cube, metric=metric, **kwargs)

    np.savetxt(outpath, metrics, delimiter=",")
    return outpath


def frame_select_file(
    filename,
    metricfile,
    q=0,
    force=False,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="selected", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(path, header=True, output_verify="silentfix")
    metrics = np.loadtxt(metricfile, delimiter=",")

    mask = metrics >= np.quantile(metrics, q)
    selected = cube[mask]

    header["VPP_REF"] = metrics[mask].argmax() + 1, "Index of frame with highest metric"

    fits.writeto(
        outpath, selected, header=header, overwrite=True, checksum=True, output_verify="silentfix"
    )
    return outpath
