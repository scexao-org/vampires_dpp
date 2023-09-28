import numpy as np
from astropy.io import fits

from vampires_dpp.indexing import (
    cutout_slice,
    lamd_to_pixel,
    mbi_centers,
    window_slices,
)
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
    cube,
    header,
    filename,
    metric="l2norm",
    coronagraphic=False,
    force=False,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="metric", filetype=".csv", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    if coronagraphic:
        base_rad = kwargs.pop("radius")

    if "MBI" in header["OBS-MOD"]:
        ctrs = mbi_centers(header["OBS-MOD"], header["U_CAMERA"], flip=True)
        metrics = []
        if coronagraphic:
            for field, ctr in zip(("F760", "F720", "F670", "F610"), reversed(ctrs)):
                radius = lamd_to_pixel(base_rad, field)
                kwargs["center"] = ctr
                metrics.append(
                    measure_satellite_spot_metrics(cube, metric=metric, radius=radius, **kwargs)
                )
        else:
            for field, ctr in zip(("F760", "F720", "F670", "F610"), reversed(ctrs)):
                radius = lamd_to_pixel(base_rad, field)
                kwargs["center"] = ctr
                metrics.append(measure_metric(cube, metric=metric, radius=radius, **kwargs))
    else:
        if coronagraphic:
            radius = lamd_to_pixel(base_rad, header["FILTER01"])
            metrics = measure_satellite_spot_metrics(cube, metric=metric, **kwargs)
        else:
            metrics = measure_metric(cube, metric=metric, **kwargs)

    np.savetxt(outpath, metrics, delimiter=",")
    return outpath


def frame_select_cube(cube, metrics, q=0, header=None, **kwargs):
    mask = metrics >= np.quantile(metrics, q)
    selected = cube[mask]
    if header is not None:
        header["DPP_REF"] = metrics[mask].argmax() + 1, "Index of frame with highest metric"

    return selected, header


def frame_select_file(
    cube,
    header,
    metric_file,
    filename,
    force=False,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="selected", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    metrics = np.atleast_2d(np.loadtxt(metric_file, delimiter=","))
    metrics = np.median(metrics, axis=0)
    selected, header = frame_select_cube(cube, metrics, header=header, **kwargs)

    fits.writeto(outpath, selected, header=header, overwrite=True)
    return outpath
