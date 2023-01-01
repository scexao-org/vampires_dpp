from astropy.io import fits
import numpy as np
from pathlib import Path

from vampires_dpp.indexing import cutout_slice, window_slices


def measure_metric(cube, metric="l2norm", center=None, window=None):
    if window is not None:
        inds = cutout_slice(cube[0], center=center, window=window)
        view = cube[..., inds[0], inds[1]]
    else:
        view = cube

    if metric == "max":
        values = np.max(view, axis=(-2, -1))
    elif metric == "l2norm":
        values = np.mean(view**2, axis=(-2, -1))
    elif metric == "normvar":
        var = np.var(view, axis=(-2, -1))
        mean = np.mean(view, axis=(-2, -1))
        values = var / mean
    return values


def measure_satellite_spot_metrics(cube, metric="l2norm", **kwargs):
    slices = window_slices(cube[0], **kwargs)
    values = np.zeros_like(cube, shape=(cube.shape[0]))
    for sl in slices:
        view = cube[..., sl[0], sl[1]]
        if metric == "max":
            values += np.max(view, axis=(-2, -1))
        elif metric == "l2norm":
            values += np.mean(view**2, axis=(-2, -1))
        elif metric == "normvar":
            var = np.var(view, axis=(-2, -1))
            mean = np.mean(view, axis=(-2, -1))
            values += var / mean
    return values / len(slices)


def measure_metric_file(
    filename,
    metric="l2norm",
    coronagraphic=False,
    skip=False,
    output=None,
    **kwargs,
):
    if output is None:
        path = Path(filename)
        output = path.with_name(f"{path.stem}_metric.csv")
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    cube, header = fits.getdata(filename, header=True)
    if coronagraphic:
        metrics = measure_satellite_spot_metrics(cube, metric=metric, **kwargs)
    else:
        metrics = measure_metric(cube, metric=metric, **kwargs)

    np.savetxt(output, metrics, delimiter=",")
    return output


def frame_select_file(
    filename,
    metricfile,
    q=0,
    skip=False,
    output=None,
    **kwargs,
):
    if output is None:
        path = Path(filename)
        output = path.with_name(f"{path.stem}_selected{path.suffix}")
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    cube, header = fits.getdata(filename, header=True)
    metrics = np.loadtxt(metricfile, delimiter=",")

    mask = metrics >= np.quantile(metrics, q)
    selected = cube[mask]

    header["VPP_REF"] = metrics[mask].argmax() + 1, "Index of frame with highest metric"

    fits.writeto(output, selected, header=header, overwrite=True)
    return output
