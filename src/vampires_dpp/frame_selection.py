import numpy as np
from astropy.io import fits

from vampires_dpp.util import get_paths


def frame_select_cube(cube, metrics, q=0, header=None, **kwargs):
    mask = metrics >= np.quantile(metrics, q)
    selected = cube[mask]
    if header is not None:
        header["DPP_REF"] = metrics[mask].argmax() + 1, "Index of frame with highest metric"

    return selected, header


def frame_select_file(
    filename,
    metric_file,
    force=False,
    **kwargs,
):
    path, outpath = get_paths(filename, suffix="selected", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(
        path,
        header=True,
    )
    metrics = np.loadtxt(metric_file, delimiter=",")
    selected, header = frame_select_cube(cube, metrics, header=header, **kwargs)

    fits.writeto(outpath, selected, header=header, overwrite=True)
    return outpath
