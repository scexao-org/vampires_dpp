from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from numpy.typing import ArrayLike
from skimage.measure import centroid
from skimage.registration import phase_cross_correlation

from vampires_dpp.image_processing import collapse_cube, shift_cube
from vampires_dpp.indexing import frame_center
from vampires_dpp.psf_models import fit_model
from vampires_dpp.util import get_paths


def model_background(cube, slices, center):
    # get long-exposure frame
    long_expo = np.median(cube, axis=0)
    # mask out satellite spots
    for sl in slices:
        long_expo[sl[0], sl[1]] = np.nan
    # fit model
    fitter = fitting.LevMarLSQFitter()
    model = models.Gaussian2D(
        x_mean=center[1],
        y_mean=center[0],
        x_stddev=10,
        y_stddev=10,
        amplitude=long_expo.max(),
    )
    Y, X = np.mgrid[0 : long_expo.shape[0], 0 : long_expo.shape[1]]
    bestfit_model = fitter(model, X, Y, long_expo)
    # return model evaluated over image grid
    return bestfit_model(X, Y)


def offset_dft(frame, inds, psf, upsample_factor=5):
    cutout = frame[inds]
    dft_offset = phase_cross_correlation(
        psf, cutout, return_error=False, upsample_factor=upsample_factor
    )
    ctr = np.array(frame_center(psf)) - dft_offset
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_centroid(frame, inds):
    """NaN-friendly centroid"""
    wy, wx = np.ogrid[inds[-2], inds[-1]]
    cutout = frame[inds]
    ctr = centroid(cutout)
    # mask = np.isfinite(cutout)
    # cy = np.sum(wy * cutout, where=mask)
    # cx = np.sum(wx * cutout, where=mask)
    # ctr = np.asarray((cy, cx)) / np.sum(cutout, where=mask)
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_peak(frame, inds):
    view = frame[inds]
    ctr = np.asarray(np.unravel_index(np.nanargmax(view), view.shape))
    # offset based on indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_modelfit(frame, inds, model, **kwargs):
    model_fit = fit_model(frame, inds, model, **kwargs)
    # normalize outputs
    ctr = np.array((model_fit["y"], model_fit["x"]))
    return ctr


def register_cube(cube, offsets, header=None, **kwargs):
    # reshape offsets into a single average
    shifted = shift_cube(cube, -offsets)
    return shifted, header


def register_file(filename, offset_file, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="aligned", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(
        path,
        header=True,
    )
    offsets = np.loadtxt(offset_file, delimiter=",")
    shifted, header = register_cube(cube, offsets, header=header, **kwargs)

    fits.writeto(outpath, shifted, header=header, overwrite=True)
    return outpath


def pad_cube(cube, pad_width: int, header=None, **pad_kwargs):
    new_shape = (cube.shape[0], cube.shape[1] + 2 * pad_width, cube.shape[2] + 2 * pad_width)
    output = np.empty_like(cube, shape=new_shape)

    for idx in range(cube.shape[0]):
        output[idx] = np.pad(cube[idx], pad_width, constant_values=np.nan)
    if header is not None:
        pass  # TODO
    return output, header


def frame_select_cube(cube, metrics, q=0, header=None, **kwargs):
    mask = metrics >= np.nanquantile(metrics, q)
    selected = cube[mask]
    if header is not None:
        header["DPP_REF"] = np.nanargmax(metrics[mask]) + 1, "Index of frame with highest metric"

    return selected, header


def lucky_image_file(
    hdu,
    outpath,
    method="median",
    register="com",
    frame_select=None,
    metric_file=None,
    force: bool = False,
    select_cutoff=0,
    **kwargs,
) -> Path:
    if (
        not force
        and outpath.is_file()
        and Path(metric_file).stat().st_mtime < outpath.stat().st_mtime
    ):
        with fits.open(outpath) as hdul:
            return hdul[0]

    cube, header = hdu.data, hdu.header
    # load metrics
    metrics = np.load(metric_file)
    mask = None
    if frame_select is not None and select_cutoff > 0:
        values = metrics[frame_select]
        if values.ndim == 2:
            values = np.mean(values, axis=0)
        mask = values > np.nanquantile(values, select_cutoff)
        cube, header = frame_select_cube(cube, values, select_cutoff, header=header)

    if register is not None:
        cx = metrics[f"{register}_x"]
        cy = metrics[f"{register}_y"]
        if cx.ndim == 2:
            cx = np.mean(cx, axis=0)
        if cy.ndim == 2:
            cy = np.mean(cy, axis=0)
        if mask is not None:
            cx = cx[mask]
            cy = cy[mask]
        centroids = np.column_stack((cy, cx))
        # print(centroids)
        offsets = centroids - frame_center(cube)
        # determine maximum padding, with sqrt(2)
        # for radial coverage
        rad_factor = (np.sqrt(2) - 1) * np.max(cube.shape[-2:]) / 2
        npad = np.ceil(rad_factor + 1).astype(int)
        cube_padded, header = pad_cube(cube, npad, header=header)
        cube, header = register_cube(cube_padded, offsets, header=header, **kwargs)

    ## Step 3: Collapse
    frame, header = collapse_cube(cube, method=method, header=header, **kwargs)
    hdu = fits.PrimaryHDU(frame, header=header)
    hdu.writeto(outpath, overwrite=True)
    return hdu
