from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from numpy.typing import ArrayLike
from skimage.registration import phase_cross_correlation

from vampires_dpp.frame_selection import frame_select_cube
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
    mask = np.isfinite(cutout)
    cy = np.sum(wy * cutout, where=mask)
    cx = np.sum(wx * cutout, where=mask)
    ctr = np.asarray((cy, cx)) / np.sum(cutout, where=mask)
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
    mean_offsets = np.mean(offsets.reshape(len(offsets), -1, 2), axis=1)
    shifted = shift_cube(cube, -mean_offsets)

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


def lucky_image_file(
    filename, metric_file=None, offsets_file=None, force: bool = False, q=0, **kwargs
) -> Path:
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(
        path,
        header=True,
    )

    ## Step 1: Frame select
    mask = np.ones(cube.shape[0], dtype=bool)
    if metric_file is not None:
        metrics = np.loadtxt(metric_file, delimiter=",")
        cube, header = frame_select_cube(cube, metrics, header=header, q=q, **kwargs)
        mask &= metrics >= np.quantile(metrics, q)

    ## Step 2: Register
    if offsets_file is not None:
        offsets = np.loadtxt(offsets_file, delimiter=",")
        # mask offsets with selected metrics
        offsets_masked = offsets[mask]
        # determine maximum padding, with sqrt(2)
        # for radial coverage
        rad_factor = (np.sqrt(2) - 1) * np.max(cube.shape[-2:]) / 2
        npad = np.ceil(rad_factor + 1)
        cube_padded, header = pad_cube(cube, int(npad), header=header)
        cube, header = register_cube(cube_padded, offsets_masked, header=header, **kwargs)

    ## Step 3: Collapse
    frame, header = collapse_cube(cube, header=header, **kwargs)

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath
