from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from numpy.typing import ArrayLike
from skimage.measure import centroid
from skimage.registration import phase_cross_correlation

from vampires_dpp.frame_selection import frame_select_cube
from vampires_dpp.image_processing import collapse_cube, shift_cube
from vampires_dpp.indexing import (
    cutout_slice,
    frame_center,
    lamd_to_pixel,
    window_slices,
)
from vampires_dpp.util import get_paths


def satellite_spot_offsets(
    cube: ArrayLike,
    method="com",
    refmethod="peak",
    refidx=0,
    upsample_factor=1,
    center=None,
    smooth: bool = False,
    **kwargs,
):
    slices = window_slices(cube[0], center=center, **kwargs)
    center = frame_center(cube)

    if smooth:
        for i in range(cube.shape[0]):
            cube[i] = cv2.GaussianBlur(cube[i], (0, 0), 3)

    offsets = np.zeros((cube.shape[0], len(slices), 2))

    if method == "com":
        for i, frame in enumerate(cube):
            for j, sl in enumerate(slices):
                offsets[i, j] = offset_centroid(frame, sl) - center
    elif method == "dft":
        refframe = cube[refidx]
        # measure peak index from each
        refshift = 0
        for sl in slices:
            if refmethod == "com":
                refshift += offset_centroid(refframe, sl)
            elif refmethod == "peak":
                refshift += offset_peak(refframe, sl)
        refoffset = refshift / len(slices) - center

        for i in range(cube.shape[0]):
            if i == refidx:
                offsets[i, j] = refoffset
                continue
            frame = cube[i]
            for j, sl in enumerate(slices):
                refview = refframe[sl[0], sl[1]]
                view = frame[sl[0], sl[1]]
                dft_offset = phase_cross_correlation(
                    refview, view, return_error=False, upsample_factor=upsample_factor
                )
                offsets[i, j] = refoffset - dft_offset
    elif method == "peak":
        for i, frame in enumerate(cube):
            for j, sl in enumerate(slices):
                offsets[i, j] = offset_peak(frame, sl) - center
    elif method in ("moffat", "airydisk", "gaussian"):
        fitter = fitting.LevMarLSQFitter()
        for i, frame in enumerate(cube):
            for j, sl in enumerate(slices):
                offsets[i, j] += offset_modelfit(frame, sl, method, fitter) - center

    return offsets


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


def psf_offsets(
    cube: ArrayLike,
    method="peak",
    refmethod="peak",
    refidx=0,
    center=None,
    window=None,
    upsample_factor=1,
    **kwargs,
):
    if window is not None:
        inds = cutout_slice(cube[0], center=center, window=window)
    else:
        inds = (
            slice(0, cube.shape[-2]),
            slice(0, cube.shape[-1]),
        )
    center = frame_center(cube)
    offsets = np.zeros((cube.shape[0], 1, 2))

    if method == "com":
        for i, frame in enumerate(cube):
            offsets[i, 0] = offset_centroid(frame, inds) - center
    elif method == "dft":
        refframe = cube[refidx]
        if refmethod == "com":
            refoffset = offset_centroid(refframe, inds) - center
        elif refmethod == "peak":
            refoffset = offset_peak(refframe, inds) - center
        refview = refframe[inds[0], inds[1]]

        for i, frame in enumerate(cube):
            if i == refidx:
                offsets[i] = refoffset
                continue
            view = frame[inds[0], inds[1]]
            dft_offset = phase_cross_correlation(
                refview, view, return_error=False, upsample_factor=upsample_factor
            )
            offsets[i, 0] = refoffset - dft_offset
    elif method == "peak":
        for i, frame in enumerate(cube):
            offsets[i, 0] = offset_peak(frame, inds) - center
    elif method in ("moffat", "airydisk", "gaussian"):
        fitter = fitting.LevMarLSQFitter()
        for i, frame in enumerate(cube):
            offsets[i, 0] = offset_modelfit(frame, inds, method, fitter) - center

    return offsets


def offset_centroid(frame, inds):
    view = frame[inds[0], inds[1]]
    ctr = centroid(view)
    # offset based on indices
    ctr[0] += inds[0].start
    ctr[1] += inds[1].start
    return ctr


def offset_peak(frame, inds):
    view = frame[inds]
    ctr = np.asarray(np.unravel_index(view.argmax(), view.shape))
    # offset based on indices
    ctr[0] += inds[0].start
    ctr[1] += inds[1].start
    return ctr


def offset_modelfit(frame, inds, method, fitter=fitting.LevMarLSQFitter()):
    view = frame[inds[0], inds[1]]
    y, x = np.mgrid[0 : view.shape[0], 0 : view.shape[1]]
    view_center = frame_center(view)
    peak = np.quantile(view.ravel(), 0.9)
    if method == "moffat":
        # bounds = {"amplitude": (0, peak), "gamma": (1, 15), "alpha": }
        model = models.Moffat2D(
            amplitude=peak, x_0=view_center[1], y_0=view_center[0], gamma=2, alpha=2
        )
    elif method == "gaussian":
        model = models.Gaussian2D(
            amplitude=peak,
            x_mean=view_center[1],
            y_mean=view_center[0],
            x_stddev=2,
            y_stddev=2,
        )
    elif method == "airydisk":
        model = models.AiryDisk2D(amplitude=peak, x_0=view_center[1], y_0=view_center[0], radius=2)

    model_fit = fitter(model, x, y, view)

    # normalize outputs
    if method == "moffat" or method == "airydisk":
        offset = np.array((model_fit.y_0.value, model_fit.x_0.value))
    elif method == "gaussian":
        offset = np.array((model_fit.y_mean.value, model_fit.x_mean.value))

    offset[0] += inds[0].start
    offset[1] += inds[1].start

    return offset


def measure_offsets_file(filename, method="peak", coronagraphic=False, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="offsets", filetype=".csv", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    cube, header = fits.getdata(
        path,
        header=True,
    )
    if "DPP_REF" in header:
        refidx = header["DPP_REF"]
    else:
        refidx = np.nanargmax(np.nanmax(cube, axis=(-2, -1)))
    if coronagraphic:
        kwargs["radius"] = lamd_to_pixel(kwargs["radius"], header["U_FILTER"])
        offsets = satellite_spot_offsets(cube, method=method, refidx=refidx, **kwargs)
    else:
        offsets = psf_offsets(cube, method=method, refidx=refidx, **kwargs)

    # flatten offsets
    # this puts them in (y1, x1, y2, x2, y3, x3, ...) order
    offsets_flat = offsets.reshape(len(offsets), -1)

    np.savetxt(outpath, offsets_flat, delimiter=",")
    return outpath


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
        cube, header = register_cube(cube, offsets_masked, header=header, **kwargs)

    ## Step 3: Collapse
    frame, header = collapse_cube(cube, header=header, **kwargs)

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath
