from astropy.io import fits
from astropy.modeling import models, fitting
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.measure import centroid

from vampires_dpp.image_processing import frame_center, shift_cube
from vampires_dpp.satellite_spots import (
    cutout_slice,
    window_slices,
)


def satellite_spot_offsets(
    cube: ArrayLike,
    method="com",
    refmethod="peak",
    refidx=0,
    upsample_factor=1,
    center=None,
    background_subtract=True,
    **kwargs,
):

    slices = window_slices(cube[0], center=center, **kwargs)
    center = frame_center(cube)

    if background_subtract:
        background = model_background(cube, slices, center=center)
        cube = cube - background

    offsets = np.zeros((cube.shape[0], 2))

    if method == "com":
        for i, frame in enumerate(cube):
            for sl in slices:
                offsets[i] += offset_centroid(frame, sl)
            offsets[i] = offsets[i] / len(slices) - center
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
                offsets[i] = refoffset
                continue
            frame = cube[i]
            for sl in slices:
                refview = refframe[sl[0], sl[1]]
                view = frame[sl[0], sl[1]]
                dft_offset = phase_cross_correlation(
                    refview, view, return_error=False, upsample_factor=upsample_factor
                )
                offsets[i] += dft_offset
            offsets[i] = refoffset - offsets[i] / len(slices)
    elif method == "peak":
        for i, frame in enumerate(cube):
            for sl in slices:
                offsets[i] += offset_peak(frame, sl)
            offsets[i] = offsets[i] / len(slices) - center
    elif method in ("moffat", "airydisk", "gaussian"):
        fitter = fitting.LevMarLSQFitter()
        for i, frame in enumerate(cube):
            for sl in slices:
                offsets[i] += offset_modelfit(frame, sl, method, fitter)
            offsets[i] = offsets[i] / len(slices) - center

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
    offsets = np.zeros((cube.shape[0], 2))

    if method == "com":
        for i, frame in enumerate(cube):
            offsets[i] = offset_centroid(frame, inds) - center
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
                refview, view, return_error=False, **kwargs
            )
            offsets[i] = refoffset - dft_offset
    elif method == "peak":
        for i, frame in enumerate(cube):
            offsets[i] = offset_peak(frame, inds) - center
    elif method in ("moffat", "airydisk", "gaussian"):
        fitter = fitting.LevMarLSQFitter()
        for i, frame in enumerate(cube):
            offsets[i] = offset_modelfit(frame, inds, method, fitter) - center

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
        model = models.AiryDisk2D(
            amplitude=peak, x_0=view_center[1], y_0=view_center[0], radius=2
        )

    model_fit = fitter(model, x, y, view)

    # normalize outputs
    if method == "moffat" or method == "airydisk":
        offset = np.array((model_fit.y_0.value, model_fit.x_0.value))
    elif method == "gaussian":
        offset = np.array((model_fit.y_mean.value, model_fit.x_mean.value))

    offset[0] += inds[0].start
    offset[1] += inds[1].start

    return offset


def measure_offsets(
    filename, method="peak", coronagraphic=False, skip=False, output=None, **kwargs
):
    if output is None:
        path = Path(filename)
        output = path.with_stem(f"{path.stem}_offsets.csv")
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    cube, header = fits.getdata(filename, header=True)
    if "VPP_REF" in header:
        refidx = header["VPP_REF"]
    else:
        refidx = cube.max((-2, -1)).argmax()
    if coronagraphic:
        offsets = satellite_spot_offsets(cube, method=method, refidx=refidx, **kwargs)
    else:
        offsets = psf_offsets(cube, method=method, refidx=refidx, **kwargs)

    np.savetxt(output, offsets, delimiter=",")
    return output


def register_file(filename, offset_file, output=None, skip=False, **kwargs):
    if output is None:
        path = Path(filename)
        output = path.with_stem(f"{path.stem}_registered{path.suffix}")
    else:
        output = Path(output)

    if skip and output.is_file():
        return output

    cube, header = fits.getdata(filename, header=True)
    offsets = np.loadtxt(offset_file, delimeter=",")
    shifted = shift_cube(cube, -offsets)

    fits.writeto(output, shifted, header=header, overwrite=True)
    return output
