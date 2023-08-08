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
    mbi_centers,
    window_slices,
)
from vampires_dpp.psf_models import fit_model
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
                offsets[i] = refoffset
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


def offset_modelfit(frame, inds, model, **kwargs):
    model_fit = fit_model(frame, inds, model, **kwargs)
    # normalize outputs
    ctr = np.array((model_fit["y"], model_fit["x"]))
    return ctr


def measure_offsets_file(
    cube, header, filename, method="peak", coronagraphic=False, force=False, **kwargs
):
    path, outpath = get_paths(filename, suffix="offsets", filetype=".csv", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath
    if "DPP_REF" in header:
        refidx = header["DPP_REF"]
    else:
        refidx = np.nanargmax(np.nanmax(cube, axis=(-2, -1)))
    if "MBI" in header["OBS-MOD"]:
        ctrs = mbi_centers(header["OBS-MOD"], header["U_CAMERA"], flip=True)
        offsets = []
        base_rad = kwargs.pop("radius")
        if coronagraphic:
            for field, ctr in zip(("F770", "F720", "F670", "F620"), ctrs[::-1]):
                radius = lamd_to_pixel(base_rad, field)
                kwargs["center"] = ctr
                offsets.append(
                    satellite_spot_offsets(
                        cube, method=method, refidx=refidx, radius=radius, **kwargs
                    )
                )
        else:
            for field, ctr in zip(("F770", "F720", "F670", "F620"), ctrs[::-1]):
                kwargs["center"] = ctr
                offsets.append(psf_offsets(cube, method=method, refidx=refidx, **kwargs))
        # flatten offsets
        # this puts them in (y1, x1, y2, x2, y3, x3, ...) order
        offsets = np.swapaxes(offsets, 0, 1)
    else:
        if coronagraphic:
            kwargs["radius"] = lamd_to_pixel(kwargs["radius"], header["FILTER01"])
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


def register_mbi_cube(cube, offsets, header=None, **kwargs):
    # reshape offsets into a single average
    stack = []
    cube = np.atleast_3d(cube)
    offsets = np.atleast_3d(offsets)
    for i in range(offsets.shape[1]):
        offs = offsets[:, i]
        mean_offsets = np.mean(offs.reshape(len(offs), -1, 2), axis=1)
        shifted = shift_cube(cube, -mean_offsets)
        stack.append(shifted)

    stack = np.swapaxes(stack, 0, 1)
    ## crop
    crop_size = 500
    rad_factor = (np.sqrt(2) - 1) * crop_size / 2
    window = np.ceil(rad_factor + 1) + crop_size
    cropinds = cutout_slice(stack, window=window)
    stack = np.flip(stack[..., cropinds[0], cropinds[1]], axis=1)
    return stack, header


def register_file(cube, header, offset_file, filename, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="aligned", **kwargs)

    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    cube, header = fits.getdata(
        path,
        header=True,
    )
    offsets = np.loadtxt(offset_file, delimiter=",")
    if offsets.shape[1] > 8:
        offsets = offsets.reshape(offsets.shape[0], -1, 8)
        shifted, header = register_mbi_cube(cube, offsets, header=header, **kwargs)
    else:
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
    cube, header, filename, metric_file=None, offsets_file=None, force: bool = False, q=0, **kwargs
) -> Path:
    path, outpath = get_paths(filename, suffix="collapsed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        frame, header = fits.getdata(path, header=True)
        return frame, header
    cube = np.atleast_3d(cube)
    ## Step 1: Frame select
    mask = np.ones(cube.shape[0], dtype=bool)
    if metric_file is not None:
        metrics = np.atleast_2d(np.loadtxt(metric_file, delimiter=","))
        metrics = np.median(metrics, axis=0)
        cube, header = frame_select_cube(cube, metrics, header=header, q=q, **kwargs)
        mask &= metrics >= np.quantile(metrics, q)

    ## Step 2: Register
    if offsets_file is not None:
        offsets = np.atleast_2d(np.loadtxt(offsets_file, delimiter=","))
        # mask offsets with selected metrics
        offsets_masked = offsets[mask]
        if "MBI" in header["OBS-MOD"].upper():
            offsets_masked = offsets_masked.reshape(offsets_masked.shape[0], 4, -1)
            cube, header = register_mbi_cube(cube, offsets_masked, header=header, **kwargs)
        else:
            # determine maximum padding, with sqrt(2)
            # for radial coverage
            rad_factor = (np.sqrt(2) - 1) * np.max(cube.shape[-2:]) / 2
            npad = np.ceil(rad_factor + 1)
            cube_padded, header = pad_cube(cube, int(npad), header=header)
            cube, header = register_cube(cube_padded, offsets_masked, header=header, **kwargs)
    elif "MBI" in header["OBS-MOD"].upper():
        # need to align MBI into cuts
        ctrs = mbi_centers(header["OBS-MOD"], header["U_CAMERA"], flip=True)
        fctr = frame_center(cube)

        offs = np.array([np.array(ctr) - np.array(fctr) for ctr in ctrs])
        offs = np.tile(offs, (len(cube), 1, 1))
        cube, header = register_mbi_cube(cube, offs, header=header, **kwargs)

    ## Step 3: Collapse
    frame, header = collapse_cube(cube, header=header, **kwargs)
    return frame, header
    # fits.writeto(outpath, frame, header=header, overwrite=True)
    # return outpath
