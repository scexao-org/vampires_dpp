import logging
from typing import Annotated, Literal, TypeAlias

import numpy as np
from annotated_types import Gt
from astropy.io import fits
from astropy.nddata import Cutout2D
from photutils import centroids
from skimage.registration import phase_cross_correlation

from .image_processing import shift_frame
from .indexing import cutout_inds, frame_center, get_mbi_centers

__all__ = ("register_hdul",)

logger = logging.getLogger(__file__)

RegisterMethod: TypeAlias = Literal["peak", "com", "dft"]


def offset_dft(frame, inds, psf, *, upsample_factor: Annotated[int, Gt(0)] = 30):
    cutout = frame[inds]
    dft_offset, _, _ = phase_cross_correlation(
        psf, cutout, upsample_factor=upsample_factor, normalization=None
    )
    ctr = np.array(frame_center(psf)) - dft_offset
    # plt.imshow(cutout, origin="lower", cmap="magma")
    # plt.imshow(psf, origin="lower", cmap="magma")
    # plt.scatter(ctr[-1], ctr[-2], marker='+', ms=100, c="green")
    # plt.show(block=True)
    # offset based on    indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    return ctr


def offset_peak_and_com(frame, inds):
    cutout = frame[inds]

    peak_yx = np.unravel_index(np.nanargmax(cutout), cutout.shape)
    com_xy = centroids.centroid_com(cutout)
    # offset based on indices
    offx = inds[-1].start
    offy = inds[-2].start
    ctrs = {
        "peak": np.array((peak_yx[0] + offy, peak_yx[1] + offx)),
        "com": np.array((com_xy[1] + offy, com_xy[0] + offx)),
    }
    return ctrs


def get_centroids_from(metrics, input_key):
    cx = np.swapaxes(metrics[f"{input_key[:4]}x"], 0, 2)
    cy = np.swapaxes(metrics[f"{input_key[:4]}y"], 0, 2)
    # if there are values from multiple PSFs (e.g. satspots)
    # take the mean centroid of each PSF
    if cx.ndim == 3:
        cx = np.mean(cx, axis=1)
    if cy.ndim == 3:
        cy = np.mean(cy, axis=1)

    # stack so size is (Nframes, Nfields, x/y)
    centroids = np.stack((cy, cx), axis=-1)
    return centroids


def register_hdul(
    hdul: fits.HDUList,
    metrics,
    *,
    align: bool = True,
    method: RegisterMethod = "dft",
    crop_width: int = 536,
) -> fits.HDUList:
    # load centroids
    # reminder, this has shape (nframes, nlambda, npsfs, 2)
    # take mean along PSF axis
    nframes, ny, nx = hdul[0].shape
    center = frame_center(hdul[0].data)
    header = hdul[0].header
    if align:
        centroids = get_centroids_from(metrics, method)
    elif "MBIR" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(hdul[0].data, reduced=True)
        centroids = np.zeros((nframes, 3, 2))
        fields = ("F670", "F720", "F760")
        for idx, key in enumerate(fields):
            centroids[:, idx] = ctr_dict[key]
    elif "MBI" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(hdul[0].data)
        centroids = np.zeros((nframes, 4, 2))
        fields = ("F610", "F670", "F720", "F760")
        for idx, key in enumerate(fields):
            centroids[:, idx] = ctr_dict[key]
    else:
        centroids = np.zeros((nframes, 1, 2))
        centroids[:] = center

    # determine maximum padding, with sqrt(2)
    # for radial coverage
    rad_factor = crop_width / np.sqrt(2)
    # round to nearest even number
    npad = int((rad_factor // 2) * 2)

    aligned_data = []
    aligned_err = []

    for tidx in range(centroids.shape[0]):
        frame = hdul[0].data[tidx]
        frame_err = hdul["ERR"].data[tidx]

        aligned_frames = []
        aligned_err_frames = []

        for wlidx in range(centroids.shape[1]):
            # determine offset for each field
            field_ctr = centroids[tidx, wlidx]
            offset = center - field_ctr
            # generate cutouts with crop width
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            cutout_err = Cutout2D(frame_err, field_ctr[::-1], size=crop_width, mode="partial")

            # pad and shift data
            frame_padded = np.pad(cutout.data, npad, constant_values=np.nan)
            shifted = shift_frame(frame_padded, offset)
            aligned_frames.append(shifted)

            # pad and shift error
            frame_err_padded = np.pad(cutout_err.data, npad, constant_values=np.nan)
            shifted_err = shift_frame(frame_err_padded, offset)
            aligned_err_frames.append(shifted_err)

        aligned_data.append(aligned_frames)
        aligned_err.append(aligned_err_frames)

    aligned_cube = np.array(aligned_data)
    aligned_err_cube = np.array(aligned_err)

    # generate output HDUList
    output_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(aligned_cube, header=hdul[0].header),
            fits.ImageHDU(aligned_err_cube, header=hdul["ERR"].header, name="ERR"),
            *hdul[2:],
        ]
    )

    # update header info
    info = fits.Header()
    info["hierarch DPP ALIGN METHOD"] = method, "Frame alignment method"

    for hdu_idx in range(len(hdul)):
        output_hdul[hdu_idx].header.update(info)

    return output_hdul


def recenter_hdul(
    hdul: fits.HDUList,
    *,
    method: RegisterMethod = "dft",
    window_size: int = 30,
    dft_factor: int = 30,
    psfs: None = None,
):
    data_cube = hdul[0].data
    err_cube = hdul["ERR"].data
    field_center = frame_center(data_cube)
    inds = cutout_inds(data_cube, center=field_center, window=window_size)
    ## Measure centroid
    for wl_idx in range(data_cube.shape[0]):
        frame = data_cube[wl_idx]
        match method:
            case "com" | "peak":
                center = offset_peak_and_com(frame, inds)[method]
            case "dft":
                assert psfs is not None
                center = offset_dft(frame, inds, psf=psfs[wl_idx])

        offset = center - field_center
        data_cube[wl_idx] = shift_frame(frame, offset)
        err_cube[wl_idx] = shift_frame(err_cube[wl_idx], offset)

    info = fits.Header()
    info["hierarch DPP RECENTER"] = True, "Data was registered after coadding"
    info["hierarch DPP RECENTER METHOD"] = method, "DPP recentering registration method"

    for hdu in hdul:
        hdu.header.update(info)

    return hdul
