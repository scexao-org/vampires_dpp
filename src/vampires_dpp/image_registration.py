import logging
from typing import Annotated, Literal, TypeAlias

import numpy as np
from annotated_types import Gt
from astropy.io import fits
from astropy.nddata import Cutout2D
from photutils import centroids
from skimage.registration import phase_cross_correlation

from .image_processing import shift_frame
from .indexing import frame_center

__all__ = ("register_hdul",)

logger = logging.getLogger(__file__)

RegisterMethod: TypeAlias = Literal["peak", "com", "dft"]


def offset_dft(frame, inds, psf, *, upsample_factor: Annotated[int, Gt(0)] = 30):
    cutout = frame[inds]
    dft_offset = phase_cross_correlation(
        psf, cutout, return_error=False, upsample_factor=upsample_factor, normalization=None
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


def register_hdul(
    hdul: fits.HDUList, metrics, *, method: RegisterMethod = "dft", crop_width: int = 536, **kwargs
) -> fits.HDUList:
    # load centroids
    # reminder, this has shape (nframes, nlambda, npsfs, 2)
    # take mean along PSF axis
    centroids = np.mean(metrics[method], axis=2)
    center = frame_center(hdul[0].data)

    # determine maximum padding, with sqrt(2)
    # for radial coverage
    nframes, ny, nx = hdul[0].shape
    rad_factor = crop_width / np.sqrt(2)
    # round to nearest even number
    npad = int((rad_factor // 2) * 2)

    aligned_data = []
    aligned_err = []

    for tidx in range(aligned_data.shape[0]):
        frame = hdul[0].data[tidx]
        frame_err = hdul["ERR"].data[tidx]

        aligned_frames = []
        aligned_err_frames = []

        for wlidx in range(aligned_data.shape[1]):
            # determine offset for each field
            field_ctr = centroids[tidx, wlidx]
            offset = center - field_ctr
            # generate cutouts with crop width
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            cutout_err = Cutout2D(frame_err, field_ctr[::-1], size=crop_width, mode="partial")

            # pad and shift data
            frame_padded = np.pad(cutout.data, npad, constant_values=np.nan)
            shifted = shift_frame(frame_padded, offset, **kwargs)
            aligned_frames.append(shifted)

            # pad and shift error
            frame_err_padded = np.pad(cutout_err.data, npad, constant_values=np.nan)
            shifted_err = shift_frame(frame_err_padded, offset, **kwargs)
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
    info["hierarch DPP REGISTER METHOD"] = method, "Frame registration method"

    for hdu_idx in range(len(hdul)):
        output_hdul[hdu_idx].header |= info

    return output_hdul
