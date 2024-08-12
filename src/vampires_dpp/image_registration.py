import numpy as np
from photutils import centroids
from skimage.registration import phase_cross_correlation

from .indexing import frame_center


def offset_dft(frame, inds, psf, *, upsample_factor):
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


def offset_centroids(frame, frame_err, inds, psf=None, dft_factor=30):
    """NaN-friendly centroids"""
    # wy, wx = np.ogrid[inds[-2], inds[-1]]
    cutout = frame[inds]
    # cutout_err = frame_err[inds] if frame_err is not None else None
    peak_yx = np.unravel_index(np.nanargmax(cutout), cutout.shape)
    com_xy = centroids.centroid_com(cutout)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     try:
    #         gauss_xy = centroids.centroid_2dg(cutout, error=cutout_err)
    #     except Exception:
    #         warnings.warn(f"Failed to fit Gaussian, using centroid values instead")
    #         gauss_xy = com_xy

    # offset based on indices
    offx = inds[-1].start
    offy = inds[-2].start
    ctrs = {
        "peak": np.array((peak_yx[0] + offy, peak_yx[1] + offx)),
        "com": np.array((com_xy[1] + offy, com_xy[0] + offx)),
        # "gauss": np.array((gauss_xy[1] + offy, gauss_xy[0] + offx)),
    }
    if psf is not None and dft_factor > 0:
        dft_off, _, _ = phase_cross_correlation(
            psf, cutout, upsample_factor=dft_factor, normalization=None
        )
        # need to update with center of frame
        ctr_off = np.array(frame_center(cutout)) - dft_off

        # fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(psf, origin="lower", cmap="magma")
        # axs[1].imshow(cutout, origin="lower", cmap="magma")
        # axs[1].scatter(ctr_off[-1], ctr_off[-2], marker='+', s=100, c="green")
        # axs[1].scatter(com_xy[0], com_xy[1], marker='x', s=100, c="blue")
        # plt.show(block=True)
        ctrs["dft"] = np.array((ctr_off[0] + offy, ctr_off[1] + offx))

    return ctrs
