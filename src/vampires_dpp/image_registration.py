import itertools
import logging
from typing import Literal, TypeAlias

import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.nddata import Cutout2D
from image_registration import chi2_shift
from photutils import centroids
from skimage import filters

from vampires_dpp.indexing import frame_radii
from vampires_dpp.specphot.filters import determine_filterset_from_header
from vampires_dpp.synthpsf import create_synth_psf

from .image_processing import shift_frame
from .indexing import cutout_inds, frame_center, get_mbi_centers

__all__ = ("register_hdul",)

logger = logging.getLogger(__file__)

RegisterMethod: TypeAlias = Literal["peak", "com", "dft"]


def offset_dft(frame, inds, psf):
    cutout = frame[inds]
    xoff, yoff, exoff, eyoff = chi2_shift(psf, cutout, upsample_factor="auto", return_error=True)
    dft_offset = np.array((yoff, xoff))
    ctr = np.array(frame_center(psf)) + dft_offset
    # offset based on    indices
    ctr[-2] += inds[-2].start
    ctr[-1] += inds[-1].start
    # plt.imshow(frame, origin="lower", cmap="magma")
    # # plt.imshow(psf, origin="lower", cmap="magma")
    # plt.scatter(ctr[-1], ctr[-2], marker='+', s=100, c="green")
    # plt.show(block=True)
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


def get_intersection(xs, ys):
    idxs = np.argsort(xs, axis=1)
    xs = np.take_along_axis(xs, idxs, axis=1)
    ys = np.take_along_axis(ys, idxs, axis=1)

    a = xs[:, 0] * ys[:, 3] - ys[:, 0] * xs[:, 3]
    b = xs[:, 1] * ys[:, 2] - ys[:, 1] * xs[:, 2]
    d = (xs[:, 0] - xs[:, 3]) * (ys[:, 1] - ys[:, 2]) - (ys[:, 0] - ys[:, 3]) * (
        xs[:, 1] - xs[:, 2]
    )
    px = (a * (xs[:, 1] - xs[:, 2]) - (xs[:, 0] - xs[:, 3]) * b) / d
    py = (a * (ys[:, 1] - ys[:, 2]) - (ys[:, 0] - ys[:, 3]) * b) / d

    return px, py


def get_centroids_from(metrics, input_key):
    cx = np.swapaxes(metrics[f"{input_key[:4]}x"], 0, 2)
    cy = np.swapaxes(metrics[f"{input_key[:4]}y"], 0, 2)
    # if there are values from multiple PSFs (e.g. satspots)
    # determine
    if cx.ndim == 3:
        cx, cy = get_intersection(cx, cy)

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
    fields = determine_filterset_from_header(header)
    if align:
        centroids = get_centroids_from(metrics, method)
    elif "MBIR" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(hdul[0].data, reduced=True)
        centroids = np.zeros((nframes, 3, 2))
        for idx, key in enumerate(fields):
            centroids[:, idx] = ctr_dict[key]
    elif "MBI" in header["OBS-MOD"]:
        ctr_dict = get_mbi_centers(hdul[0].data)
        centroids = np.zeros((nframes, 4, 2))
        for idx, key in enumerate(fields):
            centroids[:, idx] = ctr_dict[key]
    else:
        centroids = np.zeros((nframes, 1, 2))
        centroids[:] = center

    # determine maximum padding, with sqrt(2)
    # for radial coverage
    rad_factor = (crop_width / 2) * (np.sqrt(2) - 1)
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
            # generate cutouts with crop width
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            cutout_err = Cutout2D(frame_err, field_ctr[::-1], size=crop_width, mode="partial")

            offset = cutout.center_original[::-1] - field_ctr

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
        ]
    )
    for wlidx in range(centroids.shape[1]):
        hdr = header.copy()
        hdr["FIELD"] = fields[wlidx]
        output_hdul.append(fits.ImageHDU(header=hdr, name=hdr["FIELD"]))

    # update header info
    info = fits.Header()
    info["hierarch DPP ALIGN METHOD"] = method, "Frame alignment method"

    for hdu_idx in range(len(hdul)):
        output_hdul[hdu_idx].header.update(info)

    return output_hdul


def recenter_hdul(
    hdul: fits.HDUList,
    window_centers,
    *,
    method: RegisterMethod = "dft",
    window_size: int = 21,
    psfs: None = None,
):
    data_cube = hdul[0].data
    err_cube = hdul["ERR"].data
    # General strategy: use window centers to know where to search for PSFs
    # cast window_centers to array
    window_array = np.array(list(window_centers.values()))
    window_offsets = window_array - np.mean(window_array, axis=1, keepdims=True)
    field_center = frame_center(data_cube)
    ## Measure centroid
    for wl_idx in range(data_cube.shape[0]):
        frame = data_cube[wl_idx]
        offsets = []
        for offset in window_offsets[wl_idx]:
            inds = cutout_inds(frame, center=field_center + offset, window=window_size)
            match method:
                case "com" | "peak":
                    center = offset_peak_and_com(frame, inds)[method]
                case "dft":
                    assert psfs is not None
                    center = offset_dft(frame, inds, psf=psfs[wl_idx])

            offsets.append(field_center - center)
        offsets = np.array(offsets)
        ox, oy = get_intersection(offsets[None, :, 1], offsets[None, :, 0])
        offset = np.array((oy[0], ox[0]))
        data_cube[wl_idx] = shift_frame(frame, offset)
        err_cube[wl_idx] = shift_frame(err_cube[wl_idx], offset)

    info = fits.Header()
    info["hierarch DPP RECENTER"] = True, "Data was registered after coadding"
    info["hierarch DPP RECENTER METHOD"] = method, "DPP recentering registration method"

    for hdu in hdul:
        hdu.header.update(info)

    return hdul


def euclidean_distance(p1, p2):
    """Calculate distance between two points p1 and p2."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_right_triangles(vertices: list[tuple[int, int]], radius=3):
    """Find all sets of vertices that form a right triangle."""
    right_triangles = []

    # Check all combinations of three vertices
    for triplet in itertools.combinations(vertices, 3):
        p1, p2, p3 = triplet
        # Calculate squared distances
        dist_12 = euclidean_distance(p1, p2)
        dist_23 = euclidean_distance(p2, p3)
        dist_31 = euclidean_distance(p3, p1)

        # Sort distances to identify the longest one
        distances = np.sort((dist_12, dist_23, dist_31))

        if (
            np.isclose(distances[0], distances[1], atol=2 * radius)
            and np.isclose(distances[2] / np.mean(distances[:2]), np.sqrt(2), atol=5e-2)
            and np.all(distances > 38)
        ):
            right_triangles.append(triplet)

    return right_triangles


def test_triangle_plus_one_is_square(
    triangle: list[tuple[int, int]], point: tuple[int, int], radius: float = 3
):
    """Given a set of vertices that form a right triangles, see if the test point forms a square"""
    if point in triangle:
        return False
    vertices = [*triangle, point]

    distances = [
        euclidean_distance(pair[0], pair[1]) for pair in itertools.combinations(vertices, 2)
    ]
    distances.sort()

    diff1s = np.diff(distances[:4])
    diff2s = np.diff(distances[4:])
    return (
        np.isclose(np.mean(diff1s), 0, atol=2 * radius)
        and np.isclose(np.mean(diff2s), 0, atol=2 * radius)
        and np.isclose(np.mean(distances[4:]) / np.mean(distances[:4]), np.sqrt(2), atol=5e-2)
    )


def find_square_satspots(frame, radius=3, max_counter=50):
    # initialize
    # frame = frame.copy()
    max_ind = np.nanargmax(frame)

    counter = 0
    locs = []  # locs is a list of cartesian indices
    # tricombs is a ??
    tricombs = []

    # data structures for memoization
    while counter < max_counter:
        # grab newest spot candidate and add to list of locations
        inds = np.unravel_index(max_ind, frame.shape)
        locs.append(inds)
        frame = mask_circle(frame, inds, radius)
        # if we have at least four locations and some 3-spot candidates we can start to evaluate sets
        if len(locs) >= 4 and len(tricombs) >= 1:
            # generate all combinations of pairs of two coordinates, without repetition
            for last_spot in locs:
                for tricomb in reversed(tricombs):
                    if test_triangle_plus_one_is_square(tricomb, last_spot):
                        return [*tricomb, last_spot]

        # as soon as we have 3 spots, start the list of good candidates
        if len(locs) >= 3:
            # generate all combinations of pairs of two coordinates, without repetition
            triangle_sets = find_right_triangles(locs)
            tricombs.extend(triangle_sets)
        # setup next iteration
        max_ind = np.nanargmax(frame)
        counter += 1

    msg = "Could not fit satelliate spots!"
    print(msg)
    # raise RuntimeError(msg)
    return None


def mask_circle(frame, inds, radius):
    rs = frame_radii(frame, center=inds)
    mask = rs <= radius
    frame[mask] = np.nan
    return frame


def get_mbi_cutout(
    data, camera: int, field: Literal["F610", "F670", "F720", "F760"], reduced: bool = False
):
    hy, hx = frame_center(data)
    # use cam2 as reference
    match field:
        case "F610":
            x = hx * 0.25
            y = hy * 1.5
        case "F670":
            x = hx * 0.25
            y = hy * 0.5
        case "F720":
            x = hx * 0.75
            y = hy * 0.5
        case "F760":
            x = hx * 1.75
            y = hy * 0.5
        case _:
            msg = f"Invalid MBI field {field}"
            raise ValueError(msg)
    if reduced:
        y *= 2
    # flip y axis for cam 1 indices
    if camera == 1:
        y = data.shape[-2] - y
    return Cutout2D(data, (x, y), 300, mode="partial")


def autocentroid_hdul(hdul: fits.HDUList, coronagraphic: bool = False, psfs=None, window_size=21):
    data = np.nanmedian(hdul[0].data, axis=0)

    if psfs is None:
        if "MBI" in hdul[0].header["OBS-MOD"]:
            psfs = [
                create_synth_psf(hdul[0].header, filt, npix=20)
                for filt in ("F610", "F670", "F720", "F760")
            ]
        else:
            psfs = [create_synth_psf(hdul[0].header, npix=20)]
        if "MBIR" in hdul[0].header["OBS-MOD"]:
            del psfs["F610"]

    if hdul[0].header["OBS-MOD"].endswith("MBI"):
        cutouts = [
            get_mbi_cutout(data, hdul[0].header["U_CAMERA"], field)
            for field in ["F610", "F670", "F720", "F760"]
        ]
    elif hdul[0].header["OBS-MOD"].endswith("MBIR"):
        cutouts = [
            get_mbi_cutout(data, hdul[0].header["U_CAMERA"], field)
            for field in ["F670", "F720", "F760"]
        ]
    else:
        cutouts = [Cutout2D(data, frame_center(data), data.shape[-1], mode="partial")]

    for cutout, psf in zip(cutouts, psfs, strict=True):
        rough_ctr = centroids.centroid_com(cutout.data)
        rough_cutout = Cutout2D(cutout.data, rough_ctr, 200, mode="partial")
        filtered_cutout = rough_cutout.data - filters.median(rough_cutout.data, np.ones((9, 9)))
        filtered_cutout = convolve_fft(filtered_cutout, psf)

        # if coronagraphic:
        #     points = find_square_satspots(filtered_cutout)
        # else:
        #     points = None # TODO!
        # fig, axs = plt.subplots(ncols=3)
        # axs[0].imshow(cutout.data, origin="lower", cmap="magma")
        # axs[0].scatter(*rough_ctr[::-1], marker="+", s=100, c="green")
        # axs[1].imshow(filtered_cutout, origin="lower", cmap="magma")

        # axs[2].imshow(filtered_cutout, origin="lower", cmap="magma")
        # if points is not None:
        #     xs = [p[1] for p in points]
        #     ys = [p[0] for p in points]
        # # plt.imshow(psf, origin="lower", cmap="magma")
        # axs[1].scatter(*rough_cutout.position_cutout, marker="+", s=100, c="green")
        # axs[2].scatter(*rough_cutout.position_cutout, marker="+", s=100, c="green")
        # if points is not None:
        #     axs[1].scatter(xs, ys, marker="x", s=100, c="cyan")
        #     axs[1].scatter(np.mean(xs), np.mean(ys), marker="x", s=100, c="cyan")
        #     axs[2].scatter(xs, ys, marker="x", s=100, c="cyan")
        #     axs[2].scatter(np.mean(xs), np.mean(ys), marker="x", s=100, c="cyan")
        # plt.show(block=True)
