import itertools
import logging
from typing import Literal, TypeAlias

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from image_registration import chi2_shift
from photutils import centroids
from skimage import filters, transform

from vampires_dpp.headers import fix_header
from vampires_dpp.image_processing import shift_frame, warp_frame
from vampires_dpp.indexing import cutout_inds, frame_center, frame_radii, get_mbi_centers
from vampires_dpp.specphot.filters import determine_filterset_from_header
from vampires_dpp.synthpsf import create_synth_psf
from vampires_dpp.util import get_center

__all__ = ("register_hdul",)

logger = logging.getLogger(__file__)

RegisterMethod: TypeAlias = Literal["peak", "com", "dft"]


def offset_dft(frame, inds, psf):
    cutout = frame[inds]
    xoff, yoff = chi2_shift(psf, cutout, upsample_factor="auto", return_error=False)
    dft_offset = np.array((yoff, xoff))
    ctr = np.array(frame_center(psf)) + dft_offset
    # offset based on indices
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


def intersect_point(xs, ys):
    # sort points so we know how to pair into intersecting lines
    idxs = np.argsort(xs, axis=-1)
    xs = np.take_along_axis(xs, idxs, axis=-1)
    ys = np.take_along_axis(ys, idxs, axis=-1)

    # calculations are verbose, storing in temp variables
    a = xs[..., 0] * ys[..., 3] - ys[..., 0] * xs[..., 3]
    b = xs[..., 1] * ys[..., 2] - ys[..., 1] * xs[..., 2]
    d = (xs[..., 0] - xs[..., 3]) * (ys[..., 1] - ys[..., 2]) - (ys[..., 0] - ys[..., 3]) * (
        xs[..., 1] - xs[..., 2]
    )
    # calculate intersection from determinants of line segments
    px = (a * (xs[..., 1] - xs[..., 2]) - (xs[..., 0] - xs[..., 3]) * b) / d
    py = (a * (ys[..., 1] - ys[..., 2]) - (ys[..., 0] - ys[..., 3]) * b) / d

    return np.stack((px, py), axis=-1)


def get_centroids_from(metrics, input_key):
    cx = np.swapaxes(metrics[f"{input_key[:4]}x"], 0, 2)
    cy = np.swapaxes(metrics[f"{input_key[:4]}y"], 0, 2)
    # if there are values from multiple PSFs (e.g. satspots)
    # determine
    if cx.shape[1] == 4:
        centroids_xy = []
        for wlidx in range(cx.shape[-1]):
            centroids_xy.append(intersect_point(cx[..., wlidx], cy[..., wlidx]))
        centroids_xy = np.swapaxes(centroids_xy, 0, 1)
    else:
        centroids_xy = np.stack((cx[:, 0], cy[:, 0]), axis=-1)

    # stack so size is (Nframes, Nfields, x/y)
    return centroids_xy[..., ::-1]


def register_hdul(
    hdul: fits.HDUList,
    metrics,
    *,
    init_centroids=None,
    align: bool = True,
    method: RegisterMethod = "dft",
    crop_width: int = 536,
    reproject_tforms: None | dict[str, transform.SimilarityTransform],
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
    elif init_centroids is not None:
        init_centroids = np.mean(list(init_centroids.values()), axis=1)
        centroids = np.zeros((nframes, len(init_centroids), 2))
        for j in range(centroids.shape[1]):
            centroids[:, j] = get_center(
                hdul[0].data, init_centroids[j], hdul[0].header["U_CAMERA"]
            )
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

    if reproject_tforms is not None and hdul[0].header["U_CAMERA"] == 2:
        tforms = [reproject_tforms[field] for field in fields]
    else:
        tforms = None
    # determine maximum padding, with sqrt(2)
    # for radial coverage
    rad_factor = (crop_width / 2) * (np.sqrt(2) - 1)
    # round to nearest even number
    npad = int((rad_factor // 2) * 2)
    npix = crop_width + 2 * npad
    aligned_cube = np.empty((*centroids.shape[:2], npix, npix), dtype="f4")
    aligned_err_cube = np.empty((*centroids.shape[:2], npix, npix), dtype="f4")
    for tidx in range(centroids.shape[0]):
        frame = hdul[0].data[tidx]
        frame_err = hdul["ERR"].data[tidx]

        for wlidx in range(centroids.shape[1]):
            # determine offset for each field
            field_ctr = centroids[tidx, wlidx]
            # generate cutouts with crop width
            cutout = Cutout2D(frame, field_ctr[::-1], size=crop_width, mode="partial")
            cutout_err = Cutout2D(frame_err, field_ctr[::-1], size=crop_width, mode="partial")

            offset = field_ctr - cutout.position_original[::-1]

            # shift arrays, since it's subpixel don't worry about losing edges
            shifted = shift_frame(cutout.data, offset)
            shifted_err = shift_frame(cutout_err.data, offset)

            # if reprojecting, scale + rotate images
            if tforms is not None:
                rotmat = cv2.getRotationMatrix2D(
                    frame_center(shifted), np.rad2deg(tforms[wlidx].rotation), tforms[wlidx].scale
                )
                shifted = warp_frame(shifted, rotmat)
                shifted_err = warp_frame(shifted_err, rotmat)

            # pad output
            aligned_cube[tidx, wlidx] = np.pad(shifted, npad, constant_values=np.nan)
            aligned_err_cube[tidx, wlidx] = np.pad(shifted_err, npad, constant_values=np.nan)

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
    info["hierarch DPP ALIGN METH"] = method, "Frame alignment method"

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
        if len(offsets) == 4:
            ox, oy = intersect_point(offsets[:, 1], offsets[:, 0])
            offset = np.array((oy, ox))
        else:
            offset = offsets[0]
        data_cube[wl_idx] = shift_frame(frame, offset)
        err_cube[wl_idx] = shift_frame(err_cube[wl_idx], offset)

    info = fits.Header()
    info["hierarch DPP RECENTER"] = True, "Data was registered after coadding"
    info["hierarch DPP RECENTER METH"] = method, "DPP recentering registration method"

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


def find_square_peaks(frame, radius=3, max_counter=50):
    # initialize
    frame = frame.copy().astype("f4")  # because we're writing NaN's in place
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

        # as soon as we have 3 spots, start the list of triangle candidates
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
    data,
    camera: int,
    field: Literal["F610", "F670", "F720", "F760"],
    reduced: bool = False,
    cutout_size=536,
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
    return Cutout2D(data, (x, y), cutout_size, mode="partial")


def autocentroid_hdul(
    hdul: fits.HDUList,
    coronagraphic: bool = False,
    planetary: bool = False,
    psfs=None,
    crop_size=200,
    window_size=21,
    plot: bool = False,
):
    # collapse cubes, if need
    data = np.nanmedian(hdul[0].data, axis=0) if hdul[0].data.ndim == 3 else hdul[0].data
    # fix header
    header = fix_header(hdul[0].header)
    output = []
    fields = determine_filterset_from_header(header)
    if psfs is None:
        psfs = [create_synth_psf(header, filt, npix=window_size) for filt in fields]
    # for MBI data, divide input image into octants and account for
    # image flip between cam1 and cam2
    if "MBI" in header["OBS-MOD"]:
        reduced = "MBIR" in header["OBS-MOD"]
        cutouts = [
            get_mbi_cutout(data, header["U_CAMERA"], field, reduced=reduced) for field in fields
        ]
    # otherwise, just take the whole frame
    else:
        cutouts = [Cutout2D(data, frame_center(data)[::-1], data.shape[-1])]

    # for each frame (1, 3, or 4)
    for idx in range(len(fields)):
        cutout = cutouts[idx]
        # find centroid of image with square scaling to help bias towards PSF
        rough_ctr = centroids.centroid_com(cutout.data**2, mask=np.isnan(cutout.data))
        # take a large crop, large enough to see satellite spots plus misregistration
        rough_cutout = Cutout2D(
            cutout.data, rough_ctr, min(*cutout.shape[-2:], crop_size), mode="partial"
        )
        # high-pass filter the data with a large median-- note, this requires the data to have
        # a certain level of S/N or it will wipe out the satellite spots. Therefore it's only suggested
        # to run the autocentroid on a big stack of mean-combined data instead of individual frames
        filtered_cutout = rough_cutout.data - filters.median(rough_cutout.data, np.ones((9, 9)))
        # convolve high-pass filtered data with the PSF for better S/N (unsharp-mask-ish)
        filtered_cutout = convolve_fft(filtered_cutout, psfs[idx])

        # when using the coronagraph, find four maxima which form a square
        if coronagraphic:
            points = find_square_peaks(filtered_cutout)
            # refine with DFT
            for point_idx in range(len(points)):
                inds = Cutout2D(
                    filtered_cutout, points[point_idx][::-1], psfs[idx].shape
                ).slices_original
                points[point_idx] = offset_dft(filtered_cutout, inds, psfs[idx])
            # make sure to offset for indices
            rough_points = [rough_cutout.to_original_position(p[::-1]) for p in points]
            orig_points = [cutout.to_original_position(p) for p in rough_points]
        elif planetary:
            ellipse = fit_ellipse_to_image(cutout.data, psf=psfs[idx])
            ell_xy = ellipse[:2]
            orig_points = [
                [ell_xy[0] + cutout.origin_original[0], ell_xy[1] + cutout.origin_original[1]]
            ]
        else:
            # otherwise use DFT cross-correlation to find the PSF localized around peak index
            ctr = np.unravel_index(np.nanargmax(filtered_cutout), filtered_cutout.shape)
            inds = Cutout2D(
                filtered_cutout, ctr[::-1], size=psfs[idx].shape[-2:], mode="partial"
            ).slices_original
            points = [offset_dft(filtered_cutout, inds, psfs[idx])]
            # make sure to offset for indices
            rough_points = [rough_cutout.to_original_position(p[::-1]) for p in points]
            orig_points = [cutout.to_original_position(p) for p in rough_points]

        ## plotting
        if plot:
            fig, axs = plt.subplots(ncols=2)
            norm = simple_norm(cutout.data, stretch="sqrt")
            axs[0].imshow(cutout.data, origin="lower", cmap="magma", norm=norm)
            axs[0].scatter(*rough_ctr, marker="+", s=100, c="green")
            norm = None if coronagraphic else simple_norm(filtered_cutout, stretch="sqrt")

            if planetary:
                plot_ellipse(cutout.data, ellipse, ax=axs[1])
            elif points is not None:
                axs[1].imshow(filtered_cutout, origin="lower", cmap="magma", norm=norm)
                axs[1].scatter(*rough_cutout.position_cutout, marker="+", s=100, c="green")
                xs = np.array([p[1] for p in points])
                ys = np.array([p[0] for p in points])
                axs[1].scatter(xs, ys, marker=".", s=100, c="cyan")
                if len(xs) == 4:
                    # plot lines
                    idxs = np.argsort(xs)
                    xs = xs[idxs]
                    ys = ys[idxs]
                    axs[1].plot([xs[0], xs[3]], [ys[0], ys[3]], c="cyan")
                    axs[1].plot([xs[1], xs[2]], [ys[1], ys[2]], c="cyan")
                    px, py = intersect_point(xs, ys)
                    axs[1].scatter(px, py, marker="x", s=100, c="cyan")
                else:
                    axs[1].scatter(xs[0], ys[0], marker="x", s=100, c="cyan")

            axs[0].set_title("Starting cutout")
            axs[1].set_title("Centroided cutout")
            fig.suptitle(f"Field: {fields[idx]}")
            fig.tight_layout()

        output.append(orig_points)
    if plot:
        plt.show(block=True)
    return np.array(output)


def ellipse_func(xy, x0, y0, a, b, theta):
    """
    Equation of an ellipse.
    """
    x, y = xy
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))

    term1 = (((x - x0) * cos_theta + (y - y0) * sin_theta) / a) ** 2
    term2 = (((x - x0) * sin_theta - (y - y0) * cos_theta) / b) ** 2

    return term1 + term2


def fit_ellipse_to_image(data, psf=None):
    """
    Fit an ellipse to the bright region of the image (assuming Neptune).
    :param image: Input grayscale image of Neptune.
    :return: Parameters of the fitted ellipse (x0, y0, a, b, theta)
    """
    image = np.nan_to_num(data)
    # Threshold the image to create a binary mask
    # threshold = np.nanquantile(image, 0.7)
    gray_img = np.array((image - image.min()) / (image.max() - image.min()) * 255, dtype=np.uint8)
    if psf is None:
        blurred = cv2.medianBlur(gray_img, 5)
    else:
        gray_psf = np.array((psf - psf.min()) / (psf.max() - psf.min()) * 255, dtype=np.uint8)
        blurred = convolve_fft(gray_img, gray_psf)

    _, thresholded = cv2.threshold(
        blurred.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # frame_asu8 = image.astype(np.uint8)
    # thresholded = cv2.adaptiveThreshold(frame_asu8, np.nanmax(frame_asu8), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour corresponds to Neptune
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the largest contour
    ellipse = cv2.fitEllipse(largest_contour)
    (x0, y0), (a, b), theta = ellipse

    return x0, y0, a / 2, b / 2, theta


def plot_ellipse(image, ellipse_params, ax):
    """
    Plot the fitted ellipse on the image.
    :param image: Input image.
    :param ellipse_params: Parameters of the fitted ellipse (x0, y0, a, b, theta).
    """
    x0, y0, a, b, theta = ellipse_params
    y, x = np.mgrid[: image.shape[0], : image.shape[1]]
    fitted_ellipse = ellipse_func((x, y), x0, y0, a, b, theta)

    # Create a mask from the fitted ellipse
    ellipse_mask = fitted_ellipse <= 1

    # Plot the image and the fitted ellipse
    ax.imshow(image, cmap="magma", origin="lower")
    ax.contour(ellipse_mask, [0.5], colors="cyan")
    ax.scatter([x0], [y0], marker="x", s=100, color="cyan")  # Mark the center
