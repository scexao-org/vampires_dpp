import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import click
import numpy as np
import tomli_w
from numpy.typing import NDArray
from skimage import filters

from vampires_dpp.image_processing import collapse_frames_files
from vampires_dpp.image_registration import offset_centroids
from vampires_dpp.indexing import cutout_inds, frame_center, frame_radii
from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.util import load_fits

from . import logger

__all__ = "centroid"


def get_psf_centroids_manual(cams: Sequence[int], npsfs: int) -> dict[str, NDArray]:
    centroids = {f"cam{cam:.0f}": np.empty((1, npsfs, 2), dtype="f4") for cam in cams}
    for key, cent_arr in centroids.items():
        click.echo(f"Enter comma-separated x, y centroids for {click.style(key, bold=True)}:")
        for i in range(npsfs):
            response = click.prompt(f" - PSF: {i} (x, y)")
            cent_arr[0, i] = [float(r) for r in response.split(",")]
    # output is (nspfs, (x, y)) with +1 in x and y, following FITS/DS9 standard
    return centroids


def get_mbi_centroids_manual(
    cams: Sequence[int], fields: Sequence[str], npsfs: int
) -> dict[str, NDArray]:
    centroids = {f"cam{cam:.0f}": np.empty((len(fields), npsfs, 2), dtype="f4") for cam in cams}
    for cam_key, cent_arr in centroids.items():
        click.echo(f"Enter comma-separated x, y centroids for {click.style(cam_key, bold=True)}:")
        for i, field in enumerate(fields):
            for j in range(npsfs):
                response = click.prompt(f" - Field: {field} PSF: {i} (x, y)")
                cent_arr[i, j] = [float(r) for r in response.split(",")]
    return centroids


MPL_INSTRUCTIONS: Final[str] = "<left-click> select, <right-click> delete"


def smooth_and_mask(data, smooth=True, mask=0):
    if smooth:
        data = filters.gaussian(data, sigma=2, preserve_range=True)
    if mask > 1:
        rs = frame_radii(data)
        data[rs <= mask] = np.nan
    return data


def get_psf_centroids_mpl(
    cams: dict[str, Path], npsfs=1, smooth=True, mask=0
) -> dict[str, NDArray]:
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    fig.supxlabel(MPL_INSTRUCTIONS)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    # for each cam
    centroids = {cam: np.empty((1, npsfs, 2), dtype="f4") for cam in cams}
    for cam, img_path in cams.items():
        data = load_fits(img_path)
        data = smooth_and_mask(data, smooth=smooth, mask=mask)
        # get list of (x, y) tuples from user clicks
        cent_arr = centroids[cam]
        ax.imshow(data, origin="lower", cmap="magma", norm=col.LogNorm())
        ax.set_title(f"Please select {cam} centroids")

        fig.tight_layout()
        fig.show()
        points = fig.ginput(npsfs, show_clicks=True)

        for i, point in enumerate(points):
            inds = cutout_inds(data, center=(point[1], point[0]), window=15)
            cent_arr[0, i] = offset_centroids(data, None, inds)["com"]
            ax.text(cent_arr[0, i, 1] + 2, cent_arr[0, i, 0] + 2, str(i), c="green")
        ax.scatter(cent_arr[0, i, 1], cent_arr[0, i, 0], marker="+", c="green")
        fig.show()
        # flip output so orientation is (cam, field, x, y)
        centroids[cam] = np.flip(cent_arr, axis=-1)
    plt.show(block=False)
    plt.ioff()

    return centroids


def get_mbi_centroids_mpl(
    cams: dict[str, Path], fields: Sequence[str], npsfs=1, smooth=True, mask=0
) -> dict[str, NDArray]:
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    fig.supxlabel(MPL_INSTRUCTIONS)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    nfields = len(fields)
    is_mbir = nfields == 3
    # for each cam
    centroids = {cam: np.empty((nfields, npsfs, 2), dtype="f4") for cam in cams}
    for cam, img_path in cams.items():
        data = load_fits(img_path)
        # get list of (x, y) tuples from user clicks
        cent_arr = centroids[cam]
        for i, field in enumerate(fields):
            inds = get_mbi_cutout_inds(data, camera=int(cam[-1]), field=field, reduced=is_mbir)
            field_cutout = smooth_and_mask(data[inds], smooth=smooth, mask=mask)

            ax.imshow(field_cutout, origin="lower", cmap="magma", norm=col.LogNorm())
            ax.set_title(f"Please select {cam} centroids for field: {field}")

            fig.tight_layout()
            fig.show()
            points = fig.ginput(npsfs, show_clicks=True)

            for j, point in enumerate(points):
                wind_inds = cutout_inds(field_cutout, center=(point[1], point[0]), window=15)
                cent_arr[i, j] = offset_centroids(field_cutout, None, wind_inds)["com"]
                # ax.text(cent_arr[i, j, 1] + 2, cent_arr[i, j, 0] + 2, str(i), c="green")
            # ax.scatter(cent_arr[i, :, 1], cent_arr[i, :, 0], marker="+", c="green")
            fig.show()
            # offset by mbi inds
            cent_arr[i, :, 0] += inds[-2].start
            cent_arr[i, :, 1] += inds[-1].start
        # flip output so orientation is (cam, field, x, y)
        centroids[cam] = np.flip(cent_arr, axis=-1)
    plt.show(block=False)
    plt.ioff()

    return centroids


def get_mbi_cutout_inds(
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
    return cutout_inds(data, window=500, center=(y, x))


def create_raw_input_psfs(table, basename: Path, max_files=5) -> dict[str, Path]:
    # group by cameras
    outfiles = {}
    for cam_num, group in table.groupby("U_CAMERA"):
        paths = group["path"].sample(n=min(len(group), max_files))
        outname = basename.with_name(f"{basename.name}_cam{cam_num:.0f}.fits")
        outfile = collapse_frames_files(paths, output=outname, cubes=True, fix=True, quiet=False)
        outfiles[f"cam{cam_num:.0f}"] = outfile
        logger.info(f"Saved raw PSF frame to {outname}")
    return outfiles


def save_centroids(
    centroids: dict[str, NDArray], fields: Sequence[str], basename: Path
) -> dict[str, Path]:
    outpaths = {}
    for cam_key, cent_arr in centroids.items():
        outpaths[cam_key] = basename.with_name(f"{basename.name}_{cam_key}.toml")
        cent_dict = dict(zip(fields, cent_arr.tolist(), strict=True))
        with outpaths[cam_key].open("wb") as fh:
            tomli_w.dump(cent_dict, fh)
        logger.info(f"Saved {cam_key} centroids to {outpaths[cam_key]}")
    return outpaths


########## centroid ##########


@click.command(name="centroid", help="Get image centroid estimates")
@click.option("-m", "--manual", is_flag=True, default=False, help="Enter centroids manually")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, multiprocessing.cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def centroid(config: Path, filenames, num_proc, outdir, manual=False):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    paths = Paths(outdir)
    paths.aux.mkdir(parents=True, exist_ok=True)
    npsfs = 4 if pipeline_config.coronagraphic else 1
    table = header_table(filenames, num_proc=num_proc)
    obsmodes = table["OBS-MOD"].unique()
    # default for standard obs, overwritten by MBI
    fields = ("SCI",)
    if manual:
        cams = table["U_CAMERA"].unique()
        if "MBIR" in obsmodes[0]:
            fields = ("F670", "F720", "F760")
            centroids = get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
        elif "MBI" in obsmodes[0]:
            fields = ("F610", "F670", "F720", "F760")
            centroids = get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
        else:
            centroids = get_psf_centroids_manual(cams=cams, npsfs=npsfs)

    else:
        name = paths.aux / f"{pipeline_config.name}_raw_psf"
        raw_psf_dict = create_raw_input_psfs(table, basename=name)

        if "MBIR" in obsmodes[0]:
            fields = ("F670", "F720", "F760")
            centroids = get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, npsfs=npsfs)
        elif "MBI" in obsmodes[0]:
            fields = ("F610", "F670", "F720", "F760")
            centroids = get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, npsfs=npsfs)
        else:
            centroids = get_psf_centroids_mpl(cams=raw_psf_dict, npsfs=npsfs)

    # save outputs
    basename = paths.aux / f"{pipeline_config.name}_centroids"
    save_centroids(centroids, fields=fields, basename=basename)
