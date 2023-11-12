import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click
import numpy as np
from numpy.typing import NDArray

from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.image_processing import collapse_frames_files
from vampires_dpp.image_registration import offset_centroids
from vampires_dpp.indexing import cutout_inds, frame_center
from vampires_dpp.organization import header_table
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.util import load_fits

from . import logger

__all__ = "centroid"


def get_psf_centroids_manual(cams: Sequence[int], npsfs: int) -> dict[str, NDArray]:
    centroids = {f"cam{cam:.01f}": np.empty((npsfs, 2), dtype="f4") for cam in cams}
    for key in centroids:
        click.echo(f"Enter comma-separated x, y centroids for {click.style(key, bold=True)}")
        for i in range(npsfs):
            response = click.prompt(f" - PSF {i} (x, y)")
            centroids[key][i] = map(float, response.split(","))
    # output is (nspfs, (x, y)) with +1 in x and y, following FITS/DS9 standard
    return centroids


def get_mbi_centroids_mpl(mean_image, coronagraphic=False, suptitle=None):
    pass


def get_mbi_centroids_manual(cams: Sequence[int], fields: Sequence[str], npsfs: int) -> NDArray:
    centroids = np.empty((len(fields), npsfs, 2), dtype="f4")
    for i, field in enumerate(fields):
        click.echo(f"Field: {field}")
        for j in range(npsfs):
            response = click.prompt(f" - Enter (x, y) centroid for PSF {i} (comma-separated)")
            centroids[i, j] = map(float, response.split(","))
    return centroids


def get_psf_centroids_mpl(cams: dict[str, Path], fields: Sequence[str], npsfs=1):
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    nfields = len(fields)
    # for each cam
    centroids = {}
    for cam, img_path in cams.items():
        data = load_fits(img_path)
        # get list of (x, y) tuples from user clicks
        output = np.empty((nfields, npsfs, 2))
        fig.suptitle(cam)
        for i, field in enumerate(fields):
            inds = get_mbi_cutout_inds(
                data, camera=int(cam[-1]), field=field, reduced=len(fields) == 3
            )
            ext = (inds[1].start, inds[1].stop, inds[0].start, inds[1].stop)
            ax.imshow(data[inds], origin="lower", cmap="magma", norm=col.LogNorm(), extent=ext)
            ax.set_title(f"Please select centroids for field: {field}")

            fig.tight_layout()
            fig.show()
            points = fig.ginput(npsfs, show_clicks=True)

            for j, point in enumerate(points):
                inds = cutout_inds(data[inds], center=(point[1], point[0]), window=15)
                output[i, j] = offset_centroids(data[inds], None, inds)["com"]
                ax.text(output[i, j, 1] + 2, output[i, j, 0] + 2, str(i), c="green")
            ax.scatter(output[i, :, 1], output[i, :, 0], marker="+", c="green")
            fig.show()
        # flip output so orientation is (cam, field, x, y)
        centroids[cam] = np.flip(output, axis=-1)
    plt.show(block=False)
    plt.ioff()

    return centroids


def get_mbi_cutout_inds(
    data, camera: int, field: Literal["F610", "F670", "F720", "F670"], reduced: bool = False
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
    if reduced:
        y *= 2
    # flip y axis for cam 1 indices
    if camera == 1:
        y = data.shape[-2] - y

    return cutout_inds(data, window=500, center=(y, x))


def create_raw_input_psfs(output_directory, table, max_files=10) -> dict[str, Path]:
    # group by cameras
    outfiles = {}
    for cam_num, group in table.groupby("U_CAMERA"):
        paths = group["path"].sample(n=max_files)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outfile = collapse_frames_files(
            paths,
            output_directory=output_directory,
            suffix=f"_raw_psf_cam{cam_num:.01f}",
            cubes=True,
            quiet=False,
        )
        outfiles[f"cam{cam_num:.0f}"] = outfile
        logger.info(f"Saved raw PSF frame to {outpath.absolute()}")
    return outfiles


# def save_centroids(self, table):
#     raw_psf_filenames = create_raw_input_psfs(table)
#     for key in ("cam1", "cam2"):
#         path = self.paths.preproc_dir / f"{self.config.name}_centroids_{key}.toml"

#         im, hdr = load_fits(raw_psf_filenames[key], header=True)
#         outpath = self.paths.preproc_dir / f"{self.config.name}_{key}.png"
#         outpath.parent.mkdir(parents=True, exist_ok=True)

#         ctrs = get_psf_centroids_mpl(
#             np.squeeze(im), npsfs=npsfs, nfields=nfields, suptitle=key, outpath=outpath
#         )
#         ctrs_as_dict = dict(zip(field_keys, ctrs.tolist()))
#         with path.open("wb") as fh:
#             tomli_w.dump(ctrs_as_dict, fh)
#         logger.debug(f"Saved {key} centroids to {path}")

########## centroid ##########


@click.command(name="centroid", help="Get image centroid estimates")
@click.option("-m", "--manual", is_flag=True, default=False, help="Enter centroids manually")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, multiprocessing.cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def centroid(config: Path, filenames, num_proc, manual=False):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    outpath = "TODO"
    npsfs = 4 if pipeline_config.coronagraphic else 1
    table = header_table(filenames, num_proc=num_proc)
    obsmodes = table["OBS-MOD"].unique()
    if manual:
        cams = table["U_CAMERA"].unique()
        if "MBIR" in obsmodes.iloc[0]:
            fields = ("F670", "F720", "F760")
            get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
        elif "MBI" in obsmodes.iloc[0]:
            fields = ("F610", "F670", "F720", "F760")
            get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
        else:
            get_psf_centroids_manual(cams=cams, npsfs=npsfs)

    else:
        raw_psf_dict = create_raw_input_psfs(table, output_directory=outpath)

        if "MBIR" in obsmodes.iloc[0]:
            fields = ("F670", "F720", "F760")
            get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, npsfs=npsfs)
        elif "MBI" in obsmodes.iloc[0]:
            fields = ("F610", "F670", "F720", "F760")
            get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, npsfs=npsfs)
        else:
            get_psf_centroids_mpl(cams=raw_psf_dict, npsfs=npsfs)
