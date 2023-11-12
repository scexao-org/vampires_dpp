import functools
from pathlib import Path
from typing import Sequence, Literal

import click
import numpy as np
from numpy.typing import NDArray

from . import logger

from ..pipeline.config import PipelineConfig
from .. image_processing import collapse_frames_files
from ..image_registration import offset_centroids
from ..indexing import cutout_inds
from ..util import load_fits


def get_psf_centroids_mpl(cams: dict[str, Path], npsfs=1, nfields=1):
    pass

def get_psf_centroids_manual(cams: list[int], npsfs: int) -> dict[str, NDArray]:
    centroids = {f"cam{cam:.01f}": np.empty((npsfs, 2), dtype="f4") for cam in cams}
    for key in centroids.keys():
        click.echo(f"Enter comma-separated x, y centroids for {click.style(key, bold=True)}")
        for i in range(npsfs):
            response = click.prompt(f" - PSF {i} (x, y)")
            centroids[key][i] = map(float, response.split(","))
    # output is (nspfs, (x, y)) with +1 in x and y, following FITS/DS9 standard
    return centroids

def get_mbi_centroids_mpl(mean_image, coronagraphic=False, suptitle=None):
    pass

def get_mbi_centroids_manual(fields: Sequence, npsfs: Literal[1, 4], ) -> NDArray:
    centroids = np.empty((len(fields), npsfs, 2), dtype="f4")
    for i, field in enumerate(fields):
        click.echo(f"Field: {field}")
        for j in range(npsfs):
            response = click.prompt(f" - Enter (x, y) centroid for PSF {i} (comma-separated)")
            centroids[i, j] = map(float, response.split(","))
    return centroids

def get_psf_centroids_mpl(cams: dict[str, Path], npsfs=1):
    import matplotlib.colors as col
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    # plot mean image
    ax.imshow(mean_image, origin="lower", cmap="magma", norm=col.LogNorm())
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.tight_layout
    fig.show()

    # for each cam
    for cam, img_path in cams.values():
        # get list of (x, y) tuples from user clicks
        output = np.empty((nfields, npsfs, 2))
        for i in range(nfields):
            ax.set_title(f"Please select centroids for field {i + 1}")
            points = fig.ginput(npsfs, show_clicks=True)
            for j, point in enumerate(points):
                inds = cutout_inds(mean_image, center=(point[1], point[0]), window=15)
                output[i, j] = offset_centroids(mean_image, None, inds)["com"]
                ax.text(output[i, j, 1] + 2, output[i, j, 0] + 2, str(i), c="green")
            ax.scatter(output[i, :, 1], output[i, :, 0], marker="+", c="green")
            fig.show()
    plt.show(block=False)
    plt.ioff()

    # flip output so file is saved as (x, y)
    return np.flip(output, axis=-1)


def create_raw_input_psfs(outpath, table, max_files=10) -> dict[str, Path]:
    # group by cameras
    outfiles = {}
    for cam_num, group in table.groupby("U_CAMERA"):
        paths = group["path"].sample(n=max_files)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outfile = collapse_frames_files(paths, output=outpath, cubes=True, quiet=False)
        outfiles[f"cam{cam_num:.0f}"] = outfile
        logger.info(f"Saved raw PSF frame to {outpath.absolute()}")
    return outfiles

def save_centroids(self, table):
    raw_psf_filenames = create_raw_input_psfs(table)
    for key in ("cam1", "cam2"):
        path = self.paths.preproc_dir / f"{self.config.name}_centroids_{key}.toml"

        im, hdr = load_fits(raw_psf_filenames[key], header=True)
        outpath = self.paths.preproc_dir / f"{self.config.name}_{key}.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)

        ctrs = get_psf_centroids_mpl(
            np.squeeze(im), npsfs=npsfs, nfields=nfields, suptitle=key, outpath=outpath
        )
        ctrs_as_dict = dict(zip(field_keys, ctrs.tolist()))
        with path.open("wb") as fh:
            tomli_w.dump(ctrs_as_dict, fh)
        logger.debug(f"Saved {key} centroids to {path}")

########## centroid ##########

@click.command(name="centroid", help="Get image centroid estimates")
@click.option("-m", "--manual", is_flag=True, default=False, help="Enter centroids manually")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def centroid(config: Path, filenames, outdir, num_proc, manual=False):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    nspfs = 4 if pipeline_config.coronagraphic else 1
    table = header_table(filenames, num_proc=num_proc)
    obsmodes = table["OBS-MOD"].unique()
    if manual:
        cams = table["U_CAMERA"].unique()
        if "MBIR" in obsmodes.iloc[0]
            fields = ("F670", "F720", "F760")
            centroids = get_mbi_centroids_manual(cams=cams, fields=fields, nspfs=npsfs)
        elif "MBI" in obsmodes.iloc[0]
            fields = ("F610", "F670", "F720", "F760")
            centroids = get_mbi_centroids_manual(cams=cams, fields=fields, nspfs=npsfs)
        else:
            centroids = get_psf_centroids_manual(cams=cams, nspfs=npsfs)

    else:
        raw_psf_dict = create_raw_input_psfs(table)

        if "MBIR" in obsmodes.iloc[0]
            fields = ("F670", "F720", "F760")
            centroids = get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, nspfs=npsfs)
        elif "MBI" in obsmodes.iloc[0]
            fields = ("F610", "F670", "F720", "F760")
            centroids = get_mbi_centroids_mpl(cams=raw_psf_dict, fields=fields, nspfs=npsfs)
        else:
            centroids = get_psf_centroids_mpl(cams=raw_psf_dict, nspfs=npsfs)

