import multiprocessing
import os
from collections.abc import Sequence
from pathlib import Path

import click
import numpy as np
import tomli_w
from astropy.io import fits
from numpy.typing import NDArray

from vampires_dpp.coadd import collapse_cubes_filelist
from vampires_dpp.image_registration import autocentroid_hdul
from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.specphot.filters import determine_filterset_from_header

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


def create_raw_input_psfs(table, basename: Path, max_files=5) -> dict[str, Path]:
    # group by cameras
    outhduls = {}
    for cam_num, group in table.groupby("U_CAMERA"):
        paths = group["path"].sample(n=min(len(group), max_files))
        outname = basename.with_name(f"{basename.name}_cam{cam_num:.0f}.fits")
        if outname.exists():
            logger.info(f"Loading raw PSF frame from {outname}")
            hdul = fits.open(outname)
        else:
            logger.info(f"Creating mean frame from {len(paths)} files")
            hdul = collapse_cubes_filelist(paths, fix=True)
            hdul.writeto(outname, overwrite=True)
            logger.info(f"Saved raw PSF frame to {outname}")
        outhduls[f"cam{cam_num:.0f}"] = hdul
    return outhduls


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
@click.option(
    "-p/-np",
    "--plot/--no-plot",
    is_flag=True,
    default=bool(os.environ.get("DISPLAY", None)),
    help="Show centroiding plots (requires a display)",
)
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
def centroid(config: Path, filenames, num_proc, outdir, manual, plot):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    paths = Paths(outdir)
    paths.aux.mkdir(parents=True, exist_ok=True)
    npsfs = 4 if pipeline_config.coronagraphic else 1
    # choose 5 random files
    table = header_table(filenames, num_proc=num_proc)
    obsmodes = table["OBS-MOD"].unique()
    if len(obsmodes) > 1:
        msg = f"Found {len(obsmodes)} unique OBS-MOD, make sure you're only processing one type of VAMPIRES data. Will proceed with first mode: {obsmodes.iloc[0]}"
        click.echo(msg)
    # default for standard obs, overwritten by MBI
    fields = determine_filterset_from_header(table.iloc[0])
    if manual:
        cams = table["U_CAMERA"].unique()
        if "MBI" in obsmodes[0]:
            centroids = get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
        else:
            centroids = get_psf_centroids_manual(cams=cams, npsfs=npsfs)

    else:
        name = paths.aux / f"{pipeline_config.name}_mean_image"
        # choose 4 to 20 files, depending on file size (avoid loading more than 500 frames, ~2GB of MBI)
        number_files = int(max(2, min(10, 500 // table["NAXIS3"].median())))
        input_hduls_dict = create_raw_input_psfs(table, basename=name, max_files=number_files)
        centroids = {}
        for key, input_hdul in input_hduls_dict.items():
            centroids[key] = (
                autocentroid_hdul(
                    input_hdul,
                    coronagraphic=pipeline_config.coronagraphic,
                    planetary=pipeline_config.planetary,
                    window_size=pipeline_config.analysis.window_size,
                    plot=plot,
                )
                + 1
            )
            # add 1 to get back to ds9 from numpy coords

    # save outputs
    basename = paths.aux / f"{pipeline_config.name}_centroids"
    save_centroids(centroids, fields=fields, basename=basename)
