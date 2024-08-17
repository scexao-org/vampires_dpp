import multiprocessing
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import click
import numpy as np
import tomli_w
from numpy.typing import NDArray

from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths
from vampires_dpp.pipeline.config import PipelineConfig

from . import logger
from .centroids import (
    create_raw_input_psfs,
    get_mbi_centroids_manual,
    get_mbi_centroids_mpl,
    get_psf_centroids_manual,
    get_psf_centroids_mpl,
)

__all__ = "astro"


def calculate_astrometry_from_centroids(
    centroids: OrderedDict[int, NDArray]
) -> OrderedDict[int, OrderedDict[str, list]]:
    astrom = OrderedDict()
    for key, ctr in centroids.items():
        abs_ctr = ctr.mean(axis=-2, keepdims=True)  # x, y FITS
        delta = ctr - abs_ctr
        angs = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))
        sep = np.linalg.norm(delta, axis=-1).mean(axis=-1)
        astrom[key] = OrderedDict(separation=sep, angle=np.mean(angs % 90))
    return astrom


def get_psf_astrometry_manual(
    cams: Sequence[int], npsfs: int
) -> OrderedDict[int, OrderedDict[str, list]]:
    centroids = get_psf_centroids_manual(cams=cams, npsfs=npsfs)
    return calculate_astrometry_from_centroids(centroids)


def get_mbi_astrometry_manual(
    cams: Sequence[int], fields: Sequence[str], npsfs: int
) -> OrderedDict[int, OrderedDict[str, list]]:
    centroids = get_mbi_centroids_manual(cams=cams, fields=fields, npsfs=npsfs)
    return calculate_astrometry_from_centroids(centroids)


MPL_INSTRUCTIONS: Final[str] = "<left-click> select, <right-click> delete"


def get_psf_astrometry_mpl(
    cams: OrderedDict[str, Path], npsfs=1, smooth=True
) -> OrderedDict[int, OrderedDict[str, list]]:
    centroids = get_psf_centroids_mpl(cams=cams, npsfs=npsfs, smooth=smooth, mask=15)
    return calculate_astrometry_from_centroids(centroids)


def get_mbi_astrometry_mpl(
    cams: OrderedDict[str, Path], fields: Sequence[str], npsfs=1, smooth=True
) -> OrderedDict[int, OrderedDict[str, list]]:
    centroids = get_mbi_centroids_mpl(cams=cams, fields=fields, npsfs=npsfs, smooth=smooth, mask=15)
    return calculate_astrometry_from_centroids(centroids)


def save_astrometry(
    astrometry: OrderedDict[int, OrderedDict[str, list]], fields: Sequence[str], basename: Path
) -> Path:
    outpath = basename.with_name(f"{basename.name}.toml")

    astrom = OrderedDict()
    for key, vals in astrometry.items():
        astrom[key] = {
            "fields": fields,
            "separation": vals["separation"].tolist(),
            "angle": vals["angle"].tolist(),
        }

    with outpath.open("wb") as fh:
        tomli_w.dump(astrom, fh)
        logger.info(f"Saved astrometry information to {outpath}")
    return outpath


########## astro ##########


@click.command(name="astro", help="Get image astrometric solution")
@click.option("-m", "--manual", is_flag=True, default=False, help="Enter solution manually")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, multiprocessing.cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def astro(config: Path, filenames, num_proc, manual=False):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    paths = Paths(Path.cwd())
    paths.aux.mkdir(parents=True, exist_ok=True)
    table = header_table(filenames, num_proc=num_proc)
    obsmodes = table["OBS-MOD"].unique()
    # default for standard obs, overwritten by MBI
    fields = ("SCI",)
    if manual:
        cams = table["U_CAMERA"].unique()
        if "MBIR" in obsmodes[0]:
            fields = ("F670", "F720", "F760")
            astrometry = get_mbi_astrometry_manual(cams=cams, fields=fields, npsfs=4)
        elif "MBI" in obsmodes[0]:
            fields = ("F610", "F670", "F720", "F760")
            astrometry = get_mbi_astrometry_manual(cams=cams, fields=fields, npsfs=4)
        else:
            astrometry = get_psf_astrometry_manual(cams=cams, npsfs=4)

    else:
        name = paths.aux / f"{pipeline_config.name}_raw_psf"
        raw_psf_dict = create_raw_input_psfs(table, basename=name)

        if "MBIR" in obsmodes[0]:
            fields = ("F670", "F720", "F760")
            astrometry = get_mbi_astrometry_mpl(cams=raw_psf_dict, fields=fields, npsfs=4)
        elif "MBI" in obsmodes[0]:
            fields = ("F610", "F670", "F720", "F760")
            astrometry = get_mbi_astrometry_mpl(cams=raw_psf_dict, fields=fields, npsfs=4)
        else:
            astrometry = get_psf_astrometry_mpl(cams=raw_psf_dict, npsfs=4)

    # save outputs
    basename = paths.aux / f"{pipeline_config.name}_astrometry"
    save_astrometry(astrometry, fields=fields, basename=basename)
