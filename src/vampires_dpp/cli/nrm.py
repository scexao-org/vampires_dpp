import multiprocessing
from pathlib import Path

import click
import numpy as np

from vampires_dpp._logging import configure_logging
from vampires_dpp.cli.centroids import create_raw_input_psfs
from vampires_dpp.nrm.alignment import check_mask_align
from vampires_dpp.nrm.params import get_amical_parameters
from vampires_dpp.nrm.windowing import window_cube
from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths
from vampires_dpp.pipeline.config import PipelineConfig

logger = configure_logging()


@click.group(name="nrm", short_help="NRM specific tools", help="NRM mask alignment, analysis")
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, multiprocessing.cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.pass_context
def nrm(ctx, num_proc):
    # prepare context
    ctx.ensure_object(dict)
    ctx.obj["num_proc"] = num_proc


@nrm.command(
    name="check_align",
    short_help="Check uv scale/theta alignment",
    help="Check uv scale/theta alignment from mean combined data in `aux` directory. Plots are saved to `nrm/figures`. Adjust `nrm.uv` and `nrm.theta` in the TOML file to test different values.",
)
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.pass_context
def check_align(ctx, config: Path, filenames):
    # make sure versions match within SemVar
    pipeline_config = PipelineConfig.from_file(config)
    # figure out outpath
    paths = Paths(Path.cwd())
    paths.aux.mkdir(parents=True, exist_ok=True)
    # npsfs = 4 if pipeline_config.coronagraphic else 1
    # choose 5 random files
    table = header_table(filenames, num_proc=ctx.obj["num_proc"])
    obsmodes = table["OBS-MOD"].unique()
    if len(obsmodes) > 1:
        msg = f"Found {len(obsmodes)} unique OBS-MOD, make sure you're only processing one type of VAMPIRES data. Will proceed with first mode: {obsmodes.iloc[0]}"
        click.echo(msg)
    # default for standard obs, overwritten by MBI
    # fields = determine_filterset_from_header(table.iloc[0])

    name = paths.aux / f"{pipeline_config.name}_mean_image"
    # choose 4 to 20 files, depending on file size (avoid loading more than 500 frames, ~2GB of MBI)
    number_files = int(max(2, min(10, 500 // table["NAXIS3"].median())))
    input_hduls_dict = create_raw_input_psfs(table, basename=name, max_files=number_files)
    figdir = paths.nrm / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for key, input_hdul in input_hduls_dict.items():
        cube, header = window_cube(
            np.nan_to_num(input_hdul[0].data[None, :, :]), size=80, header=input_hdul[0].header
        )

        save_basename = figdir / f"{key}_"
        check_mask_align(
            cube, params=get_amical_parameters(input_hdul[0].header), save_path=str(save_basename)
        )
