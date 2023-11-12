from multiprocessing import cpu_count
from pathlib import Path

import click

import vampires_dpp as dpp
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.pipeline.config import (
    PipelineConfig,
)
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.util import check_version

from . import logger

########## run ##########


@click.command(name="run", help="Run the data processing pipeline")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
def run(config: Path, filenames, num_proc, outdir):
    # make sure versions match within SemVar
    logfile = outdir / "debug.log"
    logfile.unlink(missing_ok=True)
    logger.add(logfile, level="DEBUG", enqueue=True, colorize=False)
    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir)
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    pipeline.run(filenames, num_proc=num_proc)
    pipeline.run_polarimetry(num_proc=num_proc)


########## pdi ##########


@click.command(name="pdi", help="Run the polarimetric differential imaging pipeline")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def pdi(config, filenames, num_proc, quiet, outdir):
    # make sure versions match within SemVar
    logger.add(outdir / "debug.log", level="DEBUG", enqueue=True, colorize=False)
    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir)
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    pipeline.run_polarimetry(num_proc=num_proc)
