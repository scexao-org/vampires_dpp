from multiprocessing import cpu_count
from pathlib import Path

import click

import vampires_dpp as dpp
from vampires_dpp._logging import add_logfile, configure_logging
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.util import check_version

__all__ = ("run", "pdi")

########## run ##########


@click.command(name="run", help="Run the data processing pipeline (including PDI)")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--verbose", "-v", is_flag=True, help="Print debug statements.")
def run(config: Path, filenames, num_proc, outdir, verbose):
    logger = configure_logging()
    logger = add_logfile(outdir, logger)

    logger.info(f"VAMPIRES DPP: v{dpp.__version__}")
    logger.info(f"Using {num_proc} processes")

    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir, verbose=verbose)

    # make sure versions match within SemVar
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        msg = f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with \
        installed version of `vampires_dpp` ({dpp.__version__}). Try running \
        `dpp upgrade {config}`."
        raise ValueError(msg)
    pipeline.run(filenames, num_proc=num_proc)
    # only run PDI if specified
    if pipeline.config.polarimetry is not None:
        pipeline.run_polarimetry(num_proc=num_proc)


########## pimport warnings ##########


@click.command(name="pdi", help="Run the PDI pipeline only")
@click.argument("config", type=click.Path(dir_okay=False, readable=True, path_type=Path))
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-o", "--outdir", default=Path.cwd(), type=Path, help="Output file directory")
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--verbose", "-v", is_flag=True, help="Print debug statements.")
def pdi(config, filenames, num_proc, verbose, outdir):
    logger = configure_logging()
    logger = add_logfile(outdir, logger)

    logger.info(f"VAMPIRES DPP: v{dpp.__version__}")
    logger.info(f"Using {num_proc} processes")

    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir, verbose=verbose)

    # make sure versions match within SemVar
    if not check_version(pipeline.config.dpp_version, dpp.__version__):
        msg = f"Input pipeline version ({pipeline.config.dpp_version}) is not compatible with \
                installed version of `vampires_dpp` ({dpp.__version__}). Try running \
                `dpp upgrade {config}`."
        raise ValueError(msg)

    pipeline.run_polarimetry(num_proc=num_proc)
