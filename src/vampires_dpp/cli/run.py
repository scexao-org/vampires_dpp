from multiprocessing import cpu_count
from pathlib import Path

import click

import vampires_dpp as dpp
from vampires_dpp.logging_utils import add_logfile, configurelogging_utils
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.pipeline.pipeline import PIPELINE_STAGES, Pipeline

__all__ = ("run", "pdi")


########## run ##########
def log_intro(logger, num_proc, outdir):
    logger.info(f"VAMPIRES DPP: v{dpp.__version__}")

    logger.info(f"Using {num_proc} processes")
    logfile = outdir / "debug.log"
    logger.info(f"Note: a detailed log can be found in {logfile}")
    logger.info(f"Tip: watch it live with `tail -f {logfile}`")


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
@click.option(
    "--redo",
    default=None,
    type=click.Choice(PIPELINE_STAGES),
    help="Force redo a pipeline stage; updated outputs cascade to downstream stages automatically.",
)
def run(config: Path, filenames, num_proc, outdir, verbose, redo):
    logger = configurelogging_utils(level="DEBUG" if verbose else "INFO")
    logger = add_logfile(outdir, logger)

    log_intro(logger, num_proc, outdir)

    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir, verbose=verbose)

    if len(filenames) == 0:
        msg = "No files input to pipeline! Double check command-line input for typos"
        raise ValueError(msg)
    pipeline.run(filenames, num_proc=num_proc, redo=redo)
    # adi and diff are independent of pdi — skip polarimetry when targeting either
    if pipeline.config.polarimetry is not None and redo not in ("adi", "diff"):
        pipeline.run_polarimetry(num_proc=num_proc, redo=redo)


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
@click.option(
    "--redo",
    default=None,
    type=click.Choice(["pdi", "all"]),
    help="Force redo a pipeline stage; updated outputs cascade to downstream stages automatically.",
)
def pdi(config, filenames, num_proc, verbose, outdir, redo):
    logger = configurelogging_utils(level="DEBUG" if verbose else "INFO")
    logger = add_logfile(outdir, logger)

    log_intro(logger, num_proc, outdir)

    pipeline = Pipeline(PipelineConfig.from_file(config), workdir=outdir, verbose=verbose)

    if len(filenames) == 0:
        msg = "No files input to pipeline! Double check command-line input for typos"
        raise ValueError(msg)

    pipeline.run_polarimetry(num_proc=num_proc, redo=redo)
