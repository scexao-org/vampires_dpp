import multiprocessing as mp
from multiprocessing import cpu_count
from pathlib import Path

import click
import tqdm.auto as tqdm

from vampires_dpp.calib.calib_files import process_background_files, process_flat_files
from vampires_dpp.calib.normalize import normalize_file

__all__ = ("calib", "back", "flat", "norm")

########## calib ##########


@click.group(
    name="calib",
    short_help="Prepare calibration files",
    help="Create calibration files from background files (darks or sky frames) and flats.",
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path.cwd(),
    help="Output directory.",
)
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--quiet", "-q", is_flag=True, help="Silence progress bars and extraneous logging.")
@click.pass_context
def calib(ctx, outdir, quiet, num_proc):
    # prepare context
    ctx.ensure_object(dict)
    ctx.obj["outdir"] = outdir
    ctx.obj["quiet"] = quiet
    ctx.obj["num_proc"] = num_proc


@calib.command(
    name="back",
    short_help="background files (darks/skies)",
    help="Create background files from darks/skies. Each input file will be collapsed. Groups of files with the same exposure time, EM gain, and frame size will be median-combined together to create a super-background file.",
)
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--collapse",
    "-c",
    type=click.Choice(["median", "mean", "varmean", "biweight"], case_sensitive=False),
    default="median",
    help="Frame collapse method",
    show_default=True,
)
@click.option("--force", "-f", is_flag=True, help="Force processing of files")
@click.pass_context
def back(ctx, filenames, collapse, force):
    process_background_files(
        filenames,
        collapse=collapse,
        force=force,
        output_directory=ctx.obj["outdir"] / "background",
        quiet=ctx.obj["quiet"],
        num_proc=ctx.obj["num_proc"],
    )


@calib.command(
    name="flat",
    short_help="flat-field files",
    help="Create flat-field files. Each input file will be collapsed with background-subtraction if files are provided. Groups of files with the same exposure time, EM gain, frame size, and filter will be median-combined together to create a super-flat file.",
)
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--collapse",
    "-c",
    type=click.Choice(["median", "mean", "varmean", "biweight"], case_sensitive=False),
    default="median",
    help="Frame collapse method",
    show_default=True,
)
@click.option(
    "-n/-u", "--norm/--no-norm", default=True, help="Normalize flat fields by field median"
)
@click.option("--force", "-f", is_flag=True, help="Force processing of files")
@click.pass_context
def flat(ctx, filenames, collapse, norm: bool, force: bool):
    # if directory, filter non-FITS files and sort for background files
    back = ctx.obj["outdir"]
    calib_files = list(back.glob("**/[!._]*.fits")) + list(back.glob("**/[!._]*.fits.fz"))
    process_flat_files(
        filenames,
        collapse=collapse,
        normalize=norm,
        force=force,
        output_directory=ctx.obj["outdir"] / "flat",
        quiet=ctx.obj["quiet"],
        num_proc=ctx.obj["num_proc"],
        background_files=calib_files,
    )


@click.command(name="norm", help="Normalize VAMPIRES data files")
@click.argument(
    "filenames", nargs=-1, type=click.Path(dir_okay=False, readable=True, path_type=Path)
)
@click.option("-o", "--outdir", type=Path, default=Path.cwd() / "prep", help="Output directory")
@click.option(
    "-d",
    "--deint",
    is_flag=True,
    help="Deinterleave files into FLC states (WARNING: only apply this to old VAMPIRES data downloaded directly from `sonne`)",
)
@click.option(
    "-nf",
    "--no-filter-empty",
    is_flag=True,
    help="Don't filter empty frames from data (post deinterleaving, if applicable)",
)
@click.option(
    "--num-proc",
    "-j",
    default=1,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--quiet", "-q", is_flag=True, help="Silence progress bars and extraneous logging.")
def norm(filenames, deint: bool, no_filter_empty: bool, num_proc: int, quiet: bool, outdir: Path):
    jobs = []
    kwargs = dict(deinterleave=deint, filter_empty=not no_filter_empty, output_directory=outdir)
    with mp.Pool(num_proc) as pool:
        for filename in filenames:
            jobs.append(pool.apply_async(normalize_file, args=(filename,), kwds=kwargs))

        iter = jobs if quiet else tqdm.tqdm(jobs, desc="Normalizing files")
        results = [job.get() for job in iter]

    return results
