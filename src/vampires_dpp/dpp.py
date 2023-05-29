import glob
import logging
import os
import readline
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import cpu_count

from astropy.io import fits
import astropy.units as u
import numpy as np
import tomli
import click
from rich.logging import RichHandler

import vampires_dpp as dpp
from vampires_dpp.calibration import make_master_background, make_master_flat
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.organization import check_files, header_table, sort_files
from vampires_dpp.pipeline.config import (
    CamCtrOption,
    CamFileInput,
    CollapseOptions,
    CoordinateOptions,
    CoronagraphOptions,
    FrameSelectOptions,
    PipelineOptions,
    RegisterOptions,
    SatspotOptions,
)
from vampires_dpp.pipeline.deprecation import upgrade_config
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import (
    DEFAULT_DIRS,
    VAMPIRES_BLANK,
    VAMPIRES_PDI,
    VAMPIRES_SDI,
    VAMPIRES_SINGLECAM,
)
from vampires_dpp.util import check_version
from vampires_dpp.wcs import get_gaia_astrometry
from trogon import tui

# set up logging
handler = RichHandler(
    level=logging.INFO,
    log_time_format="%Y-%m-%d %H:%M:%S",
    show_path=False,
)


# callback that will confirm if a flag is false
def abort_if_false(ctx, param, value):
    if not value:
        ctx.abort()


def abort_if_true(ctx, param, value):
    if value:
        ctx.abort()


########## sort ##########


@click.command(
    name="sort",
    short_help="Sort raw data",
    help="Sorts raw data based on the data type. This will either use the `DATA-TYP` header value or the `U_OGFNAM` header, depending on when your data was taken.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
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
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--ext", "-e", default=0, help="HDU extension")
@click.option(
    "--copy/--no-copy",
    "-c/",
    callback=abort_if_false,
    prompt="Are you sure you want to move data files?",
    help="copy files instead of moving them",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def sort_raw(filenames, outdir=Path.cwd(), num_proc=DEFAULT_NPROC, ext=0, copy=False, quiet=False):
    sort_files(
        filenames,
        copy=copy,
        ext=ext,
        output_directory=outdir,
        num_proc=num_proc,
        quiet=quiet,
    )


########## prep ##########


@click.group(
    name="prep",
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
@click.pass_context
def prep(ctx, outdir, quiet, num_proc):
    # prepare context
    ctx.ensure_object(dict)
    ctx.obj["outdir"] = outdir
    ctx.obj["quiet"] = quiet
    ctx.obj["num_proc"] = num_proc


@prep.command(
    name="back",
    short_help="background files (darks/skies)",
    help="Create background files from darks/skies. Each input file will be collapsed. Groups of files with the same exposure time, EM gain, and frame size will be median-combined together to create a super-background file.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
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
    make_master_background(
        filenames,
        collapse=collapse,
        force=force,
        output_directory=ctx["outdir"],
        quiet=ctx["quiet"],
        num_proc=ctx["num_proc"],
    )


@prep.command(
    name="flat",
    short_help="flat-field files",
    help="Create flat-field files. Each input file will be collapsed with background-subtraction if files are provided. Groups of files with the same exposure time, EM gain, frame size, and filter will be median-combined together to create a super-flat file.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--back",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    help="Background file to subtract from each flat-field. If a directory, will match the background files in that directory to the exposure times, EM gains, and frame sizes.",
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
def flat(ctx, filenames, back, collapse, force):
    # if directory, filter non-FITS files and sort for background files
    background_files = []
    if back.is_dir():
        fits_files = back.rglob("*.fits.*") + back.rglob("*.fts.*") + back.rglob("*.fit.*")
        background_files.extend(
            filter(lambda f: fits.getkey(f, "CAL_TYPE") == "BACKGROUND", fits_files)
        )
    make_master_flat(
        filenames,
        collapse=collapse,
        force=force,
        output_directory=ctx["outdir"],
        quiet=ctx["quiet"],
        num_proc=ctx["num_proc"],
    )


########## new ##########


def pathCompleter(text, state):
    """
    This is the tab completer for systems paths.
    Only tested on *nix systems
    """
    # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
    if "~" in text:
        text = os.path.expanduser(text)

    # autocomplete directories with having a trailing slash
    if os.path.isdir(text):
        text += "/"

    return [x for x in glob.glob(text + "*")][state]


def createListCompleter(items):
    """
    This is a closure that creates a method that autocompletes from
    the given list.

    Since the autocomplete function can't be given a list to complete from
    a closure is used to create the listCompleter function with a list to complete
    from.
    """
    list_strings = map(str, items)

    def listCompleter(text, state):
        if not text:
            return list(list_strings)[state]
        else:
            matches = filter(lambda s: s.startswith(text), list_strings)
            return list(matches)[state]

    return listCompleter


@click.command(name="new", help="Generate configuration files")
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
def new_config(config):
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")

    ## check if output file exists
    if config.is_file():
        response = (
            input(
                f"{config.name} already exists in output directory, would you like to overwrite it? [y/N]: "
            )
            .strip()
            .lower()
        )
        if response != "y":
            return

    ## get template
    template_choices = ["singlecam", "pdi", "sdi"]

    readline.set_completer(createListCompleter(template_choices))
    template = (
        input(f"Choose a starting template [{'/'.join(template_choices)}] (optional): ")
        .strip()
        .lower()
    )
    match template:
        case "singlecam":
            tpl = VAMPIRES_SINGLECAM
        case "pdi":
            tpl = VAMPIRES_PDI
        case "sdi":
            tpl = VAMPIRES_SDI
        case _:
            tpl = VAMPIRES_BLANK
    readline.set_completer()

    ## get name
    name_guess = config.stem
    name = input(f"Path-friendly name for this reduction [{name_guess}]: ").strip().lower()
    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")

    ## get target
    object = input(f"SIMBAD-friendly object name (optional): ").strip()
    obj = None if object == "" else object
    coord = None
    if obj is not None:
        rad = 1
        cat = "dr3"
        while True:
            coord = get_gaia_astrometry(obj, catalog=cat, radius=rad)
            if coord is not None:
                break

            print(f'Could not find {obj} in GAIA {cat.upper()} with {rad}" radius.')
            _input = input(
                "Query different catalog (dr1/dr2/dr3), enter search radius in arcsec, or enter new object name (optional): "
            ).strip()
            match _cat := _input.lower():
                case "":
                    # give up
                    break
                case "dr1" | "dr2" | "dr3":
                    # try different catalog
                    cat = _cat
                case _:
                    try:
                        # if a number was entered, increase search radius
                        rad = float(_cat)
                    except:
                        # otherwise try a new object
                        obj = _input

        if coord is not None:
            tpl.coordinate = CoordinateOptions(
                object=obj,
                ra=coord.ra.to_string("hour", sep=":", pad=True),
                dec=coord.dec.to_string("deg", sep=":", pad=True),
                parallax=coord.distance.to(u.mas, equivalencies=u.parallax()).value,
                pm_ra=coord.pm_ra_cosdec.to(u.mas / u.year).value,
                pm_dec=coord.pm_dec.to(u.mas / u.year).value,
                frame=coord.frame.name,
                obstime=str(coord.obstime),
            )
    if coord is None:
        print("No coordinate information set; will only use header values.")

    ## darks
    have_darks = input("Do you have dark files? [Y/n]: ").strip().lower()
    if have_darks == "" or have_darks == "y":
        readline.set_completer(pathCompleter)
        cam1_path = input("Enter path to cam1 dark (optional): ").strip()
        cam1_path = None if cam1_path == "" else cam1_path
        cam2_path = None
        if template != "singlecam":
            if cam1_path is not None:
                cam2_default = cam1_path.replace("cam1", "cam2")
                cam2_path = input(f"Enter path to cam2 dark [{cam2_default}]: ").strip()
                if cam2_path == "":
                    cam2_path = cam2_default
            else:
                cam2_path = input(f"Enter path to cam2 dark (optional): ").strip()
                if cam2_path == "":
                    cam2_path = None
        readline.set_completer()
        tpl.calibrate.master_darks = CamFileInput(cam1=cam1_path, cam2=cam2_path)

    ## flats
    have_flats = input("Do you have flat files? [Y/n]: ").strip().lower()
    if have_flats == "" or have_flats == "y":
        readline.set_completer(pathCompleter)
        cam1_path = input("Enter path to cam1 flat (optional): ").strip()
        cam1_path = None if cam1_path == "" else cam1_path
        cam2_path = None
        if template != "singlecam" or cam1_path is None:
            if cam1_path is not None:
                cam2_default = cam1_path.replace("cam1", "cam2")
                cam2_path = input(f"Enter path to cam2 flat [{cam2_default}]: ").strip()
                if cam2_path == "":
                    cam2_path = cam2_default
            else:
                cam2_path = input(f"Enter path to cam2 flat (optional): ").strip()
                if cam2_path == "":
                    cam2_path = None
        readline.set_completer()
        tpl.calibrate.master_flats = CamFileInput(cam1=cam1_path, cam2=cam2_path)

    ## Coronagraph
    iwa_choices = ["36", "55", "92", "129"]

    readline.set_completer(createListCompleter(iwa_choices))
    have_coro = input("Did you use a coronagraph? [y/N]: ").strip().lower()
    if have_coro == "y":
        iwa = float(input(f"  Enter coronagraph IWA (mas) [{'/'.join(iwa_choices)}]: ").strip())
        tpl.coronagraph = CoronagraphOptions(iwa=iwa)
        readline.set_completer()

    ## Satellite spots
    _default = tpl.coronagraph is not None
    prompt = "Y/n" if _default else "y/N"
    have_satspot = input(f"Did you use satellite spots? [{prompt}]: ").strip().lower()
    if (_default and have_satspot == "") or have_satspot == "y":
        _default = 15.8
        radius = input(f"  Enter satspot radius (lam/D) [{_default}]: ").strip()
        spotrad = _default if radius == "" else float(radius)

        _default = 50
        amp = input(f"  Enter satspot amplitude (nm) [{_default}]: ").strip()
        spotamp = _default if amp == "" else float(amp)
        tpl.satspots = SatspotOptions(radius=spotrad, amp=spotamp)

    ## Frame centers
    set_frame_centers = input(f"Do you want to specify frame centers? [y/N]: ").strip().lower()
    if set_frame_centers == "y":
        cam1_ctr = cam2_ctr = None
        cam1_ctr_input = input("Enter comma-separated center for cam1 (x, y) (optional): ").strip()
        if cam1_ctr_input != "":
            cam1_ctr = _parse_cam_center(cam1_ctr_input)
        if template != "singlecam" or cam1_ctr is None:
            if cam1_ctr is not None:
                cam2_ctr_input = input(
                    f"Enter comma-separated center for cam2 (x, y) [{cam1_ctr}]: "
                ).strip()
                if cam2_ctr_input == "":
                    cam2_ctr = cam1_ctr
                else:
                    cam2_ctr = _parse_cam_center(cam2_ctr_input)
            else:
                cam2_ctr_input = input(
                    f"Enter comma-separated center for cam2 (x, y) (optional): "
                ).strip()
                if cam2_ctr_input == "":
                    cam2_ctr = None
                else:
                    cam2_ctr = _parse_cam_center(cam2_ctr_input)
        tpl.frame_centers = CamCtrOption(cam1=cam1_ctr, cam2=cam2_ctr)

    ## Frame selection
    do_frame_select = input(f"Would you like to do frame selection? [y/N]: ").strip().lower()
    if do_frame_select == "y":
        cutoff = float(
            input("  Enter a cutoff quantile (0 to 1, larger means more discarding): ").strip()
        )

        metric_choices = ["normvar", "l2norm", "peak"]
        readline.set_completer(createListCompleter(metric_choices))
        frame_select_metric = (
            input(f"  Choose a frame selection metric ([normvar]/l2norm/peak): ").strip().lower()
        )
        readline.set_completer()
        if frame_select_metric == "":
            frame_select_metric = "normvar"
        tpl.frame_select = FrameSelectOptions(
            cutoff=cutoff,
            metric=frame_select_metric,
            output_directory=DEFAULT_DIRS[FrameSelectOptions],
        )
    else:
        tpl.frame_select = None

    ## Registration
    do_register = input(f"Would you like to do frame registration? [Y/n]: ").strip().lower()
    if do_register == "y" or do_register == "":
        method_choices = ["com", "peak", "dft", "moffat", "gaussian", "airy"]
        readline.set_completer(createListCompleter(method_choices))
        register_method = (
            input(f"  Choose a registration method ([com]/peak/dft/moffat/gaussian/airy): ")
            .strip()
            .lower()
        )
        readline.set_completer()
        if register_method == "":
            register_method = "com"
        opts = RegisterOptions(
            method=register_method, output_directory=DEFAULT_DIRS[RegisterOptions]
        )
        if register_method == "dft":
            dft_factor_input = input("    Enter DFT upsample factor (default is 1): ").strip()
            opts.dft_factor = 1 if dft_factor_input == "" else int(dft_factor_input)

        smooth_input = input(f"  Smooth data before measurement? [y/N]: ").strip().lower()
        opts.smooth = smooth_input == "y"
        tpl.register = opts
    else:
        tpl.register = None

    ## Collapsing
    do_collapse = input(f"Would you like to collapse your data? [Y/n]: ").strip().lower()
    if do_collapse == "y" or do_collapse == "":
        collapse_choices = ["median", "mean", "varmean", "biweight"]
        readline.set_completer(createListCompleter(collapse_choices))
        collapse_method = (
            input("  Choose a collapse method ([median]/mean/varmean/biweight): ").strip().lower()
        )
        readline.set_completer()
        if collapse_method == "":
            collapse_method = "median"
        tpl.collapse = CollapseOptions(
            collapse_method, output_directory=DEFAULT_DIRS[CollapseOptions]
        )
    else:
        tpl.collapse = None

    tpl.to_file(config)

    return config


def _parse_cam_center(input_string):
    toks = input_string.replace(" ", "").split(",")
    ctr = list(map(float, toks))
    return ctr


########## check ##########


@click.command(
    name="check",
    short_help="Check raw data for problems",
    help="Checks each file to see if it can be opened (by calling fits.open) and whether there are any empty slices. Creates two text files with the selected and rejected filenames if any bad files are found.",
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
@click.option("--ext", "-e", default=0, help="HDU extension")
def check(filenames, num_proc, quiet):
    filenames = np.asarray(filenames, dtype=str)
    valid_files = check_files(filenames, num_proc=num_proc, quiet=quiet)
    cnt_valid = np.sum(valid_files)
    cnt_invalid = len(valid_files) - cnt_valid
    if cnt_invalid == 0:
        print("No invalid files found.")
        return

    print(f"{cnt_invalid} invalid files found (either couldn't be opened or have empty slices)")
    mask = np.asarray(valid_files, dtype=bool)
    accept_files = filenames[mask]
    accept_path = Path.cwd() / "files_select.txt"
    with accept_path.open("w") as fh:
        fh.writelines("\n".join(str(f) for f in accept_files))

    reject_files = filenames[~mask]
    reject_path = Path.cwd() / "files_reject.txt"
    with reject_path.open("w") as fh:
        fh.writelines("\n".join(str(f) for f in reject_files))

    print(str(accept_path))
    print(str(reject_path))


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
def run(config, filenames, num_proc, quiet):
    # make sure versions match within SemVar
    pipeline = Pipeline.from_file(config)
    if not check_version(pipeline.version, dpp.__version__):
        raise ValueError(
            f"Input pipeline version ({pipeline.version}) is not compatible with installed version of `vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade {config}`."
        )
    pipeline.run(filenames, num_proc=num_proc, quiet=quiet)


########## table ##########


@click.command(
    name="table",
    short_help="Create CSV from headers",
    help="Go through each file and combine the header information into a single CSV.",
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=(Path.cwd() / "header_table.csv").name,
    help="Output path.",
    show_default=True,
)
@click.option(
    "--num-proc",
    "-j",
    default=DEFAULT_NPROC,
    type=click.IntRange(1, cpu_count()),
    help="Number of processes to use.",
    show_default=True,
)
@click.option("--ext", "-e", default=0, help="HDU extension")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Silence progress bars and extraneous logging.",
)
def table(filenames, output, ext, num_proc, quiet):
    # handle name clashes
    outpath = Path(output).resolve()
    if outpath.is_file():
        resp = input(f"{outpath.name} already exists in the output directory. Overwrite? [y/N]: ")
        if resp.strip().lower() != "y":
            return
    # tryparse ext as int
    try:
        ext = int(ext)
    except ValueError:
        pass
    df = header_table(filenames, ext=ext, num_proc=num_proc, quiet=quiet)
    df.to_csv(outpath)


########## upgrade ##########


@click.command(
    name="upgrade",
    short_help="Upgrade configuration file",
    help=f"Tries to automatically upgrade a configuration file to the current version ({dpp.__version__}), prompting where necessary.",
)
@click.argument(
    "config",
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    callback=abort_if_false,
    prompt="Are you sure you want to modify your configuration in-place?",
    help="Output path.",
)
def upgrade(config, output):
    with open(config, "rb") as fh:
        input_toml = tomli.load(fh)
    output_config = upgrade_config(input_toml)
    outpath = config if output is None else output
    output_config.to_file(outpath)
    return outpath


########## main ##########


@tui()
@click.group(name="main")
@click.version_option(dpp.__version__, "--version", "-v", prog_name="vampires_dpp")
def main():
    pass


# add sub-commands
main.add_command(sort_raw)
main.add_command(prep)
main.add_command(check)
main.add_command(table)
main.add_command(upgrade)
main.add_command(run)
main.add_command(new_config)

if __name__ == "__main__":
    main()
