import glob
import logging
import os
import readline
from argparse import ArgumentParser
from pathlib import Path

import astropy.units as u
import numpy as np

import vampires_dpp as vpp
from vampires_dpp.calibration import make_master_dark, make_master_flat
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
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import (
    DEFAULT_DIRS,
    VAMPIRES_BLANK,
    VAMPIRES_PDI,
    VAMPIRES_SDI,
    VAMPIRES_SINGLECAM,
)
from vampires_dpp.wcs import get_gaia_astrometry

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up command line arguments
parser = ArgumentParser(prog="dpp")
parser.add_argument("-v", "--version", action="store_true", help="print version information")
subparser = parser.add_subparsers(help="command to run")

########## sort ##########


def sort(args):
    outdir = args.output if args.output else Path.cwd()
    try:
        ext = int(args.ext)
    except ValueError:
        ext = args.ext
    sort_files(
        args.filenames,
        copy=args.copy,
        ext=ext,
        output_directory=outdir,
        num_proc=args.num_proc,
        quiet=args.quiet,
    )


sort_parser = subparser.add_parser(
    "sort",
    aliases="s",
    help="sort raw VAMPIRES data",
    description="Sorts raw data based on the data type. This will either use the `DATA-TYP` header value or the `U_OGFNAM` header, depending on when your data was taken.",
)
sort_parser.add_argument("filenames", nargs="+", help="FITS files to sort")
sort_parser.add_argument(
    "-o", "--output", help="output directory, if not specified will use current working directory"
)
sort_parser.add_argument(
    "-c", "--copy", action="store_true", help="copy files instead of moving them"
)
sort_parser.add_argument("-e", "--ext", default=0, help="FITS extension/HDU to use")
sort_parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=DEFAULT_NPROC,
    help="number of processors to use for multiprocessing (default is %(default)d)",
)
sort_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="silence the progress bar",
)
sort_parser.set_defaults(func=sort)

########## calib ##########


def calib(args):
    outdir = args.output if args.output else Path.cwd()

    master_darks = master_flats = None
    if args.darks is not None:
        master_darks = make_master_dark(
            args.darks,
            collapse=args.collapse,
            force=args.force,
            output_directory=outdir,
            quiet=args.quiet,
            num_proc=args.num_proc,
        )
    if args.flats is not None:
        master_flats = make_master_flat(
            args.flats,
            collapse=args.collapse,
            master_darks=master_darks,
            force=args.force,
            output_directory=outdir,
            quiet=args.quiet,
            num_proc=args.num_proc,
        )


calib_parser = subparser.add_parser(
    "calib",
    aliases="c",
    help="create calibration files",
    description="Create calibration files from darks and flats.",
)
calib_parser.add_argument("--darks", nargs="*", help="FITS files to use as dark frames")
calib_parser.add_argument("--flats", nargs="*", help="FITS files to use as flat frames")
calib_parser.add_argument(
    "-c", "--collapse", default="median", choices=("median", "mean", "varmean", "biweight")
)
calib_parser.add_argument(
    "-o", "--output", help="output directory, if not specified will use current working directory"
)
calib_parser.add_argument(
    "-f", "--force", action="store_true", help="Force recomputation and overwrite existing files."
)
calib_parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=DEFAULT_NPROC,
    help="number of processors to use for multiprocessing (default is %(default)d)",
)
calib_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="silence the progress bar",
)
calib_parser.set_defaults(func=calib)

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


def new_config(args):
    path = Path(args.config)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")

    ## check if output file exists
    if path.is_file():
        response = (
            input(
                f"{path.name} already exists in output directory, would you like to overwrite it? [y/N]: "
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
    name_guess = path.stem
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

    ## save output TOML
    toml_str = tpl.to_toml()
    with path.open("w") as fh:
        fh.write(toml_str)

    return path


def _parse_cam_center(input_string):
    toks = input_string.replace(" ", "").split(",")
    ctr = list(map(float, toks))
    return ctr


new_parser = subparser.add_parser("new", aliases="n", help="generate configuration files")
new_parser.add_argument("config", help="path to configuration file")
new_parser.set_defaults(func=new_config)

########## check ##########


def check(args):
    filenames = np.asarray(args.filenames, dtype=str)
    valid_files = check_files(filenames, num_proc=args.num_proc, quiet=args.quiet)
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


check_parser = subparser.add_parser(
    "check",
    help="check raw data for problems",
    description="Checks each file to see if it can be opened (by calling fits.open) and whether there are any empty slices. Creates two text files with the selected and rejected filenames if any bad files are found.",
)
check_parser.add_argument("filenames", nargs="*", help="FITS files to check")
check_parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=DEFAULT_NPROC,
    help="number of processors to use for multiprocessing (default is %(default)d)",
)
check_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="silence the progress bar",
)
check_parser.set_defaults(func=check)


########## run ##########


def run(args):
    path = Path(args.config)
    pipeline = Pipeline.from_file(path)
    pipeline.run(args.filenames, num_proc=args.num_proc)


run_parser = subparser.add_parser("run", aliases="r", help="run the data processing pipeline")
run_parser.add_argument("config", help="path to configuration file")
run_parser.add_argument("filenames", nargs="*", help="FITS files to run through pipeline")
run_parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=DEFAULT_NPROC,
    help="number of processors to use for multiprocessing (default is %(default)d)",
)
run_parser.set_defaults(func=run)

########## table ##########


def table(args):
    # handle name clashes
    outpath = Path(args.output).resolve()
    if outpath.is_file():
        resp = input(f"{outpath.name} already exists in the output directory. Overwrite? [y/N]: ")
        if resp.strip().lower() != "y":
            return
    # tryparse ext as int
    try:
        ext = int(args.ext)
    except ValueError:
        ext = args.ext
    df = header_table(args.filenames, ext=ext, num_proc=args.num_proc, quiet=args.quiet)
    df.to_csv(outpath)


table_parser = subparser.add_parser(
    "table",
    aliases="t",
    help="create CSV from headers",
    description="Go through each file and combine the header information into a single CSV.",
)
table_parser.add_argument("filenames", nargs="+", help="FITS files to parse headers from")
table_parser.add_argument(
    "-o",
    "--output",
    default="header_table.csv",
    help="Output CSV filename (default is '%(default)s')",
)
table_parser.add_argument("-e", "--ext", default=0, help="FITS extension/HDU to use")
table_parser.add_argument(
    "-j",
    "--num-proc",
    default=DEFAULT_NPROC,
    type=int,
    help="Number of processes to use for multi-processing (default is %(default)d)",
)
table_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="silence the progress bar",
)
table_parser.set_defaults(func=table)

########## main ##########


def main():
    args = parser.parse_args()
    if args.version:
        return vpp.__version__
    if hasattr(args, "func"):
        return args.func(args)
    # no inputs, print help
    parser.print_help()


if __name__ == "__main__":
    main()
