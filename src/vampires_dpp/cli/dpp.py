import glob
import logging
import os
import readline
from argparse import ArgumentParser
from pathlib import Path

import astropy.units as u
from serde.toml import to_toml

import vampires_dpp as vpp
from vampires_dpp.calibration import make_master_dark, make_master_flat
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.organization import header_table, sort_files
from vampires_dpp.pipeline.config import (
    CamCtrOption,
    CamFileInput,
    CoordinateOptions,
    CoronagraphOptions,
    SatspotOptions,
)
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import (
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


class TabCompleter:
    """
    A tab completer that can either complete from
    the filesystem or from a list.

    Partially taken from:
    http://stackoverflow.com/questions/5637124/tab-completion-in-pythons-raw-input
    """

    def pathCompleter(self, text, state):
        """
        This is the tab completer for systems paths.
        Only tested on *nix systems
        """
        readline.get_line_buffer().split()

        # replace ~ with the user's home dir. See https://docs.python.org/2/library/os.path.html
        if "~" in text:
            text = os.path.expanduser("~")

        # autocomplete directories with having a trailing slash
        if os.path.isdir(text):
            text += "/"

        return [x for x in glob.glob(text + "*")][state]

    def createListCompleter(self, ll):
        """
        This is a closure that creates a method that autocompletes from
        the given list.

        Since the autocomplete function can't be given a list to complete from
        a closure is used to create the listCompleter function with a list to complete
        from.
        """

        def listCompleter(text, state):
            line = readline.get_line_buffer()

            if not line:
                return [c + " " for c in ll][state]

            else:
                return [c + " " for c in ll if c.startswith(line)][state]

        return listCompleter


def new_config(args):
    path = Path(args.config)
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")

    completer = TabCompleter()

    iwa_choices = ["36", "55", "92", "129"]
    iwa_completer = completer.createListCompleter(iwa_choices)

    ## get template
    template_choices = ["singlecam", "pdi", "halpha"]
    template_completer = completer.createListCompleter(template_choices)

    readline.set_completer(template_completer)
    template = input(f"Choose a starting template [{'/'.join(template_choices)}]: ").strip().lower()
    match template:
        case "singlecam":
            tpl = VAMPIRES_SINGLECAM
        case "pdi":
            tpl = VAMPIRES_PDI
        case "halpha":
            tpl = VAMPIRES_SDI
        case _:
            raise ValueError(f"template not recognized {template}")
    readline.set_completer()

    ## get name
    name_guess = path.stem
    name = input(f"Path-friendly name for this reduction [{name_guess}]: ").strip().lower()
    tpl.name = name_guess if name == "" else name.replace(" ", "_").replace("/", "")

    ## get target
    object = input(f"SIMBAD-friendly object name (optional): ").strip()
    obj = None if object == "" else object
    if obj is not None:
        coord = None
        rad = 1
        cat = "dr3"
        while True:
            try:
                coord = get_gaia_astrometry(obj, catalog=cat, radius=rad)
                break
            except:
                print(f'Could not find {obj} in GAIA {cat.upper()} with {rad}" radius.')
                _input = input(
                    "Query different catalog (dr1/dr2/dr3), enter search radius in arcsec, or enter new object name (optional): "
                ).strip()
                match _input.lower():
                    case "":
                        break
                    case "dr1":
                        cat = "dr1"
                    case "dr2":
                        cat = "dr2"
                    case "dr3":
                        cat = "dr3"
                    case _:
                        try:
                            rad = float(_input)
                        except:
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
        readline.set_completer(completer.pathCompleter)
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
        readline.set_completer(completer.pathCompleter)
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
    have_coro = input("Did you use a coronagraph? [y/N]: ").strip().lower()
    if have_coro == "y":
        readline.set_completer(iwa_completer)
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
    set_framecenters = input(f"Do you want to specify frame centers? [y/N]: ").strip().lower()
    if set_framecenters == "y":
        cam1_ctr = cam2_ctr = None
        cam1_ctr_input = input("Enter comma-separated center for cam1 (x, y) (optional): ").strip()
        if cam1_ctr_input != "":
            toks = cam1_ctr_input.replace(" ", "").split(",")
            cam1_ctr = list(map(float, toks))
        if template != "singlecam" or cam1_ctr is None:
            if cam1_ctr is not None:
                cam2_ctr_input = input(
                    f"Enter comma-separated center for cam2 (x, y) [{cam1_ctr}]: "
                ).strip()
                if cam2_ctr_input == "":
                    cam2_ctr = cam1_ctr
                else:
                    toks = cam2_ctr_input.replace(" ", "").split(",")
                    cam2_ctr = list(map(float, toks))
            else:
                cam2_ctr_input = input(
                    f"Enter comma-separated center for cam2 (x, y) (optional): "
                ).strip()
                if cam2_ctr_input == "":
                    cam2_ctr = None
                else:
                    toks = cam2_ctr_input.replace(" ", "").split(",")
                    cam2_ctr = list(map(float, toks))
        tpl.frame_centers = CamCtrOption(cam1=cam1_ctr, cam2=cam2_ctr)

    toml_str = to_toml(tpl)

    if path.is_file():
        response = (
            input(
                f"{path.name} already exists in output directory, would you like to overwrite it? [y/N] "
            )
            .strip()
            .lower()
        )
        if response != "y":
            return

    with path.open("w") as fh:
        fh.write(toml_str)

    return path


new_parser = subparser.add_parser("new", aliases="n", help="generate configuration files")
new_parser.add_argument("config", help="path to configuration file")
new_parser.set_defaults(func=new_config)

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
