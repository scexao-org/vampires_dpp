import logging
from argparse import ArgumentParser
from pathlib import Path

from serde.toml import to_toml

import vampires_dpp as vpp
from vampires_dpp.calibration import make_master_dark, make_master_flat
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.organization import header_table, sort_files
from vampires_dpp.pipeline.config import (
    CoordinateOptions,
    CoronagraphOptions,
    SatspotOptions,
    u,
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


def new_config(args):
    path = Path(args.config)
    return interactive_config(path)
    # if args.interactive:
    # match args.template:
    #     case "singlecam":
    #         t = VAMPIRES_SINGLECAM
    #     case "pdi":
    #         t = VAMPIRES_PDI
    #     case "halpha":
    #         t = VAMPIRES_SDI
    #     case "all":
    #         t = VAMPIRES_MAXIMAL
    #     case _:
    #         raise ValueError(f"template not recognized {args.template}")
    # t.target = args.object
    # t.name = path.stem
    # if args.iwa:
    #     t.coronagraph = CoronagraphOptions(args.iwa)
    #     t.satspots = SatspotOptions()
    #     t.register.method = "com"

    # toml_str = to_toml(t)
    # if args.preview:
    #     # print(toml_str)
    #     print(f"{'-'*12} PREVIEW {path.name} {'-'*12}")
    #     print(toml_str)
    #     print(f"{'-'*12} END PREVIEW {'-'*12}")
    #     response = input(f"Would you like to save this configuration? [y/N] ").strip().lower()
    #     if response != "y":
    #         return

    # if path.is_file():
    #     response = (
    #         input(
    #             f"{path.name} already exists in output directory, would you like to overwrite it? [y/N] "
    #         )
    #         .strip()
    #         .lower()
    #     )
    #     if response != "y":
    #         return

    # with path.open("w") as fh:
    #     fh.write(toml_str)

    # return path


import readline
import sys
from os import environ


class ChoiceCompleter(object):  # Custom completer
    def __init__(self, options):
        self.options = sorted(options)

    def complete(self, text, state):
        if state == 0:  # on first trigger, build possible matches
            if not text:
                self.matches = self.options[:]
            else:
                self.matches = [s for s in self.options if s and s.startswith(text)]

        # return match indexed by state
        try:
            return self.matches[state]
        except IndexError:
            return None

    def display_matches(self, substitution, matches, longest_match_length):
        line_buffer = readline.get_line_buffer()
        columns = environ.get("COLUMNS", 80)

        print()

        tpl = "{:<" + str(int(max(map(len, matches)) * 1.2)) + "}"

        buffer = ""
        for match in matches:
            match = tpl.format(match[len(substitution) :])
            if len(buffer + match) > columns:
                print(buffer)
                buffer = ""
            buffer += match

        if buffer:
            print(buffer)

        print("> ", end="")
        print(line_buffer, end="")
        sys.stdout.flush()


readline.set_completer_delims(" \t\n;")
readline.parse_and_bind("tab: complete")

template_choices = ["singlecam", "pdi", "halpha"]
template_completer = ChoiceCompleter(template_choices)

iwa_choices = ["36", "55", "92", "129"]
iwa_completer = ChoiceCompleter(template_choices)


def interactive_config(path):

    ## get template
    readline.set_completer(template_completer.complete)
    readline.set_completion_display_matches_hook(template_completer.display_matches)
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
    target = input(f"SIMBAD-friendly target name (optional): ").strip()
    targ = None if target == "" else target
    if targ is not None:
        coord = None
        rad = 1
        cat = "dr3"
        while True:
            try:
                coord = get_gaia_astrometry(targ, catalog=cat, radius=rad)
                break
            except:
                print(f'Could not find {targ} in GAIA {cat.upper()} with {rad}" radius.')
                _input = input(
                    "Query different catalog (dr1/dr2/dr3), enter search radius in arcsec, or enter new target name (optional): "
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
                            targ = _input

        if coord is not None:
            tpl.coordinate = CoordinateOptions(
                target=targ,
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
        pass
        # TODO get darks

    ## flats
    have_flats = input("Do you have flat files? [Y/n]: ").strip().lower()
    if have_flats == "" or have_flats == "y":
        pass
        # TODO get flats

    ## Coronagraph
    have_coro = input("Did you use a coronagraph? [y/N]: ").strip().lower()
    if have_coro == "y":
        readline.set_completer(iwa_completer.complete)
        readline.set_completion_display_matches_hook(iwa_completer.display_matches)
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
new_parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Launch interactive generator (ignores other command line arguments)",
)
new_parser.add_argument(
    "-t",
    "--template",
    default="all",
    choices=("singlecam", "pdi", "halpha", "all"),
    help="template configuration to make",
)
new_parser.add_argument("-o", "--object", default="", help="SIMBAD-compatible target name")
new_parser.add_argument(
    "-c", "--coronagraph", dest="iwa", type=float, help="if coronagraphic, specify IWA (mas)"
)
new_parser.add_argument("-p", "--preview", action="store_true", help="preview generated TOML")
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
