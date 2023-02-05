import logging
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

from serde.toml import to_toml

import vampires_dpp as vpp
from vampires_dpp.calibration import make_master_dark, make_master_flat
from vampires_dpp.constants import DEFAULT_NPROC
from vampires_dpp.organization import header_table, sort_files
from vampires_dpp.pipeline.config import CoronagraphOptions, SatspotOptions
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import *

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up command line arguments
parser = ArgumentParser()
parser.add_argument("--version", action="store_true", help="print version information")
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
    match args.template:
        case "singlecam":
            t = VAMPIRES_SINGLECAM
        case "pdi":
            t = VAMPIRES_PDI
        case "halpha":
            t = VAMPIRES_HALPHA
        case "all":
            t = VAMPIRES_MAXIMAL
        case _:
            raise ValueError(f"template not recognized {args.template}")
    t.target = args.object
    t.name = t.target
    if args.iwa:
        t.coronagraph = CoronagraphOptions(args.iwa)
        t.satspots = SatspotOptions()
        t.coregister.method = "com"

    toml_str = to_toml(t)
    if args.show:
        print(toml_str)
    with path.open("w") as fh:
        fh.write(toml_str)

    return path


new_parser = subparser.add_parser("new", aliases="n", help="generate configuration files")
new_parser.add_argument("config", help="path to configuration file")
new_parser.add_argument(
    "-t",
    "--template",
    required=True,
    choices=("singlecam", "pdi", "halpha", "all"),
    help="template configuration to make",
)
new_parser.add_argument("-o", "--object", default="", help="SIMBAD-compatible target name")
new_parser.add_argument(
    "-c", "--coronagraph", dest="iwa", type=float, help="if coronagraphic, specify IWA (mas)"
)
new_parser.add_argument("-s", "--show", action="store_true", help="display generated TOML")
new_parser.set_defaults(func=new_config)

########## run ##########


def run(args):
    path = Path(args.config)
    pipeline = Pipeline.from_file(path)
    pipeline.run(num_proc=args.num_proc)


run_parser = subparser.add_parser("run", aliases="r", help="run the data processing pipeline")
run_parser.add_argument("config", help="path to configuration file")
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
