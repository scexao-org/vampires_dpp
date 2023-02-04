import logging
import multiprocessing as mp
import shutil
from argparse import ArgumentParser
from pathlib import Path

from astropy.io import fits
from serde.toml import to_toml

import vampires_dpp as vpp
from vampires_dpp.organization import header_table
from vampires_dpp.pipeline.config import CoronagraphOptions, SatspotOptions
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import *

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# default "main" run function
def run(args):
    path = Path(args.config)
    pipeline = Pipeline.from_file(path)
    # log file in local directory and output directory
    # pipeline.logger.setLevel(logging.DEBUG)
    # stream_handle = logging.StreamHandler()
    # stream_handle.setLevel(logging.INFO)
    # stream_handle.setFormatter(formatter)
    # pipeline.logger.addHandler(stream_handle)
    # log_filename = f"{path.stem}_debug.log"
    # log_files = path.parent / log_filename, pipeline.output_dir / log_filename
    # for log_file in log_files:
    #     file_handle = logging.FileHandler(log_file, mode="w")
    #     file_handle.setLevel(logging.DEBUG)
    #     file_handle.setFormatter(formatter)
    #     pipeline.logger.addHandler(file_handle)
    # run pipeline
    pipeline.run()


# set up command line arguments
parser = ArgumentParser()
parser.add_argument("--version", action="store_true", help="print version information")
subparser = parser.add_subparsers(help="command to run")

run_parser = subparser.add_parser("run", aliases="r", help="run the data processing pipeline")
run_parser.add_argument("config", help="path to configuration file")
run_parser.set_defaults(func=run)


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

    t.name = args.name
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
    choices=("singlecam", "pdi", "all"),
    help="template configuration to make",
)
new_parser.add_argument("-n", "--name", default="", help="name of configuration")
new_parser.add_argument(
    "-c", "--coronagraph", dest="iwa", type=float, help="if coronagraphic, specify IWA (mas)"
)
new_parser.add_argument("-s", "--show", action="store_true", help="display generated TOML")
new_parser.set_defaults(func=new_config)

sort_parser = subparser.add_parser("sort", aliases="s", help="sort raw VAMPIRES data")
sort_parser.add_argument("filenames", nargs="+", help="FITS files to sort")
sort_parser.add_argument(
    "-o", "--output", help="output directory, if not specified will use current working directory"
)
sort_parser.add_argument(
    "-c", "--copy", action="store_true", help="copy files instead of moving them"
)
sort_parser.set_defaults(func=sort)
sort_parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=min(mp.cpu_count(), 8),
    help="number of processers to use for multiprocessing (default is %(default)d)",
)


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
    description="Go through each file and combine the header information into a single CSV."
)
table_parser.add_argument("filenames", nargs="+", help="FITS files to parse headers from")
table_parser.add_argument(
    "-o",
    "--output",
    default="header_table.csv",
    help="Output CSV filename (default is '%(default)s')",
)
table_parser.add_argument("-e", "--ext", help="FITS extension/HDU to use")
table_parser.add_argument(
    "-j",
    "--num-proc",
    default=min(8, mp.cpu_count()),
    type=int,
    help="Number of processes to use for multi-processing (default is %(default)d)",
)
table_parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="silence the progress bar",
)
table_parser.set_default(func=table)


def main():
    args = parser.parse_args()
    if args.version:
        return vpp.__version__
    args.func(args)


if __name__ == "__main__":
    main()
