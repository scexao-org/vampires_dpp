import logging
import multiprocessing as mp
import shutil
from argparse import ArgumentParser
from pathlib import Path

from astropy.io import fits
from serde.toml import to_toml
from tqdm.auto import tqdm

import vampires_dpp as vpp
from vampires_dpp.pipeline.config import CoronagraphOptions, SatspotOptions
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import *

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up commands for parser to dispatch to
def sort(args):
    if args.output:
        outdir = Path(args.output)
    else:
        outdir = Path.cwd()
    jobs = []
    with mp.Pool(args.num_proc) as pool:
        for filename in args.filenames:
            kwds = dict(outdir=outdir, copy=args.copy)
            jobs.append(pool.apply_async(sort_file, args=(filename,), kwds=kwds))

        for job in tqdm(jobs, desc="Sorting files"):
            job.get()


def foldername_new(outdir, header):
    match header["DATA-TYP"]:
        case "OBJECT":
            foldname = outdir / header["OBJECT"].replace(" ", "_") / "raw"
        case "DARK":
            foldname = outdir / "darks"
        case "SKYFLAT":
            foldname = outdir / "skies"
        case "FLAT" | "DOMEFLAT":
            foldname = outdir / "flats"
        case "COMPARISON":
            foldname = outdir / "pinholes"
        case _:
            foldname = outdir

    return foldname


def foldername_old(outdir, path, header):
    name = header.get("U_OGFNAM", path.name)
    if "dark" in name:
        foldname = outdir / "darks"
    elif "skies" in name or "sky" in name:
        foldname = outdir / "skies"
    elif "flat" in name:
        foldname = outdir / "flats"
    elif "pinhole" in name:
        foldname = outdir / "pinholes"
    else:
        foldname = outdir / header["OBJECT"].replace(" ", "_") / "raw"

    return foldname


def sort_file(filename, outdir, copy=False):
    path = Path(filename)
    with fits.open(path) as hdus:
        header = hdus[0].header

    if header["DATA-TYP"] == "ACQUISITION":
        foldname = foldername_old(outdir, path, header)
    else:
        foldname = foldername_new(outdir, header)

    newname = foldname / path.name
    foldname.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy(path, newname)
    else:
        path.replace(newname)


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


def run(args):
    pass
    # path = Path(args.config)
    # pipeline = Pipeline.from_file(path)
    # # set up logging - INFO in STDOUT, DEBUG in file
    # # log file in local directory and output directory
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
    # # run pipeline
    # pipeline.run()


# set up command line arguments
parser = ArgumentParser()
parser.add_argument(
    "-j",
    "--num-proc",
    type=int,
    default=min(mp.cpu_count(), 8),
    help="number of processers to use for multiprocessing (default is %(default)d)",
)
parser.add_argument("--version", action="store_true", help="print version information")
subp = parser.add_subparsers(help="command to run")
sort_parser = subp.add_parser("sort", aliases="s", help="sort raw VAMPIRES data")
sort_parser.add_argument("filenames", nargs="+", help="FITS files to sort")
sort_parser.add_argument(
    "-o", "--output", help="output directory, if not specified will use current working directory"
)
sort_parser.add_argument(
    "-c", "--copy", action="store_true", help="copy files instead of moving them"
)
sort_parser.set_defaults(func=sort)

new_parser = subp.add_parser("new", aliases="n", help="generate configuration files")
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

run_parser = subp.add_parser("run", aliases="r", help="run the data processing pipeline")
run_parser.add_argument("config", help="path to configuration file")
run_parser.set_defaults(func=run)


def main():
    args = parser.parse_args()
    if args.version:
        return vpp.__version__
    args.func(args)


if __name__ == "__main__":
    main()
