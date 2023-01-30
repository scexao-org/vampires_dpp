import logging
import multiprocessing as mp
import shutil
from argparse import ArgumentParser
from pathlib import Path

from astropy.io import fits
from serde.toml import to_toml
from tqdm.auto import tqdm

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


def sort_file(filename, outdir, copy=False):
    path = Path(filename)
    with fits.open(path) as hdus:
        header = hdus[0].header
        datayp = header["DATA-TYP"]
        obj = header["OBJECT"]

    match datayp:
        case "OBJECT":
            foldname = outdir / obj.replace(" ", "_") / "raw"
        case "DARK":
            foldname = outdir / "darks" / "raw"
        case "SKY":
            foldname = outdir / "skies" / "raw"
        case "FLAT":
            foldname = outdir / "flats" / "raw"
        case "COMPARISON":
            foldname = outdir / "pinholes" / "raw"
        case _:
            foldname = outdir
    foldname.mkdir(parents=True, exist_ok=True)
    newname = foldname / path.name
    if copy:
        shutil.copy(path, newname)
    else:
        path.replace(newname)


def create(args):
    path = Path(args.config)
    match args.template:
        case "singlecam":
            t = VAMPIRES_SINGLECAM
        case "pdi":
            t = VAMPIRES_PDI
        case "halpha":
            t = VAMPIRES_HALPHA

    t.name = args.name
    if args.coronagraph:
        t.coronagraph = CoronagraphOptions(args.coronagraph)
        t.satspot = SatspotOptions()
        t.coregister.method = "com"

    toml_str = to_toml(t)
    if args.preview:
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
    help="number of processers to use for multiprocessing",
)
subp = parser.add_subparsers()
sort_parser = subp.add_parser("sort", help="sort raw VAMPIRES data")
sort_parser.add_argument("filenames", nargs="+", help="FITS files to sort")
sort_parser.add_argument(
    "-o", "--output", help="output directory, if not specified will use current working directory"
)
sort_parser.add_argument(
    "-c", "--copy", action="store_true", help="copy files instead of moving them"
)
sort_parser.set_defaults(func=sort)

create_parser = subp.add_parser("create", help="generate configuration files")
create_parser.add_argument("config", help="path to configuration file")
create_parser.add_argument(
    "-t", "--template", choices=("singlecam", "pdi"), help="template configuration to make"
)
create_parser.add_argument("-n", "--name", default="", help="name of configuration")
create_parser.add_argument(
    "-c", "--coronagraph", type=float, help="if coronagraphic, specify IWA (mas)"
)
create_parser.add_argument("-p", "--preview", action="store_true", help="display generated TOML")
create_parser.set_defaults(func=create)

run_parser = subp.add_parser("run", help="run the data processing pipeline")
run_parser.add_argument("config", help="path to configuration file")
run_parser.set_defaults(func=run)


def main():
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
