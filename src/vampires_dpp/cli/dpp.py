import logging
from argparse import ArgumentParser
from pathlib import Path

from vampires_dpp.pipeline.config import CoronagraphOptions, SatelliteSpotOptions
from vampires_dpp.pipeline.templates import *

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up command line arguments
parser = ArgumentParser()
grp = parser.add_argument_group("create")
grp.add_argument("config", help="path to configuration file")
grp.add_argument(
    "-t", "--template", choices=("singlecam", "pdi"), help="template configuration to make"
)
grp.add_argument("-n", "--name", default="", help="name of configuration")
grp.add_argument("-c", "--coronagraph", type=float, help="if coronagraphic, specify IWA (mas)")

grp = parser.add_argument_group("run")
grp.add_argument("config", help="path to configuration file")


def create(args):
    match args.template:
        case "singlecam":
            t = VAMPIRES_SINGLECAM
        case "pdi":
            t = VAMPIRES_PDI

    t.name = args.name
    if args.coronagraph:
        t.coronagraph = CoronagraphOptions(args.coronagraph)
        t.satspot = SatelliteSpotOptions()


def main():
    args = parser.parse_args()
    for config in args.config:
        path = Path(config)
        pipeline = Pipeline.from_file(path)
        # set up logging - INFO in STDOUT, DEBUG in file
        # log file in local directory and output directory
        pipeline.logger.setLevel(logging.DEBUG)
        stream_handle = logging.StreamHandler()
        stream_handle.setLevel(logging.INFO)
        stream_handle.setFormatter(formatter)
        pipeline.logger.addHandler(stream_handle)
        log_filename = f"{path.stem}_debug.log"
        log_files = path.parent / log_filename, pipeline.output_dir / log_filename
        for log_file in log_files:
            file_handle = logging.FileHandler(log_file, mode="w")
            file_handle.setLevel(logging.DEBUG)
            file_handle.setFormatter(formatter)
            pipeline.logger.addHandler(file_handle)
        # run pipeline
        pipeline.run()


if __name__ == "__main__":
    main()
