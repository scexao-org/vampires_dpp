import logging
from argparse import ArgumentParser
from pathlib import Path

from serde.toml import to_toml

from vampires_dpp.pipeline.config import CoronagraphOptions, SatspotOptions
from vampires_dpp.pipeline.pipeline import Pipeline
from vampires_dpp.pipeline.templates import *

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up commands for parser to dispatch to
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
subp = parser.add_subparsers()
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

run_parser = subp.add_parser("run", help="generate configuration files")
run_parser.add_argument("config", help="path to configuration file")
run_parser.set_defaults(func=run)


def main():
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
