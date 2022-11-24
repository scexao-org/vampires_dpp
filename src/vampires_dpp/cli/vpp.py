from argparse import ArgumentParser
from pathlib import Path
import logging
from vampires_dpp.pipeline import Pipeline

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up command line arguments
parser = ArgumentParser()
parser.add_argument("config", help="path to configuration file")


def main():
    args = parser.parse_args()
    path = Path(args.config)
    pipeline = Pipeline.from_file(path)
    # set up logging
    pipeline.logger.setLevel(logging.DEBUG)
    stream_handle = logging.StreamHandler()
    stream_handle.setLevel(logging.INFO)
    stream_handle.setFormatter(formatter)
    pipeline.logger.addHandler(stream_handle)
    log_file = f"{path.stem}_debug.log"
    file_handle = logging.FileHandler(log_file, mode="w")
    file_handle.setLevel(logging.DEBUG)
    file_handle.setFormatter(formatter)
    pipeline.logger.addHandler(file_handle)
    # run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
