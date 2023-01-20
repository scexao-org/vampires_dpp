from argparse import ArgumentParser
from pathlib import Path
import logging
from vampires_dpp.pipeline import Pipeline
import tqdm.auto as tqdm

# set up logging
formatter = logging.Formatter(
    "%(asctime)s|%(name)s|%(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# set up command line arguments
parser = ArgumentParser()
parser.add_argument("config", nargs="+", help="path to configuration file(s)")


def main():
    args = parser.parse_args()
    for config in tqdm.tqdm(args.config, desc="config"):
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
