from argparse import ArgumentParser
from pathlib import Path
import logging
import toml
import pprint
import tqdm.auto as tqdm
from astropy.io import fits

import vampires_dpp as vpp
from vampires_dpp.calibration import make_dark_file

# set up logging
formatter = "%(asctime)s|%(name)s - %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    level=logging.INFO,
    format=formatter,
    datefmt=datefmt,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("VPP")

# set up command line arguments
parser = ArgumentParser()
parser.add_argument("config", help="path to configuration file")


def main():
    args = parser.parse_args()

    logger.debug(f"loading config from {args.config}")
    config = toml.load(args.config)

    # make sure versions match
    if config["version"] != vpp.__version__:
        raise ValueError(
            f"Input pipeline version ({config['version']}) does not match installed version of `vampires_dpp` ({vpp.__version__}), and is therefore incompatible."
        )

    # set up paths
    root = Path(config["directory"])
    output = Path(config["output_directory"])
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)

    ## Step 1: Fix headers and calibrate
    logger.info("Starting data calibration")
    outdir = output / config["calibration"]["output_directory"]
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    ## Step 1a: create master dark
    if config["calibration"]["darks"]:
        dark_filenames = parse_filenames(
            root, config["calibration"]["darks"]["filenames"]
        )
        skip_darks = not config["calibration"]["darks"]["force"]
        master_darks = []
        for dark in tqdm.tqdm(dark_filenames, desc="Making master darks"):
            outname = outdir / f"{dark.stem}_collapsed{dark.suffix}"
            master_darks.append(make_dark_file(dark, outname, skip=skip_darks))
    else:
        master_darks = [None, None]

    ## Step 1b: calibrate files and fix headers
    filenames = parse_filenames(root, config["calibration"]["filenames"])
    logger.info(f"Calibrating {len(filenames)} files.")
    # for filename in tqdm.tqdm(filenames, desc="Calibrating files"):
    #     pass


def parse_filenames(root, filenames):
    if isinstance(filenames, str):
        path = Path(filenames)
        if path.is_file():
            # is a file with a list of filenames
            fh = path.open("r")
            paths = [Path(f) for f in fh.readlines()]
            fh.close()
        else:
            # is a globbing expression
            paths = list(root.glob(filenames))

    else:
        # is a list of filenames
        paths = [root / f for f in filenames]

    return paths


if __name__ == "__main__":
    main()
