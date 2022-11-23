from argparse import ArgumentParser
from pathlib import Path
import logging
import toml
from packaging import version
import tqdm.auto as tqdm
from astropy.io import fits
import numpy as np

import vampires_dpp as vpp
from vampires_dpp.calibration import make_dark_file, make_flat_file, calibrate
from vampires_dpp.fixes import fix_header
from vampires_dpp.frame_selection import measure_metric_file, frame_select_file
from vampires_dpp.image_registration import measure_offsets, register_file
from vampires_dpp.satellite_spots import lamd_to_pixel

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


def check_version(config, vpp):
    config_maj, config_min, config_pat = version.parse(config).release
    vpp_maj, vpp_min, vpp_pat = version.parse(vpp).release
    return config_maj == vpp_maj and config_min == vpp_min and vpp_pat >= config_pat


def main():
    args = parser.parse_args()

    logger.debug(f"loading config from {args.config}")
    config = toml.load(args.config)

    # make sure versions match within SemVar
    if not check_version(config["version"], vpp.__version__):
        raise ValueError(
            f"Input pipeline version ({config['version']}) does not match installed version of `vampires_dpp` ({vpp.__version__}), and is therefore incompatible."
        )

    # set up paths
    root = Path(config["directory"])
    output = Path(config.get("output_directory", root))
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)

    if "frame_centers" in config:
        frame_centers = [np.array(c)[::-1] for c in config["frame_centers"]]
    else:
        frame_centers = [None, None]

    ## Step 1: Fix headers and calibrate
    logger.info("Starting data calibration")
    outdir = output / config["calibration"].get("output_directory", "")
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    ## Step 1a: create master dark
    if "darks" in config["calibration"]:
        dark_filenames = parse_filenames(
            root, config["calibration"]["darks"]["filenames"]
        )
        skip_darks = not config["calibration"]["darks"].get("force", False)
        master_darks = []
        for dark in tqdm.tqdm(dark_filenames, desc="Making master darks"):
            outname = outdir / f"{dark.stem}_collapsed{dark.suffix}"
            dark_frame = fits.getdata(make_dark_file(dark, outname, skip=skip_darks))
            master_darks.append(dark_frame)
    else:
        master_darks = [None, None]

    ## Step 1b: create master flats
    if "flats" in config["calibration"]:
        dark_filenames = parse_filenames(
            root, config["calibration"]["flats"]["filenames"]
        )
        skip_flats = not config["calibration"]["flats"].get("force", False)
        master_flats = []
        for dark, flat in zip(
            master_darks, tqdm.tqdm(dark_filenames, desc="Making master flats")
        ):
            outname = outdir / f"{flat.stem}_collapsed{flat.suffix}"
            flat_frame = fits.getdata(
                make_flat_file(flat, dark, outname, skip=skip_flats)
            )
            master_flats.append(flat_frame)
    else:
        master_flats = [None, None]

    ## Step 1c: calibrate files and fix headers
    filenames = parse_filenames(root, config["calibration"]["filenames"])
    N_files = len(filenames)
    skip_calib = not config["calibration"].get("force", False)
    working_files = []
    test_frames = [None, None]
    test_filter = None
    for filename in tqdm.tqdm(filenames, desc="Calibrating files"):
        outname = outdir / f"{filename.stem}_calib{filename.suffix}"
        if skip_calib and outname.is_file():
            working_files.append(outname)
            continue
        cube, header = fits.getdata(filename, header=True)
        header = fix_header(header)
        if test_filter is None:
            test_filter = header["U_FILTER"]
        if header["U_CAMERA"] == 1:
            calib_cube = calibrate(
                cube, discard=2, dark=master_darks[0], flat=master_flats[0], flip=True
            )
            if test_frames[0] is None:
                test_frames[0] = np.mean(calib_cube, axis=0)
        else:
            calib_cube = calibrate(
                cube, discard=2, dark=master_darks[1], flat=master_flats[1], flip=False
            )
            if test_frames[1] is None:
                test_frames[1] = np.mean(calib_cube, axis=0)
        fits.writeto(outname, calib_cube, header, overwrite=True)
        working_files.append(outname)

    logger.info("Data calibration completed")

    ## Step 2: Frame selection
    if "frame_selection" in config:
        logger.info("Performing frame selection")
        outdir = output / config["frame_selection"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        skip_select = not config["frame_selection"].get("force", False)
        metric_files = []
        ## 2a: measure metrics
        for i in tqdm.trange(N_files, desc="Measuring frame selection metric"):
            filename = working_files[i]
            header = fits.getheader(filename)
            cam_idx = int(header["U_CAMERA"] - 1)
            outname = outdir / f"{filename.stem}_metrics.csv"
            window = config["frame_selection"].get("window_size", 30)
            if "coronagraph" in config:
                r = lamd_to_pixel(
                    config["coronagraph"]["satellite_spots"]["radius"],
                    header["U_FILTER"],
                )
                ang = config["coronagraph"]["satellite_spots"].get("angle", -4)
                metric_file = measure_metric_file(
                    filename,
                    center=frame_centers[cam_idx],
                    coronagraphic=True,
                    radius=r,
                    theta=ang,
                    window=window,
                    metric=config["frame_selection"].get("metric", "l2norm"),
                    output=outname,
                    skip=skip_select,
                )
            else:
                metric_file = measure_metric_file(
                    filename,
                    center=frame_centers[cam_idx],
                    window=window,
                    metric=config["frame_selection"].get("metric", "l2norm"),
                    output=outname,
                    skip=skip_select,
                )
            metric_files.append(metric_file)

        ## 2b: perform frame selection
        quantile = config["frame_selection"].get("q", 0)
        if quantile > 0:
            for i in tqdm.trange(N_files, desc="Discarding frames"):
                filename = working_files[i]
                metric_file = metric_files[i]
                outname = outdir / f"{filename.stem}_cut{filename.suffix}"
                working_files[i] = frame_select_file(
                    filename,
                    metric_file,
                    q=quantile,
                    output=outname,
                    skip=skip_select,
                )

    logger.info("Frame selection complete")

    ## 3: Image registration
    if "registration" in config:
        logger.info("Performing image registration")
        outdir = output / config["registration"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        skip_reg = not config["registration"].get("force", False)
        offset_files = []
        ## 3a: measure offsets
        for i in tqdm.trange(N_files, desc="Measuring frame offsets"):
            filename = working_files[i]
            header = fits.getheader(filename)
            cam_idx = int(header["U_CAMERA"] - 1)
            outname = outdir / f"{filename.stem}_offsets.csv"
            window = config["registration"].get("window_size", 30)
            if "coronagraph" in config:
                r = lamd_to_pixel(
                    config["coronagraph"]["satellite_spots"]["radius"],
                    header["U_FILTER"],
                )
                ang = config["coronagraph"]["satellite_spots"].get("angle", -4)
                offset_file = measure_offsets(
                    filename,
                    method=config["registration"].get("method", "com"),
                    upsample_factor=config["registration"].get("upsample_factor", 1),
                    refmethod=config["registration"].get("reference_method", "com"),
                    center=frame_centers[cam_idx],
                    coronagraphic=True,
                    radius=r,
                    theta=ang,
                    window=window,
                    output=outname,
                    skip=skip_reg,
                )
            else:
                offset_file = measure_offsets(
                    filename,
                    method=config["registration"].get("method", "peak"),
                    upsample_factor=config["registration"].get("upsample_factor", 1),
                    refmethod=config["registration"].get("reference_method", "com"),
                    center=frame_centers[cam_idx],
                    window=window,
                    output=outname,
                    skip=skip_reg,
                )
            offset_files.append(offset_file)
        ## 3b: registration
        for i in tqdm.trange(N_files, desc="Aligning frames"):
            filename = working_files[i]
            offset_file = offset_files[i]
            outname = outdir / f"{filename.stem}_aligned{filename.suffix}"
            working_files[i] = register_file(
                filename,
                offset_file,
                output=outname,
                skip=skip_reg,
            )
        logger.info("Finished registering frames")

    ## Step 4: coadding
    if "coadd" in config:
        logger.info("Coadding registered frames")
        outdir = output / config["coadd"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        skip_coadd = not config["coadd"].get("force", False)
        for i in tqdm.trange(N_files, desc="Collapsing frames"):
            filename = working_files[i]
            outname = outdir / f"{filename.stem}_collapsed{filename.suffix}"
            if skip_coadd and outname.is_file():
                working_files[i] = outname
                continue
            cube, header = fits.getdata(filename, header=True)
            frame = np.median(cube, axis=0)
            fits.writeto(outname, frame, overwrite=True)

        logger.info("Finished coadding frames")

    logger.info("Finished running pipeline")


if __name__ == "__main__":
    main()
