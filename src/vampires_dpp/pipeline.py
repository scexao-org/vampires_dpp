from pathlib import Path
import logging
import toml
from packaging import version
import tqdm.auto as tqdm
from astropy.io import fits
import numpy as np
import sys
from typing import Dict

import vampires_dpp as vpp
from vampires_dpp.calibration import make_dark_file, make_flat_file, calibrate
from vampires_dpp.fixes import fix_header
from vampires_dpp.frame_selection import measure_metric_file, frame_select_file
from vampires_dpp.headers import observation_table
from vampires_dpp.image_processing import derotate_frame
from vampires_dpp.image_registration import measure_offsets, register_file
from vampires_dpp.satellite_spots import lamd_to_pixel
from vampires_dpp.wcs import apply_wcs, derotate_wcs


def check_version(config: str, vpp: str) -> bool:
    """
    Checks compatibility between versions following semantic versioning.

    Parameters
    ----------
    config : str
        Version string for the configuration
    vpp : str
        Version string for `vampires_dpp`

    Returns
    -------
    bool
    """
    config_maj, config_min, config_pat = version.parse(config).release
    vpp_maj, vpp_min, vpp_pat = version.parse(vpp).release
    if vpp_maj == 0:
        flag = config_maj == vpp_maj and config_min == vpp_min and vpp_pat >= config_pat
    else:
        flag = config_maj == vpp_maj and vpp_min >= config_min
        if vpp_min == config_min:
            flag = flag and vpp_pat >= config_pat
    return flag


class Pipeline:
    def __init__(self, config: Dict):
        """
        Initialize a pipeline object from a configuration dictionary.

        Parameters
        ----------
        config : Dict
            Dictionary with the configuration settings.

        Raises
        ------
        ValueError
            If the configuration `version` is not compatible with the current `vampires_dpp` version.
        """
        self.config = config
        self.logger = logging.getLogger("VPP")
        # make sure versions match within SemVar
        if not check_version(self.config["version"], vpp.__version__):
            raise ValueError(
                f"Input pipeline version ({self.config['version']}) is not compatible with installed version of `vampires_dpp` ({vpp.__version__})."
            )

    @classmethod
    def from_file(cls, filename):
        """
        Load configuration from TOML file

        Parameters
        ----------
        filename :
            Path to TOML file with configuration settings.

        Raises
        ------
        ValueError
            If the configuration `version` is not compatible with the current `vampires_dpp` version.

        Examples
        --------
        >>> Pipeline.from_file("config.toml")
        """
        config = toml.load(filename)
        return cls(config)

    @classmethod
    def from_str(cls, toml_str: str):
        """
        Load configuration from TOML string.

        Parameters
        ----------
        toml_str : str
            String of TOML configuration settings.

        Raises
        ------
        ValueError
            If the configuration `version` is not compatible with the current `vampires_dpp` version.
        """
        config = toml.loads(toml_str)
        return cls(config)

    def run(self):
        """
        Run the pipeline
        """

        # set up paths
        root = Path(self.config["directory"])
        output = Path(self.config.get("output_directory", root))
        if not output.is_dir():
            output.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Root directory is {root}")
        self.logger.debug(f"Output directory is {output}")

        if "frame_centers" in self.config:
            frame_centers = [np.array(c)[::-1] for c in self.config["frame_centers"]]
        else:
            frame_centers = [None, None]
        self.logger.debug(f"Cam 1 frame center is {frame_centers[0]} (y, x)")
        self.logger.debug(f"Cam 2 frame center is {frame_centers[1]} (y, x)")

        ## configure astrometry
        if "astrometry" in self.config:
            pxscale = self.config["astrometry"].get("pixel_scale", 6.24)  # mas/px
            pupil_offset = self.config["astrometry"].get("pupil_offset", 140.4)  # deg
        else:
            # TODO set these values in a config file somewhere?
            pxscale = 6.24
            pupil_offset = 140.4

        ## Step 1: Fix headers and calibrate
        self.logger.info("Starting data calibration")
        outdir = output / self.config["calibration"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving calibrated data to {outdir.absolute()}")

        tripwire = False
        ## Step 1a: create master dark
        if "darks" in self.config["calibration"]:
            dark_filenames = self.parse_filenames(
                root, self.config["calibration"]["darks"]["filenames"]
            )
            skip_darks = not self.config["calibration"]["darks"].get("force", False)
            if skip_darks:
                self.logger.debug("skipping darks if files exist")
            tripwire = tripwire or not skip_darks
            master_darks = []
            for dark in tqdm.tqdm(dark_filenames, desc="Making master darks"):
                outname = outdir / f"{dark.stem}_collapsed{dark.suffix}"
                dark_frame = fits.getdata(
                    make_dark_file(dark, outname, skip=skip_darks)
                )
                master_darks.append(dark_frame)
        else:
            master_darks = [None, None]

        ## Step 1b: create master flats
        if "flats" in self.config["calibration"]:
            dark_filenames = self.parse_filenames(
                root, self.config["calibration"]["flats"]["filenames"]
            )
            # if darks were remade, need to remake flats
            skip_flats = not tripwire and not self.config["calibration"]["flats"].get(
                "force", False
            )
            if skip_flats:
                self.logger.debug("skipping flats if files exist")
            tripwire = tripwire or not skip_flats
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
        filenames = self.parse_filenames(root, self.config["filenames"])
        skip_calib = not tripwire and not self.config["calibration"].get("force", False)
        if skip_calib:
            self.logger.debug("skipping calibration if files exist")
        tripwire = tripwire or not skip_calib

        working_files = []
        for filename in tqdm.tqdm(filenames, desc="Calibrating files"):
            self.logger.debug(f"calibrating {filename.absolute()}")
            outname = outdir / f"{filename.stem}_calib{filename.suffix}"
            if self.config["calibration"].get("deinterleave", False):
                outname_flc1 = outname.with_stem(f"{outname.stem}_FLC1")
                outname_flc2 = outname.with_stem(f"{outname.stem}_FLC2")
                if skip_calib and outname_flc1.is_file() and outname_flc2.is_file():
                    working_files.extend((outname_flc1, outname_flc2))
                    continue
            else:
                if skip_calib and outname.is_file():
                    working_files.append(outname)
                    continue
            cube, header = fits.getdata(filename, header=True)
            header = fix_header(header)
            header = apply_wcs(header, pxscale=pxscale, pupil_offset=pupil_offset)
            if header["U_CAMERA"] == 1:
                calib_cube = calibrate(
                    cube,
                    discard=2,
                    dark=master_darks[0],
                    flat=master_flats[0],
                    flip=True,
                )
            else:
                calib_cube = calibrate(
                    cube,
                    discard=2,
                    dark=master_darks[1],
                    flat=master_flats[1],
                    flip=False,
                )
            if self.config["calibration"].get("deinterleave", False):
                sub_cube_flc1 = calib_cube[::2]
                header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
                fits.writeto(outname_flc1, sub_cube_flc1, header, overwrite=True)
                self.logger.debug(
                    f"saved FLC 1 calibrated data to {outname_flc1.absolute()}"
                )
                working_files.append(outname_flc1)

                sub_cube_flc2 = calib_cube[1::2]
                header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
                fits.writeto(outname_flc2, sub_cube_flc2, header, overwrite=True)
                self.logger.debug(
                    f"saved FLC 2 calibrated data to {outname_flc2.absolute()}"
                )
                working_files.append(outname_flc2)
            else:
                fits.writeto(outname, calib_cube, header, overwrite=True)
                self.logger.debug(f"saved calibrated file at {outname.absolute()}")
                working_files.append(outname)

        # save header table
        table = observation_table(working_files).sort_values("DATE")
        working_files = [working_files[i] for i in table.index]
        table_name = output / f"{self.config['name']}_headers.csv"
        if not table_name.is_file():
            table.to_csv(table_name)
            self.logger.debug(f"Saved table of headers to {table_name.absolute()}")

        # save derotation angle vector
        pa_name = output / f"{self.config['name']}_derot_angles.fits"
        if not pa_name.is_file():
            fits.writeto(
                pa_name,
                np.array(table["D_IMRPAD"] + pupil_offset, "f4"),
                overwrite=True,
            )
            self.logger.debug(f"Saved derotation angle vector to {pa_name.absolute()}")

        self.logger.info("Data calibration completed")

        ## Step 2: Frame selection
        if "frame_selection" in self.config:
            self.logger.info("Performing frame selection")
            outdir = output / self.config["frame_selection"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Saving frame selection data to {outdir.absolute()}")
            skip_select = not tripwire and not self.config["frame_selection"].get(
                "force", False
            )
            if skip_select:
                self.logger.debug("skipping frame selection if files exist")
            tripwire = tripwire or not skip_select
            metric_files = []
            ## 2a: measure metrics
            for i in tqdm.trange(
                len(working_files), desc="Measuring frame selection metric"
            ):
                filename = working_files[i]
                self.logger.debug(f"Measuring metric for {filename.absolute()}")
                header = fits.getheader(filename)
                cam_idx = int(header["U_CAMERA"] - 1)
                outname = outdir / f"{filename.stem}_metrics.csv"
                window = self.config["frame_selection"].get("window_size", 30)
                if "coronagraph" in self.config:
                    r = lamd_to_pixel(
                        self.config["coronagraph"]["satellite_spots"]["radius"],
                        header["U_FILTER"],
                    )
                    ang = self.config["coronagraph"]["satellite_spots"].get("angle", -4)
                    metric_file = measure_metric_file(
                        filename,
                        center=frame_centers[cam_idx],
                        coronagraphic=True,
                        radius=r,
                        theta=ang,
                        window=window,
                        metric=self.config["frame_selection"].get("metric", "l2norm"),
                        output=outname,
                        skip=skip_select,
                    )
                else:
                    metric_file = measure_metric_file(
                        filename,
                        center=frame_centers[cam_idx],
                        window=window,
                        metric=self.config["frame_selection"].get("metric", "l2norm"),
                        output=outname,
                        skip=skip_select,
                    )
                self.logger.debug(f"saving metrics to file {metric_file.absolute()}")
                metric_files.append(metric_file)

            ## 2b: perform frame selection
            quantile = self.config["frame_selection"].get("q", 0)
            if quantile > 0:
                for i in tqdm.trange(len(working_files), desc="Discarding frames"):
                    filename = working_files[i]
                    self.logger.debug(f"discarding frames from {filename.absolute()}")
                    metric_file = metric_files[i]
                    outname = outdir / f"{filename.stem}_cut{filename.suffix}"
                    working_files[i] = frame_select_file(
                        filename,
                        metric_file,
                        q=quantile,
                        output=outname,
                        skip=skip_select,
                    )
                    self.logger.debug(f"saving data to {outname.absolute()}")

            self.logger.info("Frame selection complete")

        ## 3: Image registration
        if "registration" in self.config:
            self.logger.info("Performing image registration")
            outdir = output / self.config["registration"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving image registration data to {outdir.absolute()}")
            offset_files = []
            skip_reg = not tripwire and not self.config["registration"].get(
                "force", False
            )
            if skip_reg:
                self.logger.debug(
                    "skipping offset files and aligned files if they exist"
                )
            tripwire = tripwire or not skip_reg
            kwargs = {
                "window": self.config["registration"].get("window_size", 30),
                "skip": skip_reg,
            }
            if "registration.dft" in self.config:
                kwargs["upsample_factor"] = self.config["registration.dft"].get(
                    "upsample_factor", 1
                )
                kwargs["refmethod"] = self.config["registration.dft"].get(
                    "reference_method", "com"
                )
            ## 3a: measure offsets
            for i in tqdm.trange(len(working_files), desc="Measuring frame offsets"):
                filename = working_files[i]
                self.logger.debug(f"measuring offsets for {filename.absolute()}")
                header = fits.getheader(filename)
                cam_idx = int(header["U_CAMERA"] - 1)
                outname = outdir / f"{filename.stem}_offsets.csv"
                if "coronagraph" in self.config:
                    r = lamd_to_pixel(
                        self.config["coronagraph"]["satellite_spots"]["radius"],
                        header["U_FILTER"],
                    )
                    offset_file = measure_offsets(
                        filename,
                        method=self.config["registration"].get("method", "com"),
                        center=frame_centers[cam_idx],
                        coronagraphic=True,
                        radius=r,
                        theta=self.config["coronagraph"]["satellite_spots"].get(
                            "angle", -4
                        ),
                        output=outname,
                        **kwargs,
                    )
                else:
                    offset_file = measure_offsets(
                        filename,
                        method=self.config["registration"].get("method", "peak"),
                        center=frame_centers[cam_idx],
                        output=outname,
                        **kwargs,
                    )
                self.logger.debug(f"saving offsets to {offset_file.absolute()}")
                offset_files.append(offset_file)
            ## 3b: registration
            for i in tqdm.trange(len(working_files), desc="Aligning frames"):
                filename = working_files[i]
                offset_file = offset_files[i]
                self.logger.debug(f"aligning {filename.absolute()}")
                self.logger.debug(f"using offsets {offset_file.absolute()}")
                outname = outdir / f"{filename.stem}_aligned{filename.suffix}"
                working_files[i] = register_file(
                    filename,
                    offset_file,
                    output=outname,
                    skip=skip_reg,
                )
                self.logger.debug(f"aligned data saved to {outname.absolute()}")
            self.logger.info("Finished registering frames")

        ## Step 4: collapsing
        if "collapsing" in self.config:
            self.logger.info("Collapsing registered frames")
            outdir = output / self.config["collapsing"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving collapsed data to {outdir.absolute()}")
            skip_collapse = not tripwire and not self.config["collapsing"].get(
                "force", False
            )
            if skip_collapse:
                self.logger.debug("skipping collapsing cubes if files exist")
            tripwire = tripwire or not skip_collapse
            for i in tqdm.trange(len(working_files), desc="Collapsing frames"):
                filename = working_files[i]
                self.logger.debug(f"collapsing cube from {filename.absolute()}")
                outname = outdir / f"{filename.stem}_collapsed{filename.suffix}"
                working_files[i] = outname
                if skip_collapse and outname.is_file():
                    continue
                cube, header = fits.getdata(filename, header=True)
                frame = np.median(cube, axis=0, overwrite_input=True)
                fits.writeto(outname, frame, header=header, overwrite=True)
                self.logger.debug(f"saved collapsed data to {outname.absolute()}")

            self.logger.info("Finished collapsing frames")

        if "derotate" in self.config:
            self.logger.info("Derotating collapsed frames")
            outdir = output / self.config["derotate"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving derotated data to {outdir.absolute()}")
            skip_derot = not tripwire and not self.config["derotate"].get(
                "force", False
            )
            if skip_derot:
                self.logger.debug("skipping derotating frames if files exist")
            tripwire = tripwire or not skip_derot
            for i in tqdm.trange(len(working_files), desc="Derotating frames"):
                filename = working_files[i]
                self.logger.debug(f"derotating frame from {filename.absolute()}")
                outname = outdir / f"{filename.stem}_derot{filename.suffix}"
                working_files[i] = outname
                if skip_derot and outname.is_file():
                    continue
                frame, header = fits.getdata(filename, header=True)
                derot_frame = derotate_frame(frame, header["D_IMRPAD"] + pupil_offset)
                header = derotate_wcs(header, header["D_IMRPAD"] + pupil_offset)
                fits.writeto(outname, derot_frame, header=header, overwrite=True)
                self.logger.debug(f"saved derotated data to {outname.absolute()}")

        self.logger.info("Finished running pipeline")

    def parse_filenames(self, root, filenames):
        if isinstance(filenames, str):
            path = Path(filenames)
            if path.is_file():
                # is a file with a list of filenames
                fh = path.open("r")
                paths = [Path(f.rstrip()) for f in fh.readlines()]
                fh.close()
            else:
                # is a globbing expression
                paths = list(root.glob(filenames))

        else:
            # is a list of filenames
            paths = [root / f for f in filenames]

        # cause ruckus if no files are found
        if len(paths) == 0:
            self.logger.critical(
                "No files found; double check your configuration file. See debug information for more details"
            )
            self.logger.debug(f"Root directory: {root.absolute()}")
            self.logger.debug(f"'filenames': {filenames}")
            sys.exit(1)

        return paths
