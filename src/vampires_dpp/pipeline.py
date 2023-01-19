from pathlib import Path
import logging
import toml
from packaging import version
import tqdm.auto as tqdm
from astropy.io import fits
import numpy as np
import astropy.units as u
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sys
from os import PathLike
from typing import Dict
import cv2

import vampires_dpp as vpp
from vampires_dpp.calibration import (
    make_dark_file,
    make_flat_file,
    calibrate,
    filter_empty_frames,
)
from vampires_dpp.constants import PUPIL_OFFSET, PIXEL_SCALE, SUBARU_LOC
from vampires_dpp.frame_selection import measure_metric_file, frame_select_file
from vampires_dpp.headers import observation_table, fix_header
from vampires_dpp.image_processing import (
    derotate_frame,
    combine_frames_files,
    collapse_file,
    frame_center,
    distort_frame,
    correct_distortion,
)
from vampires_dpp.image_registration import (
    measure_offsets,
    register_file,
    coregister_file,
)
from vampires_dpp.polarization import (
    mueller_mats_files,
    mueller_matrix_calibration_files,
    measure_instpol,
    measure_instpol_satellite_spots,
    instpol_correct,
    polarization_calibration_triplediff_naive,
    write_stokes_products,
    collapse_stokes_cube,
)
from vampires_dpp.indexing import lamd_to_pixel
from vampires_dpp.util import pol_inds
from vampires_dpp.wcs import (
    apply_wcs,
    derotate_wcs,
    get_gaia_astrometry,
    get_coord_header,
)


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
    def from_file(cls, filename: PathLike):
        """
        Load configuration from TOML file

        Parameters
        ----------
        filename : PathLike
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

    def to_toml(self, filename: PathLike):
        """
        Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        with open(filename, "w") as fh:
            toml.dump(self.config, fh)

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
            pxscale = self.config["astrometry"].get(
                "pixel_scale", PIXEL_SCALE
            )  # mas/px
            pupil_offset = self.config["astrometry"].get(
                "pupil_offset", PUPIL_OFFSET
            )  # deg
            # if custom coord
            if "coord" in self.config["astrometry"]:
                coord_dict = self.config["astrometry"]["coord"]
                plx = coord_dict.get("plx", None)
                if plx is not None:
                    distance = (plx * u.mas).to(u.parsec, equivalencies=u.parallax())
                else:
                    distance = None
                if "pm_ra" in coord_dict:
                    pm_ra = coord_dict["pm_ra"] * u.mas / u.year
                else:
                    pm_ra = None
                if "pm_dec" in coord_dict:
                    pm_dec = coord_dict["pm_ra"] * u.mas / u.year
                else:
                    pm_dec = None
                coord = SkyCoord(
                    ra=coord_dict["ra"] * u.deg,
                    dec=coord_dict["dec"] * u.deg,
                    pm_ra_cosdec=pm_ra,
                    pm_dec=pm_dec,
                    distance=distance,
                    frame=coord_dict.get("frame", "ICRS"),
                    obstime=coord_dict.get("obstime", "J2016"),
                )
            elif "target" in self.config:
                coord = get_gaia_astrometry(self.config["target"])
            else:
                # get from header
                coord = None
        elif "target" in self.config:
            pxscale = PIXEL_SCALE
            pupil_offset = PUPIL_OFFSET
            # query from GAIA DR3
            coord = get_gaia_astrometry(self.config["target"])
        else:
            pxscale = PIXEL_SCALE
            pupil_offset = PUPIL_OFFSET
            coord = None

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

            # if only a single file is given, assume it is for cam 1
            if len(master_darks) == 1:
                self.logger.warning("only using one dark frame")
                master_darks.append(None)
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

            # if only a single file is given, assume it is for cam 1
            if len(master_flats) == 1:
                self.logger.warning("only using one flat frame")
                master_flats.append(None)
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
                outname_flc1 = outname.with_name(f"{outname.stem}_FLC1{outname.suffix}")
                outname_flc2 = outname.with_name(f"{outname.stem}_FLC2{outname.suffix}")
                if skip_calib and outname_flc1.is_file() and outname_flc2.is_file():
                    working_files.extend((outname_flc1, outname_flc2))
                    continue
            else:
                if skip_calib and outname.is_file():
                    working_files.append(outname)
                    continue
            raw_cube, header = fits.getdata(filename, header=True)
            cube = filter_empty_frames(raw_cube)
            if cube.shape[0] < raw_cube.shape[0] / 2 or cube.shape[0] < 3:
                self.logger.warning(
                    f"{filename} will be discarded since it is majority empty frames"
                )
                continue
            header = fix_header(header)
            time = Time(header["MJD"], format="mjd", scale="ut1", location=SUBARU_LOC)
            if coord is None:
                coord_now = get_coord_header(header, time)
            else:
                coord_now = coord.apply_space_motion(time)
            header["RA"] = coord_now.ra.to_string(unit=u.hourangle, sep=":")
            header["DEC"] = coord_now.dec.to_string(unit=u.deg, sep=":")
            header = apply_wcs(header, pxscale=pxscale, pupil_offset=pupil_offset)
            if header["U_CAMERA"] == 1:
                calib_cube, _ = calibrate(
                    cube,
                    discard=2,
                    dark=master_darks[0],
                    flat=master_flats[0],
                    flip=True,
                )
            else:
                calib_cube, _ = calibrate(
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
        # if not table_name.is_file():
        table.to_csv(table_name)
        self.logger.debug(f"Saved table of headers to {table_name.absolute()}")

        # save derotation angle vector
        pa_name = output / f"{self.config['name']}_derot_angles.fits"
        # if not pa_name.is_file():
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
            if "dft" in self.config["registration"]:
                kwargs["upsample_factor"] = self.config["registration"]["dft"].get(
                    "upsample_factor", 1
                )
                kwargs["refmethod"] = self.config["registration"]["dft"].get(
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
            collapse_files = working_files.copy()
            for cam_num in (1, 2):
                cam_files = filter(
                    lambda f: fits.getval(f, "U_CAMERA") == cam_num, collapse_files
                )
                # generate cube
                outname = (
                    outdir / f"{self.config['name']}_cam{cam_num}_collapsed_cube.fits"
                )
                collapse_cube_file = combine_frames_files(
                    cam_files, output=outname, skip=False
                )
                self.logger.debug(
                    f"saved collapsed cube to {collapse_cube_file.absolute()}"
                )
                # derot angles
                angs = [fits.getval(f, "D_IMRPAD") + pupil_offset for f in cam_files]
                derot_angles = np.asarray(angs, "f4")
                outname = (
                    output / f"{self.config['name']}_cam{cam_num}_derot_angles.fits"
                )
                fits.writeto(outname, derot_angles, overwrite=True)
                self.logger.debug(f"saved derot angles to {outname.absolute()}")
            self.logger.info("Finished collapsing frames")

        ## Step 5: re-scaling
        if "distortion" in self.config:
            if "collapsing" not in self.config:
                raise ValueError(
                    "Cannot do distortion correction without collapsing data."
                )
            self.logger.info("Correcting frame distortion")
            distort_config = self.config["distortion"]
            distort_file = distort_config["coefficients"]
            distort_coeffs = pd.read_csv(distort_file, index_col=0)

            outdir = output / distort_config.get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(
                f"saving distortion-corrected data to {outdir.absolute()}"
            )
            skip_distort = not tripwire and not distort_config.get("force", False)
            if skip_distort:
                self.logger.debug("skipping distortion correction if files exist")
            tripwire = tripwire or not skip_distort

            for i in tqdm.trange(len(working_files), desc="Correcting distortion"):
                filename = working_files[i]
                outname = outdir / f"{filename.stem}_distcorr{filename.suffix}"
                working_files[i] = outname
                if skip_distort and outname.is_file():
                    continue
                self.logger.debug(f"correcting distortion for {filename.absolute}")
                frame, hdr = fits.getdata(filename, header=True)
                params = distort_coeffs.loc[f"cam{hdr['U_CAMERA']:.0f}"]
                distcorr_frame, distcorr_hdr = correct_distortion(
                    frame, *params, header=hdr
                )

                fits.writeto(
                    outname, distcorr_frame, header=distcorr_hdr, overwrite=True
                )
                self.logger.debug(
                    f"saved distortion-corrected data to {outname.absolute()}"
                )

            distcorr_files = working_files.copy()
            for cam_num in (1, 2):
                cam_files = filter(
                    lambda f: fits.getval(f, "U_CAMERA") == cam_num, distcorr_files
                )
                # generate cube
                outname = (
                    outdir / f"{self.config['name']}_cam{cam_num}_distcorr_cube.fits"
                )
                distcorr_cube_file = combine_frames_files(
                    cam_files, output=outname, skip=False
                )
                self.logger.debug(
                    f"saved distortion-corrected cube to {distcorr_cube_file.absolute()}"
                )

            self.logger.info("Finished correcting frame distortion")

        ## Step 7: derotate
        if "derotate" in self.config:
            self.logger.info("Derotating frames")
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
            derot_files = working_files.copy()
            for i in tqdm.trange(len(working_files), desc="Derotating frames"):
                filename = working_files[i]
                self.logger.debug(f"derotating frame from {filename.absolute()}")
                outname = outdir / f"{filename.stem}_derot{filename.suffix}"
                derot_files[i] = outname
                if skip_derot and outname.is_file():
                    continue
                frame, header = fits.getdata(filename, header=True)
                derot_frame = derotate_frame(frame, header["D_IMRPAD"] + pupil_offset)
                derot_header = derotate_wcs(header, header["D_IMRPAD"] + pupil_offset)
                fits.writeto(outname, derot_frame, header=derot_header, overwrite=True)
                self.logger.debug(f"saved derotated data to {outname.absolute()}")

            # generate derotated cube
            for cam_num in (1, 2):
                cam_files = filter(
                    lambda f: fits.getval(f, "U_CAMERA") == cam_num, derot_files
                )
                # generate cube
                outname = outdir / f"{self.config['name']}_cam{cam_num}_derot_cube.fits"
                derot_cube_file = combine_frames_files(
                    cam_files, output=outname, skip=False
                )
                self.logger.debug(
                    f"saved derotated cube to {derot_cube_file.absolute()}"
                )

            self.logger.info("Finished derotating frames")

        ## Step 8: PDI
        if "polarimetry" in self.config:
            if "collapsing" not in self.config:
                raise ValueError("Cannot do PDI without collapsing data.")
            self.logger.info("Performing polarimetric calibration")
            outdir = output / self.config["polarimetry"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving Stokes data to {outdir.absolute()}")
            skip_pdi = not tripwire and not self.config["polarimetry"].get(
                "force", False
            )
            if skip_pdi:
                self.logger.debug("skipping PDI if files exist")
            tripwire = tripwire or not skip_pdi
            pol_method = self.config["polarimetry"].get("method", "triplediff")
            if pol_method == "triplediff":
                # sort table
                table = observation_table(working_files).sort_values("DATE")
                inds = pol_inds(table["U_HWPANG"], 4)
                table_filt = table.loc[inds]
                self.logger.info(
                    f"using {len(table_filt)}/{len(table)} files for triple-differential processing"
                )

                outname = outdir / f"{self.config['name']}_stokes_cube.fits"
                if not skip_pdi or not outname.is_file():
                    (
                        stokes_cube,
                        stokes_angles,
                    ) = polarization_calibration_triplediff_naive(table_filt["path"])
                    stokes_cube_file = outname
                    write_stokes_products(
                        stokes_cube, outname=stokes_cube_file, skip=False
                    )
                    self.logger.debug(f"saved Stokes cube to {outname.absolute()}")

                    if "ip" in self.config["polarimetry"]:
                        ip_config = self.config["polarimetry"]["ip"]
                        stokes_cube_file = (
                            outdir / f"{self.config['name']}_stokes_cube_ip.fits"
                        )

                        for ix in range(stokes_cube.shape[1]):
                            if "coronagraph" in self.config:
                                cQ = measure_instpol_satellite_spots(
                                    stokes_cube[0, ix],
                                    stokes_cube[1, ix],
                                    r=ip_config.get("r", 5),
                                    radius=r,
                                )
                                cU = measure_instpol_satellite_spots(
                                    stokes_cube[0, ix],
                                    stokes_cube[2, ix],
                                    r=ip_config.get("r", 5),
                                    radius=r,
                                )

                            else:
                                cQ = measure_instpol(
                                    stokes_cube[0, ix],
                                    stokes_cube[1, ix],
                                    r=ip_config.get("r", 5),
                                )
                                cU = measure_instpol(
                                    stokes_cube[0, ix],
                                    stokes_cube[2, ix],
                                    r=ip_config.get("r", 5),
                                )
                            stokes_cube[:, ix] = instpol_correct(
                                stokes_cube[:, ix], cQ, cU
                            )
                        write_stokes_products(
                            stokes_cube, outname=stokes_cube_file, skip=False
                        )
                        self.logger.debug(
                            f"saved IP-corrected Stokes cube to {outname.absolute()}"
                        )

                    stokes_cube_collapsed = collapse_stokes_cube(
                        stokes_cube, stokes_angles
                    )
                    stokes_cube_file = stokes_cube_file.with_name(
                        f"{stokes_cube_file.stem}_collapsed{stokes_cube_file.suffix}"
                    )
                    write_stokes_products(
                        stokes_cube_collapsed, outname=stokes_cube_file, skip=False
                    )
                    self.logger.debug(
                        f"saved collapsed Stokes cube to {stokes_cube_file.absolute()}"
                    )
            elif pol_method == "mueller":
                # sort table
                table = observation_table(working_files).sort_values("DATE")
                inds = pol_inds(table["U_HWPANG"], 4)
                table_filt = table.loc[inds]
                self.logger.info(
                    f"using {len(table_filt)}/{len(table)} files for triple-differential processing"
                )

                outname = outdir / f"{self.config['name']}_mueller_mats.fits"
                mueller_mat_file = mueller_mats_files(
                    working_files,
                    method="triplediff",
                    output=outname,
                    skip=skip_pdi,
                )
                self.logger.debug(
                    f"saved Mueller matrices to {mueller_mat_file.absolute()}"
                )

                # generate stokes cube
                outname = outdir / f"{self.config['name']}_stokes_cube.fits"
                stokes_cube_file = mueller_matrix_calibration_files(
                    working_files, mueller_mat_file, output=outname, skip=skip_pdi
                )
                stokes_cube, stokes_header = fits.getdata(stokes_cube_file, header=True)
                write_stokes_products(
                    stokes_cube, stokes_header, outname=stokes_cube_file, skip=False
                )
                self.logger.debug(
                    f"saved Stokes IP cube to {stokes_cube_file.absolute()}"
                )
                if "ip" in self.config["polarimetry"]:
                    ip_config = self.config["polarimetry"]["ip"]
                    # generate IP cube
                    skip_ip = not tripwire and not ip_config.get("force", False)
                    if skip_ip:
                        self.logger.debug("skipping IP correction if files exist")
                    tripwire = tripwire or not skip_ip
                    outname = stokes_cube_file.with_name(
                        f"{stokes_cube_file.stem}_collapsed{stokes_cube_file.suffix}"
                    )
                    if not skip_ip or not outname.is_file():
                        stokes_cube, stokes_hdr = fits.getdata(
                            stokes_cube_file, header=True
                        )
                        if "cQ" in ip_config:
                            cQ = ip_config["cQ"]
                        else:
                            cQ = measure_instpol(
                                stokes_cube[0],
                                stokes_cube[1],
                                r=ip_config.get("radius", 5),
                            )
                        if "cU" in ip_config:
                            cU = ip_config["cU"]
                        else:
                            cU = measure_instpol(
                                stokes_cube[0],
                                stokes_cube[2],
                                r=ip_config.get("radius", 5),
                            )
                        stokes_ip_cube = instpol_correct(stokes_cube, cQ=cQ, cU=cU)
                        stokes_hdr["cQ"] = (
                            cQ,
                            "VAMPIRES DPP I -> Q IP contribution (corrected)",
                        )
                        stokes_hdr["cU"] = (
                            cU,
                            "VAMPIRES DPP I -> U IP contribution (corrected)",
                        )
                    stokes_cube_file = outname
                    write_stokes_products(
                        stokes_ip_cube, stokes_hdr, outname=stokes_cube_file, skip=False
                    )
                    self.logger.debug(f"saved Stokes IP cube to {outname.absolute()}")

            self.logger.info("Finished PDI")

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
