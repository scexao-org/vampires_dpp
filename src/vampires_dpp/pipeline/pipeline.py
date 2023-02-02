import logging
import re
from dataclasses import dataclass
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Dict

import astropy.units as u
import numpy as np
import pandas as pd
import tomli
import tomli_w
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from serde.toml import to_toml
from tqdm.auto import tqdm

import vampires_dpp as vpp
from vampires_dpp.calibration import calibrate_file, make_dark_file, make_flat_file
from vampires_dpp.constants import PIXEL_SCALE, PUPIL_OFFSET, SUBARU_LOC
from vampires_dpp.frame_selection import frame_select_file, measure_metric_file
from vampires_dpp.headers import fix_header, observation_table
from vampires_dpp.image_processing import (
    collapse_cube_file,
    collapse_frames_files,
    combine_frames_files,
    correct_distortion_cube,
    derotate_frame,
)
from vampires_dpp.image_registration import measure_offsets_file, register_file
from vampires_dpp.indexing import lamd_to_pixel
from vampires_dpp.polarization import (
    HWP_POS_STOKES,
    collapse_stokes_cube,
    measure_instpol,
    measure_instpol_satellite_spots,
    mueller_matrix_model,
    pol_inds,
    polarization_calibration_model,
    polarization_calibration_triplediff,
    triplediff_average_angles,
    write_stokes_products,
)
from vampires_dpp.util import check_version
from vampires_dpp.wcs import (
    apply_wcs,
    derotate_wcs,
    get_coord_header,
    get_gaia_astrometry,
)

from .config import PipelineOptions

logger = logging.getLogger("DPP")


class Pipeline(PipelineOptions):
    def __post_init__(self):
        super().__post_init__()
        # make sure versions match within SemVar
        if not check_version(self.version, vpp.__version__):
            raise ValueError(
                f"Input pipeline version ({self.version}) is not compatible with installed version of `vampires_dpp` ({vpp.__version__})."
            )
        if self.output_directory is None:
            self.output_directory = Path.cwd()
        self.output_directory = Path(self.output_directory)
        self.master_darks = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}

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
        with open(filename, "rb") as fh:
            config = tomli.load(fh)
        return cls(**config)

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
        config = tomli.loads(toml_str)
        return cls(**config)

    def to_toml(self, filename: PathLike):
        """
        Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        with open(filename, "wb") as fh:
            tomli_w.dump(self.config, fh)

    def make_master_dark(self):
        # prepare input filenames
        config = self.master_dark
        config.process()
        # make darks for each camera
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        with Pool(self.num_proc) as pool:
            jobs = []
            for file_info, path in zip(config.file_infos, config.paths):
                kwds = dict(
                    output_directory=outdir,
                    force=config.force,
                    method=config.collapse,
                )
                job = pool.apply_async(make_dark_file, args=(path,), kwds=kwds)
                jobs.append((file_info.camera, job))

            cam1_darks = []
            cam2_darks = []
            for cam, job in tqdm(jobs, desc="Collapsing dark frames"):
                filename = job.get()
                if cam == 1:
                    cam1_darks.append(filename)
                else:
                    cam2_darks.append(filename)

        if len(cam1_darks) > 0:
            self.master_darks[1] = outdir / f"master_dark_cam1.fits"
            collapse_frames_files(
                cam1_darks, method=config.collapse, output=self.master_darks[1], force=config.force
            )
        if len(cam2_darks) > 0:
            self.master_darks[2] = outdir / f"master_dark_cam2.fits"
            collapse_frames_files(
                cam2_darks, method=config.collapse, output=self.master_darks[2], force=config.force
            )

    def make_master_flat(self):
        # prepare input filenames
        config = self.master_flat
        config.process()
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        with Pool(self.num_proc) as pool:
            jobs = []
            for file_info, path in zip(config.file_infos, config.paths):
                dark = self.master_darks[file_info.camera]
                kwds = dict(
                    output_directory=outdir,
                    dark_filename=dark,
                    force=config.force,
                    method=config.collapse,
                )
                job = pool.apply_async(make_flat_file, args=(path,), kwds=kwds)
                jobs.append((file_info.camera, job))

            cam1_flats = []
            cam2_flats = []
            for cam, job in tqdm(jobs, desc="Collapsing flat frames"):
                filename = job.get()
                if cam == 1:
                    cam1_flats.append(filename)
                else:
                    cam2_flats.append(filename)

        if len(cam1_flats) > 0:
            self.master_flats[1] = outdir / f"master_flat_cam1.fits"
            collapse_frames_files(
                cam1_flats, method=config.collapse, output=self.master_flats[1], force=config.force
            )
        if len(cam2_flats) > 0:
            self.master_flats[2] = outdir / f"master_flat_cam2.fits"
            collapse_frames_files(
                cam2_flats, method=config.collapse, output=self.master_flats[2], force=config.force
            )

    def process_one(self, path, file_info):
        tripwire = False
        # fix headers and calibrate
        if self.calibrate is not None:
            cur_file, tripwire = self._calibrate(path, file_info, tripwire)
            if not isinstance(cur_file, str):
                file_flc1, file_flc2 = cur_file
                self.process_post_calib(file_flc1, file_info, tripwire)
                self.process_post_calib(file_flc2, file_info, tripwire)
            else:
                self.process_post_calib(cur_file, file_info, tripwire)

    def process_post_calib(self, path, file_info, tripwire=False):
        ## Step 2: Frame selection
        # if "frame_selection" in self.config:
        #     self.frame_select()
        ## 3: Image registration
        if self.coregister is not None:
            path, tripwire = self._coregister(path, file_info, tripwire)
        ## Step 4: collapsing
        if self.collapse is not None:
            path, tripwire = self._collapse(path, file_info, tripwire)
        # ## Step 7: derotate
        # if "derotate" in self.config:
        #     self.derotate()
        ## Step 8: PDI
        # if "polarimetry" in self.config:
        #     self.polarimetry()

    def run(self, num_proc=None):
        """
        Run the pipeline
        """
        self.num_proc = num_proc
        # prepare filenames
        self.process()

        logger.debug(f"Output directory is {self.output_directory}")

        ## Create calibration files
        if self.master_dark is not None:
            self.make_master_dark()

        if self.master_flat is not None:
            self.make_master_flat()

        ## configure astrometry
        self.get_frame_centers()
        self.get_coordinate()

        ## For each file do
        with Pool(self.num_proc) as pool:
            jobs = []
            for path, file_info in zip(self.paths, self.file_infos):
                jobs.append(pool.apply_async(self.process_one, args=(path, file_info)))

            for job in tqdm(jobs, desc="Running pipeline"):
                job.get()

        logger.info("Finished running pipeline")

    def get_frame_centers(self):
        self.centers = {1: None, 2: None}
        if self.frame_centers is not None:
            if self.frame_centers.cam1 is not None:
                self.centers[1] = np.array(self.frame_centers.cam1)[::-1]
            if self.frame_centers.cam2 is not None:
                self.centers[2] = np.array(self.frame_centers.cam2)[::-1]
        logger.debug(f"Cam 1 frame center is {self.centers[1]} (y, x)")
        logger.debug(f"Cam 2 frame center is {self.centers[2]} (y, x)")

    def get_coordinate(self):
        self.pxscale = PIXEL_SCALE
        self.pupil_offset = PUPIL_OFFSET
        self.coord = None
        if self.target is not None and self.target.strip() != "":
            self.coord = get_gaia_astrometry(self.target)
        # if "astrometry" in self.config:
        #     astrom_config = self.config["astrometry"]
        #     self.pxscale = astrom_config.get("pixel_scale", PIXEL_SCALE)  # mas/px
        #     self.pupil_offset = astrom_config.get("pupil_offset", PUPIL_OFFSET)  # deg
        #     # if custom coord
        #     if "coord" in astrom_config:
        #         coord_dict = astrom_config["coord"]
        #         plx = coord_dict.get("plx", None)
        #         if plx is not None:
        #             distance = (plx * u.mas).to(u.parsec, equivalencies=u.parallax())
        #         else:
        #             distance = None
        #         if "pm_ra" in coord_dict:
        #             pm_ra = coord_dict["pm_ra"] * u.mas / u.year
        #         else:
        #             pm_ra = None
        #         if "pm_dec" in coord_dict:
        #             pm_dec = coord_dict["pm_ra"] * u.mas / u.year
        #         else:
        #             pm_dec = None
        #         self.coord = SkyCoord(
        #             ra=coord_dict["ra"] * u.deg,
        #             dec=coord_dict["dec"] * u.deg,
        #             pm_ra_cosdec=pm_ra,
        #             pm_dec=pm_dec,
        #             distance=distance,
        #             frame=coord_dict.get("frame", "ICRS"),
        #             obstime=coord_dict.get("obstime", "J2016"),
        #         )
        #     elif "target" in self.config:
        #         self.coord = get_gaia_astrometry(self.config["target"])
        # elif "target" in self.config:
        #     # query from GAIA DR3
        #     self.coord = get_gaia_astrometry(self.config["target"])

    def _calibrate(self, path, file_info, tripwire=False):
        logger.info("Starting data calibration")
        config = self.calibrate
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving calibrated data to {outdir.absolute()}")
        if config.distortion is not None:
            transform_filename = config.distortion.transform_filename
        else:
            transform_filename = None
        tripwire |= config.force
        calib_file = calibrate_file(
            path,
            dark_filename=self.master_darks[file_info.camera],
            flat_filename=self.master_flats[file_info.camera],
            transform_filename=transform_filename,
            deinterleave=config.deinterleave,
            coord=self.coord,
            output_directory=outdir,
            force=tripwire,
        )
        logger.info("Data calibration completed")
        return calib_file, tripwire

    def _coregister(self, path, file_info, tripwire=False):
        logger.info("Performing image registration")
        config = self.coregister
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving coregistered data to {outdir.absolute()}")
        tripwire |= config.force
        if self.coronagraph is not None:
            offsets_file = measure_offsets_file(
                path,
                method=config.method,
                output_directory=outdir,
                force=tripwire,
                coronagraphic=True,
                radius=self.satspots.radius,
                theta=self.satspots.angle,
            )
        else:
            offsets_file = measure_offsets_file(
                path,
                method=config.method,
                window=config.window,
                output_directory=outdir,
                force=tripwire,
            )

        reg_file = register_file(
            path,
            offsets_file,
            output_directory=outdir,
            force=tripwire,
        )
        logger.info("Data calibration completed")
        return reg_file, tripwire

    def _collapse(self, path, file_info, tripwire=False):
        logger.info("Starting data calibration")
        config = self.collapse
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving collapsed data to {outdir.absolute()}")
        tripwire |= config.force
        calib_file = collapse_cube_file(
            path,
            method=config.method,
            output_directory=outdir,
            force=tripwire,
        )
        logger.info("Data calibration completed")
        return calib_file, tripwire

    def frame_select(self):
        self.logger.info("Performing frame selection")
        select_config = self.config["frame_selection"]
        outdir = self.output_dir / select_config.get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving frame selection data to {outdir.absolute()}")
        skip_select = not self.tripwire and not select_config.get("force", False)
        if skip_select:
            self.logger.debug("skipping frame selection if files exist")
        self.tripwire = self.tripwire or not skip_select
        self.metric_files = []
        ## 2a: measure metrics
        for i in tqdm.trange(len(self.working_files), desc="Measuring frame selection metric"):
            filename = self.working_files[i]
            self.logger.debug(f"Measuring metric for {filename.absolute()}")
            header = fits.getheader(filename)
            cam_key = "cam1" if header["U_CAMERA"] == 1 else "cam2"
            outname = outdir / f"{filename.stem}_metrics.csv"
            window = select_config.get("window_size", 30)
            if "coronagraph" in self.config:
                satspot_radius = lamd_to_pixel(
                    self.config["coronagraph"]["satellite_spots"]["radius"],
                    header["U_FILTER"],
                )
                satspot_angle = self.config["coronagraph"]["satellite_spots"].get("angle", -4)
                metric_file = measure_metric_file(
                    filename,
                    center=self.frame_centers[cam_key],
                    coronagraphic=True,
                    radius=satspot_radius,
                    theta=satspot_angle,
                    window=window,
                    metric=select_config.get("metric", "l2norm"),
                    output=outname,
                    skip=skip_select,
                )
            else:
                metric_file = measure_metric_file(
                    filename,
                    center=self.frame_centers[cam_key],
                    window=window,
                    metric=select_config.get("metric", "l2norm"),
                    output=outname,
                    skip=skip_select,
                )
            self.logger.debug(f"saving metrics to file {metric_file.absolute()}")
            self.metric_files.append(metric_file)

        ## 2b: perform frame selection
        quantile = select_config.get("q", 0)
        if quantile > 0:
            for i in tqdm.trange(len(self.working_files), desc="Discarding frames"):
                filename = self.working_files[i]
                self.logger.debug(f"discarding frames from {filename.absolute()}")
                metric_file = self.metric_files[i]
                outname = outdir / f"{filename.stem}_cut{filename.suffix}"
                self.working_files[i] = frame_select_file(
                    filename,
                    metric_file,
                    q=quantile,
                    output=outname,
                    skip=skip_select,
                )
                self.logger.debug(f"saving data to {outname.absolute()}")

        self.logger.info("Frame selection complete")

    def register(self):
        self.logger.info("Performing image registration")
        outdir = self.output_dir / self.config["registration"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving image registration data to {outdir.absolute()}")
        self.offset_files = []
        skip_reg = not self.tripwire and not self.config["registration"].get("force", False)
        if skip_reg:
            self.logger.debug("skipping offset files and aligned files if they exist")
        self.tripwire = self.tripwire or not skip_reg
        kwargs = {
            "window": self.config["registration"].get("window_size", 30),
            "skip": skip_reg,
        }
        if "dft" in self.config["registration"]:
            kwargs["upsample_factor"] = self.config["registration"]["dft"].get("upsample_factor", 1)
            kwargs["refmethod"] = self.config["registration"]["dft"].get("reference_method", "com")
        ## 3a: measure offsets
        for i in tqdm.trange(len(self.working_files), desc="Measuring frame offsets"):
            filename = self.working_files[i]
            self.logger.debug(f"measuring offsets for {filename.absolute()}")
            header = fits.getheader(filename)
            cam_key = "cam1" if header["U_CAMERA"] == 1 else "cam2"
            outname = outdir / f"{filename.stem}_offsets.csv"
            if "coronagraph" in self.config:
                satspot_radius = lamd_to_pixel(
                    self.config["coronagraph"]["satellite_spots"]["radius"],
                    header["U_FILTER"],
                )
                satspot_angle = self.config["coronagraph"]["satellite_spots"].get("angle", -4)
                offset_file = measure_offsets(
                    filename,
                    method=self.config["registration"].get("method", "com"),
                    center=self.frame_centers[cam_key],
                    coronagraphic=True,
                    radius=satspot_radius,
                    theta=satspot_angle,
                    output=outname,
                    **kwargs,
                )
            else:
                offset_file = measure_offsets(
                    filename,
                    method=self.config["registration"].get("method", "peak"),
                    center=self.frame_centers[cam_key],
                    output=outname,
                    **kwargs,
                )
            self.logger.debug(f"saving offsets to {offset_file.absolute()}")
            self.offset_files.append(offset_file)
        ## 3b: registration
        for i in tqdm.trange(len(self.working_files), desc="Aligning frames"):
            filename = self.working_files[i]
            offset_file = self.offset_files[i]
            self.logger.debug(f"aligning {filename.absolute()}")
            self.logger.debug(f"using offsets {offset_file.absolute()}")
            outname = outdir / f"{filename.stem}_aligned{filename.suffix}"
            self.working_files[i] = register_file(
                filename,
                offset_file,
                output=outname,
                skip=skip_reg,
            )
            self.logger.debug(f"aligned data saved to {outname.absolute()}")
        self.logger.info("Finished registering frames")

    def collapse(self):
        self.logger.info("Collapsing registered frames")
        coll_config = self.config["collapsing"]
        outdir = self.output_dir / coll_config.get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving collapsed data to {outdir.absolute()}")
        skip_collapse = not self.tripwire and not coll_config.get("force", False)
        if skip_collapse:
            self.logger.debug("skipping collapsing cubes if files exist")
        self.tripwire = self.tripwire or not skip_collapse
        for i in tqdm.trange(len(self.working_files), desc="Collapsing frames"):
            filename = self.working_files[i]
            self.logger.debug(f"collapsing cube from {filename.absolute()}")
            outname = outdir / f"{filename.stem}_collapsed{filename.suffix}"
            self.working_files[i] = collapse_cube_file(
                filename,
                method=coll_config.get("method", "median"),
                output=outname,
                skip=skip_collapse,
            )
            self.logger.debug(f"saved collapsed data to {outname.absolute()}")
        # save cam1 and cam2 cubes
        self.collapse_files = self.working_files.copy()
        for cam_num in (1, 2):
            cam_files = filter(lambda f: fits.getval(f, "U_CAMERA") == cam_num, self.collapse_files)
            # generate cube
            outname = outdir / f"{self.config['name']}_cam{cam_num}_collapsed_cube.fits"
            collapsed_file = combine_frames_files(cam_files, output=outname, skip=False)
            self.logger.debug(f"saved collapsed cube to {collapsed_file.absolute()}")
            # derot angles
            angs = [fits.getval(f, "D_IMRPAD") + self.pupil_offset for f in cam_files]
            derot_angles = np.asarray(angs, "f4")
            outname = outdir / f"{self.config['name']}_cam{cam_num}_derot_angles.fits"
            fits.writeto(outname, derot_angles, overwrite=True)
            self.logger.debug(f"saved derot angles to {outname.absolute()}")
        self.logger.info("Finished collapsing frames")

    def derotate(self):
        self.logger.info("Derotating frames")
        outdir = self.output_dir / self.config["derotate"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving derotated data to {outdir.absolute()}")
        skip_derot = not tripwire and not self.config["derotate"].get("force", False)
        if skip_derot:
            self.logger.debug("skipping derotating frames if files exist")
        tripwire = tripwire or not skip_derot
        self.derot_files = self.working_files.copy()
        for i in tqdm.trange(len(self.working_files), desc="Derotating frames"):
            filename = self.working_files[i]
            self.logger.debug(f"derotating frame from {filename.absolute()}")
            outname = outdir / f"{filename.stem}_derot{filename.suffix}"
            self.derot_files[i] = outname
            if skip_derot and outname.is_file():
                continue
            frame, header = fits.getdata(filename, header=True)
            derot_frame = derotate_frame(frame, header["D_IMRPAD"] + self.pupil_offset)
            derot_header = derotate_wcs(header, header["D_IMRPAD"] + self.pupil_offset)
            fits.writeto(outname, derot_frame, header=derot_header, overwrite=True)
            self.logger.debug(f"saved derotated data to {outname.absolute()}")

        # generate derotated cube
        for cam_num in (1, 2):
            cam_files = filter(lambda f: fits.getval(f, "U_CAMERA") == cam_num, self.derot_files)
            # generate cube
            outname = outdir / f"{self.config['name']}_cam{cam_num}_derot_cube.fits"
            derot_cube_file = combine_frames_files(cam_files, output=outname, skip=False)
            self.logger.debug(f"saved derotated cube to {derot_cube_file.absolute()}")

        self.logger.info("Finished derotating frames")

    def polarimetry(self):
        if "collapsing" not in self.config:
            raise ValueError("Cannot do PDI without collapsing data.")
        pol_config = self.config["polarimetry"]
        self.logger.info("Performing polarimetric calibration")
        outdir = self.output_dir / pol_config.get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving Stokes data to {outdir.absolute()}")
        skip_pdi = not self.tripwire and not pol_config.get("force", False)
        if skip_pdi:
            self.logger.debug("skipping PDI if files exist")
        self.tripwire = self.tripwire or not skip_pdi

        # 1. Make diff images
        self.make_diff_images(outdir, skip=skip_pdi)

        # 2. Correct IP + crosstalk
        if "ip" in pol_config:
            self.polarimetry_ip_correct(outdir, skip=skip_pdi)
        else:
            self.diff_files_ip = self.diff_files.copy()
        # 3. Do higher-order correction
        self.polarimetry_triplediff(outdir, skip=skip_pdi)

        self.logger.info("Finished PDI")

    def make_diff_images(self, outdir, skip=False):
        self.logger.info("Making difference frames")
        table = observation_table(self.working_files).sort_values("MJD")
        groups = table.groupby("U_CAMERA")
        cam1_files = groups.get_group(1)["path"]
        cam2_files = groups.get_group(2)["path"]
        self.diff_files = []
        for cam1_file, cam2_file in tqdm.tqdm(
            zip(cam1_files, cam2_files),
            desc="making single diff and sum images",
            total=len(cam1_files),
        ):
            stem = re.sub("_cam[12]", "", cam1_file.stem)
            outname = outdir / f"{stem}_diff.fits"
            self.diff_files.append(outname)
            if skip and outname.is_file():
                continue
            self.logger.debug(f"loading cam1 image from {cam1_file.absolute()}")
            cam1_frame, header = fits.getdata(cam1_file, header=True)

            self.logger.debug(f"loading cam2 image from {cam2_file.absolute()}")
            cam2_frame = fits.getdata(cam2_file)

            diff = cam1_frame - cam2_frame
            summ = cam1_frame + cam2_frame

            stack = np.asarray((summ, diff))

            # prepare header
            del header["U_CAMERA"]
            stokes = HWP_POS_STOKES[header["U_HWPANG"]]
            header["CAXIS3"] = "STOKES"
            header["STOKES"] = f"I,{stokes}"

            fits.writeto(outname, stack, header=header, overwrite=True)
            self.logger.debug(f"saved diff image to {outname.absolute()}")
        self.logger.info("Done making difference frames")

    def polarimetry_ip_correct(self, outdir, skip=False):
        ip_method = self.config["polarimetry"]["ip"].get("method", "photometry")

        self.logger.info(f"Correcting instrumental polarization using '{ip_method}'")

        if ip_method == "photometry":
            func = self.polarimetry_ip_correct_center
        elif ip_method == "satspot_photometry":
            func = self.polarimetry_ip_correct_satspot
        elif ip_method == "mueller":
            func = lambda f, x: f
            # func = self.polarimetry_ip_correct_mueller

        self.diff_files_ip = []
        iter = tqdm.tqdm(self.diff_files, desc="Correcting IP")
        for filename in iter:
            outname = outdir / f"{filename.stem}_ip.fits"
            self.diff_files_ip.append(outname)
            if skip and outname.is_file():
                continue
            func(filename, outname)

        self.logger.info(f"Done correcting instrumental polarization")

    def polarimetry_ip_correct_center(self, filename, outname):
        stack, header = fits.getdata(filename, header=True)
        aper_rad = self.config["polarimetry"]["ip"].get("r", 5)
        pX = measure_instpol(
            stack[0],
            stack[1],
            r=aper_rad,
        )
        stack_corr = stack.copy()
        stack_corr[1] -= pX * stack[0]

        stokes = HWP_POS_STOKES[header["U_HWPANG"]]
        header[f"VPP_P{stokes}"] = pX, f"I -> {stokes} IP correction"
        fits.writeto(outname, stack_corr, header=header, overwrite=True)

    def polarimetry_ip_correct_satspot(self, filename, outname):
        stack, header = fits.getdata(filename, header=True)
        aper_rad = self.config["polarimetry"]["ip"].get("r", 5)
        satspot_config = self.config["coronagraph"]["satellite_spots"]
        satspot_radius = lamd_to_pixel(satspot_config["radius"], header["U_FILTER"])
        satspot_angle = satspot_config.get("angle", -4)
        pX = measure_instpol_satellite_spots(
            stack[0],
            stack[1],
            r=aper_rad,
            radius=satspot_radius,
            angle=satspot_angle,
        )
        stack_corr = stack.copy()
        stack_corr[1] -= pX * stack[0]

        stokes = HWP_POS_STOKES[header["U_HWPANG"]]
        header[f"VPP_P{stokes}"] = pX, f"I -> {stokes} IP correction"
        fits.writeto(outname, stack_corr, header=header, overwrite=True)

    # def polarimetry_ip_correct_mueller(self, filename, outname):
    # self.logger.warning("Mueller matrix calibration is extremely experimental.")
    # stack, header = fits.getdata(filename, header=True)
    # # info needed for Mueller matrices
    # pa = np.deg2rad(header["D_IMRPAD"] + 180 - header["D_IMRPAP"])
    # altitude = np.deg2rad(header["ALTITUDE"])
    # hwp_theta = np.deg2rad(header["U_HWPANG"])
    # imr_theta = np.deg2rad(header["D_IMRANG"])
    # # qwp are oriented with 0 on vertical axis
    # qwp1 = np.deg2rad(header["U_QWP1"]) + np.pi/2
    # qwp2 = np.deg2rad(header["U_QWP2"]) + np.pi/2

    # # get matrix for camera 1
    # M1 = mueller_matrix_model(
    #     camera=1,
    #     filter=header["U_FILTER"],
    #     flc_state=header["U_FLCSTT"],
    #     qwp1=qwp1,
    #     qwp2=qwp2,
    #     imr_theta=imr_theta,
    #     hwp_theta=hwp_theta,
    #     pa=pa,
    #     altitude=altitude,
    # )
    # # get matrix for camera 2
    # M2 = mueller_matrix_model(
    #     camera=2,
    #     filter=header["U_FILTER"],
    #     flc_state=header["U_FLCSTT"],
    #     qwp1=qwp1,
    #     qwp2=qwp2,
    #     imr_theta=imr_theta,
    #     hwp_theta=hwp_theta,
    #     pa=pa,
    #     altitude=altitude,
    # )

    # diff_M = M1 - M2
    # # IP correct
    # stokes = HWP_POS_STOKES[header["U_HWPANG"]]
    # if stokes.endswith("Q"):
    #     pX = diff_M[0, 1]
    # elif stokes.endswith("U"):
    #     pX = diff_M[0, 2]

    # stack[1] -= pX * stack[0]

    # header[f"VPP_P{stokes}"] = pX, f"I -> {stokes} IP correction"
    # fits.writeto(outname, stack, header=header, overwrite=True)

    def polarimetry_triplediff(self, outdir, skip=False):
        # sort table
        table = observation_table(self.diff_files_ip).sort_values("MJD")
        inds = pol_inds(table["U_HWPANG"], 2)
        table_filt = table.loc[inds]
        self.logger.info(
            f"using {len(table_filt)}/{len(table)} files for triple-differential processing"
        )

        outname = outdir / f"{self.config['name']}_stokes_cube.fits"
        outname_coll = outname.with_name(f"{outname.stem}_collapsed.fits")
        if not skip or not outname.is_file() or not outname_coll.is_file():
            polarization_calibration_triplediff(table_filt["path"], outname=outname)
            self.logger.debug(f"saved Stokes cube to {outname.absolute()}")
            stokes_angles = triplediff_average_angles(table_filt["path"])

            stokes_cube, header = fits.getdata(outname, header=True)

            stokes_cube_collapsed, header = collapse_stokes_cube(
                stokes_cube, stokes_angles, header=header
            )
            write_stokes_products(
                stokes_cube_collapsed,
                outname=outname_coll,
                header=header,
                skip=skip,
            )
            self.logger.debug(f"saved collapsed Stokes cube to {outname_coll.absolute()}")

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
            self.logger.debug(f"Root directory: {root.absolute()}")
            self.logger.debug(f"'filenames': {filenames}")
            self.logger.error(
                "No files found; double check your configuration file. See debug information for more details"
            )

        return paths
