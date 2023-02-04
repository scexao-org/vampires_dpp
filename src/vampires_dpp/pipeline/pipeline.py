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
from vampires_dpp.headers import fix_header
from vampires_dpp.image_processing import (
    collapse_cube_file,
    collapse_frames_files,
    combine_frames_files,
    correct_distortion_cube,
    derotate_frame,
)
from vampires_dpp.image_registration import measure_offsets_file, register_file
from vampires_dpp.indexing import lamd_to_pixel
from vampires_dpp.organization import header_table
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
        if self.frame_select is not None:
            path, tripwire = self._frame_select(path, file_info, tripwire)
        ## 3: Image registration
        if self.coregister is not None:
            path, tripwire = self._coregister(path, file_info, tripwire)
        ## Step 4: collapsing
        if self.collapse is not None:
            path, tripwire = self._collapse(path, file_info, tripwire)

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

    def _frame_select(self, path, file_info, tripwire=False):
        logger.info("Performing frame selection")
        config = self.frame_select
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        tripwire |= config.force
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving selected data to {outdir.absolute()}")
        if file_info.camera == 1:
            ctr = self.frame_centers.cam1
        else:
            ctr = self.frame_centers.cam2
        if self.coronagraph is not None:
            metric_file = measure_metric_file(
                path,
                center=ctr,
                coronagraphic=True,
                radius=self.satspots.radius,
                theta=self.satspots.angle,
                metric=config.metric,
                output_directory=outdir,
                force=tripwire,
            )
        else:
            metric_file = measure_metric_file(
                path, center=ctr, metric=config.metric, output_directory=outdir, force=tripwire
            )

        selected_file = frame_select_file(
            path, metric_file, q=config.cutoff, output_directory=outdir, force=tripwire
        )

        logger.info("Data calibration completed")
        return selected_file, tripwire

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
        if file_info.camera == 1:
            ctr = self.frame_centers.cam1
        else:
            ctr = self.frame_centers.cam2
        if self.coronagraph is not None:
            offsets_file = measure_offsets_file(
                path,
                method=config.method,
                output_directory=outdir,
                force=tripwire,
                center=ctr,
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
                center=ctr,
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
        table = header_table(self.working_files)
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
        table = header_table(self.diff_files_ip)
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
