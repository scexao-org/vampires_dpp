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
    correct_distortion_cube,
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
        self.root_dir = Path(self.config["directory"])
        self.output_dir = Path(self.config.get("output_directory", self.root_dir))
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
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Root directory is {self.root_dir}")
        self.logger.debug(f"Output directory is {self.output_dir}")

        self.frame_centers = {"cam1": None, "cam2": None}
        if "frame_centers" in self.config:
            centers_config = self.config["frame_centers"]
            if isinstance(centers_config, dict):
                for k in self.frame_centers.keys():
                    self.frame_centers[k] = np.array(centers_config[k])[::-1]
            else:
                _ctr = np.array(centers_config)[::-1]
                for k in self.frame_centers.keys():
                    self.frame_centers[k] = _ctr
        self.logger.debug(f"Cam 1 frame center is {self.frame_centers['cam1']} (y, x)")
        self.logger.debug(f"Cam 2 frame center is {self.frame_centers['cam2']} (y, x)")

        ## configure astrometry
        pxscale = PIXEL_SCALE
        pupil_offset = PUPIL_OFFSET
        coord = None
        if "astrometry" in self.config:
            astrom_config = self.config["astrometry"]
            pxscale = astrom_config.get("pixel_scale", PIXEL_SCALE)  # mas/px
            pupil_offset = astrom_config.get("pupil_offset", PUPIL_OFFSET)  # deg
            # if custom coord
            if "coord" in astrom_config:
                coord_dict = astrom_config["coord"]
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
            # query from GAIA DR3
            coord = get_gaia_astrometry(self.config["target"])

        ## Step 1: Fix headers and calibrate
        self.logger.info("Starting data calibration")
        outdir = self.output_dir / self.config["calibration"].get("output_directory", "")
        if not outdir.is_dir():
            outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving calibrated data to {outdir.absolute()}")

        tripwire = False
        ## Step 1a: create master dark
        self.master_darks = {"cam1": None, "cam2": None}
        if "darks" in self.config["calibration"]:
            dark_config = self.config["calibration"]["darks"]
            skip_darks = not dark_config.get("force", False)
            if skip_darks:
                self.logger.debug("skipping darks if files exist")
            tripwire = tripwire or not skip_darks
            for key in ("cam1", "cam2"):
                if key in dark_config:
                    dark_filenames = self.parse_filenames(self.root_dir, dark_config[key])
                    dark_frames = []
                    for filename in tqdm.tqdm(dark_filenames, desc=f"Making {key} master darks"):
                        outname = outdir / f"{filename.stem}_collapsed{filename.suffix}"
                        dark_frames.append(
                            make_dark_file(filename, output=outname, skip=skip_darks)
                        )
                    self.master_darks[key] = (
                        outdir / f"{self.config['name']}_master_dark_{key}.fits"
                    )
                    master_dark = np.mean([fits.getdata(f) for f in dark_frames], axis=0)
                    fits.writeto(self.master_darks[key], master_dark, overwrite=True)
                    self.logger.debug(f"saved master dark to {self.master_darks[key].absolute()}")
        ## Step 1b: create master flats
        self.master_flats = {"cam1": None, "cam2": None}
        if "flats" in self.config["calibration"]:
            flat_config = self.config["calibration"]["flats"]
            # if darks were remade, need to remake flats
            skip_flats = not tripwire and not flat_config.get("force", False)
            if skip_flats:
                self.logger.debug("skipping flats if files exist")
            tripwire = tripwire or not skip_flats
            for key in ("cam1", "cam2"):
                if key in flat_config:
                    flat_filenames = self.parse_filenames(self.root_dir, flat_config[key])
                    dark_filename = self.master_darks[key]
                    flat_frames = []
                    for filename in tqdm.tqdm(flat_filenames, desc=f"Making {key} master flats"):
                        outname = outdir / f"{filename.stem}_collapsed{filename.suffix}"
                        flat_frames.append(
                            make_flat_file(
                                filename,
                                dark=dark_filename,
                                output=outname,
                                skip=skip_flats,
                            )
                        )
                    self.master_flats[key] = (
                        outdir / f"{self.config['name']}_master_flat_{key}.fits"
                    )
                    master_flat = np.mean([fits.getdata(f) for f in flat_frames], axis=0)
                    fits.writeto(self.master_flats[key], master_flat, overwrite=True)
                    self.logger.debug(f"saved master flat to {self.master_flats[key].absolute()}")

        ## Step 1c: calibrate files and fix headers
        filenames = self.parse_filenames(self.root_dir, self.config["filenames"])
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
            cam_key = "cam1" if header["U_CAMERA"] == 1 else "cam2"
            if self.master_darks[cam_key] is not None:
                dark_frame = fits.getdata(self.master_darks[cam_key])
            else:
                dark_frame = None
            if self.master_flats[cam_key] is not None:
                flat_frame = fits.getdata(self.master_flats[cam_key])
            else:
                flat_frame = None
            calib_cube, _ = calibrate(
                cube,
                discard=2,
                dark=dark_frame,
                flat=flat_frame,
                flip=(cam_key == "cam1"),  # only flip cam1 data
            )
            if "distortion" in self.config["calibration"]:
                self.logger.debug("Correcting frame distortion")
                distort_config = self.config["calibration"]["distortion"]
                distort_file = distort_config["transform"]
                distort_coeffs = pd.read_csv(distort_file, index_col=0)
                params = distort_coeffs.loc[cam_key]
                calib_cube, header = correct_distortion_cube(calib_cube, *params, header=header)

            if self.config["calibration"].get("deinterleave", False):
                sub_cube_flc1 = calib_cube[::2]
                header["U_FLCSTT"] = 1, "FLC state (1 or 2)"
                fits.writeto(outname_flc1, sub_cube_flc1, header, overwrite=True)
                self.logger.debug(f"saved FLC 1 calibrated data to {outname_flc1.absolute()}")
                working_files.append(outname_flc1)

                sub_cube_flc2 = calib_cube[1::2]
                header["U_FLCSTT"] = 2, "FLC state (1 or 2)"
                fits.writeto(outname_flc2, sub_cube_flc2, header, overwrite=True)
                self.logger.debug(f"saved FLC 2 calibrated data to {outname_flc2.absolute()}")
                working_files.append(outname_flc2)
            else:
                fits.writeto(outname, calib_cube, header, overwrite=True)
                self.logger.debug(f"saved calibrated file at {outname.absolute()}")
                working_files.append(outname)

        # save header table
        table = observation_table(working_files).sort_values("DATE")
        working_files = [working_files[i] for i in table.index]
        table_name = self.output_dir / f"{self.config['name']}_headers.csv"
        # if not table_name.is_file():
        table.to_csv(table_name)
        self.logger.debug(f"Saved table of headers to {table_name.absolute()}")

        self.logger.info("Data calibration completed")

        ## Step 2: Frame selection
        if "frame_selection" in self.config:
            self.logger.info("Performing frame selection")
            select_config = self.config["frame_selection"]
            outdir = self.output_dir / select_config.get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Saving frame selection data to {outdir.absolute()}")
            skip_select = not tripwire and not select_config.get("force", False)
            if skip_select:
                self.logger.debug("skipping frame selection if files exist")
            tripwire = tripwire or not skip_select
            metric_files = []
            ## 2a: measure metrics
            for i in tqdm.trange(len(working_files), desc="Measuring frame selection metric"):
                filename = working_files[i]
                self.logger.debug(f"Measuring metric for {filename.absolute()}")
                header = fits.getheader(filename)
                cam_key = "cam1" if header["U_CAMERA"] == 1 else "cam2"
                outname = outdir / f"{filename.stem}_metrics.csv"
                window = select_config.get("window_size", 30)
                if "coronagraph" in self.config:
                    r = lamd_to_pixel(
                        self.config["coronagraph"]["satellite_spots"]["radius"],
                        header["U_FILTER"],
                    )
                    ang = self.config["coronagraph"]["satellite_spots"].get("angle", -4)
                    metric_file = measure_metric_file(
                        filename,
                        center=self.frame_centers[cam_key],
                        coronagraphic=True,
                        radius=r,
                        theta=ang,
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
                metric_files.append(metric_file)

            ## 2b: perform frame selection
            quantile = select_config.get("q", 0)
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
            outdir = self.output_dir / self.config["registration"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving image registration data to {outdir.absolute()}")
            offset_files = []
            skip_reg = not tripwire and not self.config["registration"].get("force", False)
            if skip_reg:
                self.logger.debug("skipping offset files and aligned files if they exist")
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
                cam_key = "cam1" if header["U_CAMERA"] == 1 else "cam2"
                outname = outdir / f"{filename.stem}_offsets.csv"
                if "coronagraph" in self.config:
                    r = lamd_to_pixel(
                        self.config["coronagraph"]["satellite_spots"]["radius"],
                        header["U_FILTER"],
                    )
                    offset_file = measure_offsets(
                        filename,
                        method=self.config["registration"].get("method", "com"),
                        center=self.frame_centers[cam_key],
                        coronagraphic=True,
                        radius=r,
                        theta=self.config["coronagraph"]["satellite_spots"].get("angle", -4),
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
            outdir = self.output_dir / self.config["collapsing"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving collapsed data to {outdir.absolute()}")
            skip_collapse = not tripwire and not self.config["collapsing"].get("force", False)
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
                cam_files = filter(lambda f: fits.getval(f, "U_CAMERA") == cam_num, collapse_files)
                # generate cube
                outname = outdir / f"{self.config['name']}_cam{cam_num}_collapsed_cube.fits"
                collapse_cube_file = combine_frames_files(cam_files, output=outname, skip=False)
                self.logger.debug(f"saved collapsed cube to {collapse_cube_file.absolute()}")
                # derot angles
                angs = [fits.getval(f, "D_IMRPAD") + pupil_offset for f in cam_files]
                derot_angles = np.asarray(angs, "f4")
                outname = self.output_dir / f"{self.config['name']}_cam{cam_num}_derot_angles.fits"
                fits.writeto(outname, derot_angles, overwrite=True)
                self.logger.debug(f"saved derot angles to {outname.absolute()}")
            self.logger.info("Finished collapsing frames")

        ## Step 7: derotate
        if "derotate" in self.config:
            self.logger.info("Derotating frames")
            outdir = self.output_dir / self.config["derotate"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving derotated data to {outdir.absolute()}")
            skip_derot = not tripwire and not self.config["derotate"].get("force", False)
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
                cam_files = filter(lambda f: fits.getval(f, "U_CAMERA") == cam_num, derot_files)
                # generate cube
                outname = outdir / f"{self.config['name']}_cam{cam_num}_derot_cube.fits"
                derot_cube_file = combine_frames_files(cam_files, output=outname, skip=False)
                self.logger.debug(f"saved derotated cube to {derot_cube_file.absolute()}")

            self.logger.info("Finished derotating frames")

        ## Step 8: PDI
        if "polarimetry" in self.config:
            if "collapsing" not in self.config:
                raise ValueError("Cannot do PDI without collapsing data.")
            self.logger.info("Performing polarimetric calibration")
            outdir = self.output_dir / self.config["polarimetry"].get("output_directory", "")
            if not outdir.is_dir():
                outdir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"saving Stokes data to {outdir.absolute()}")
            skip_pdi = not tripwire and not self.config["polarimetry"].get("force", False)
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
                        stokes_hdr,
                        stokes_angles,
                    ) = polarization_calibration_triplediff_naive(table_filt["path"])
                    stokes_cube_file = outname
                    write_stokes_products(
                        stokes_cube,
                        outname=stokes_cube_file,
                        header=stokes_hdr,
                        skip=False,
                    )
                    self.logger.debug(f"saved Stokes cube to {outname.absolute()}")

                    if "ip" in self.config["polarimetry"]:
                        ip_config = self.config["polarimetry"]["ip"]
                        stokes_cube_file = outdir / f"{self.config['name']}_stokes_cube_ip.fits"
                        ip_method = ip_config.get("method", "photometry")
                        aper_rad = ip_config.get("r", 5)
                        for ix in range(stokes_cube.shape[1]):
                            if ip_method == "satspot_photometry" and "coronagraph" in self.config:
                                pQ = measure_instpol_satellite_spots(
                                    stokes_cube[0, ix],
                                    stokes_cube[1, ix],
                                    r=aper_rad,
                                    radius=r,
                                )
                                pU = measure_instpol_satellite_spots(
                                    stokes_cube[0, ix],
                                    stokes_cube[2, ix],
                                    r=aper_rad,
                                    radius=r,
                                )

                            else:
                                pQ = measure_instpol(
                                    stokes_cube[0, ix],
                                    stokes_cube[1, ix],
                                    r=aper_rad,
                                )
                                pU = measure_instpol(
                                    stokes_cube[0, ix],
                                    stokes_cube[2, ix],
                                    r=aper_rad,
                                )
                            stokes_cube[:, ix] = instpol_correct(stokes_cube[:, ix], pQ, pU)
                        write_stokes_products(
                            stokes_cube,
                            outname=stokes_cube_file,
                            header=stokes_hdr,
                            skip=False,
                        )
                        self.logger.debug(f"saved IP-corrected Stokes cube to {outname.absolute()}")

                    stokes_cube_collapsed, stokes_hdr = collapse_stokes_cube(
                        stokes_cube, stokes_angles, header=stokes_hdr
                    )
                    stokes_cube_file = stokes_cube_file.with_name(
                        f"{stokes_cube_file.stem}_collapsed{stokes_cube_file.suffix}"
                    )
                    write_stokes_products(
                        stokes_cube_collapsed,
                        outname=stokes_cube_file,
                        header=stokes_hdr,
                        skip=False,
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
                self.logger.debug(f"saved Mueller matrices to {mueller_mat_file.absolute()}")

                # generate stokes cube
                outname = outdir / f"{self.config['name']}_stokes_cube.fits"
                stokes_cube_file = mueller_matrix_calibration_files(
                    working_files, mueller_mat_file, output=outname, skip=skip_pdi
                )
                stokes_cube, stokes_header = fits.getdata(stokes_cube_file, header=True)
                write_stokes_products(
                    stokes_cube, stokes_header, outname=stokes_cube_file, skip=False
                )
                self.logger.debug(f"saved Stokes IP cube to {stokes_cube_file.absolute()}")
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
                        stokes_cube, stokes_hdr = fits.getdata(stokes_cube_file, header=True)
                        if "pQ" in ip_config:
                            pQ = ip_config["pQ"]
                        else:
                            pQ = measure_instpol(
                                stokes_cube[0],
                                stokes_cube[1],
                                r=ip_config.get("radius", 5),
                            )
                        if "pU" in ip_config:
                            pU = ip_config["pU"]
                        else:
                            pU = measure_instpol(
                                stokes_cube[0],
                                stokes_cube[2],
                                r=ip_config.get("radius", 5),
                            )
                        stokes_ip_cube = instpol_correct(stokes_cube, pQ=pQ, pU=pU)
                        stokes_hdr["pQ"] = (
                            pQ,
                            "VAMPIRES DPP I -> Q IP contribution (corrected)",
                        )
                        stokes_hdr["pU"] = (
                            pU,
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
