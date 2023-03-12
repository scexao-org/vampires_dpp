import logging
import multiprocessing as mp
import re
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tomli
from astropy.io import fits
from serde.toml import to_toml
from tqdm.auto import tqdm

import vampires_dpp as vpp
from vampires_dpp.calibration import calibrate_file
from vampires_dpp.constants import PIXEL_SCALE, PUPIL_OFFSET
from vampires_dpp.frame_selection import frame_select_file, measure_metric_file
from vampires_dpp.image_processing import (
    FileSet,
    collapse_cube_file,
    combine_frames_files,
)
from vampires_dpp.image_registration import measure_offsets_file, register_file
from vampires_dpp.organization import header_table
from vampires_dpp.pipeline.config import PipelineOptions
from vampires_dpp.polarization import (
    HWP_POS_STOKES,
    collapse_stokes_cube,
    make_diff_image,
    measure_instpol,
    measure_instpol_satellite_spots,
    pol_inds,
    polarization_calibration_triplediff,
    triplediff_average_angles,
    write_stokes_products,
)
from vampires_dpp.util import any_file_newer, check_version
from vampires_dpp.wcs import get_gaia_astrometry


class Pipeline(PipelineOptions):
    __doc__ = PipelineOptions.__doc__

    def __post_init__(self):
        super().__post_init__()
        # make sure versions match within SemVar
        if not check_version(self.version, vpp.__version__):
            raise ValueError(
                f"Input pipeline version ({self.version}) is not compatible with installed version of `vampires_dpp` ({vpp.__version__})."
            )
        self.master_darks = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        # self.console = Console()
        self.logger = logging.getLogger("DPP")

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

    def to_file(self, filename: PathLike):
        """
        Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        # use serde.to_toml to serialize self
        path = Path(filename)
        path.write_text(to_toml(self))

    def run(self, filenames, num_proc=None):
        """Run the pipeline

        Parameters
        ----------
        filenames : Iterable[PathLike]
            Input filenames to process
        num_proc : Optional[int]
            Number of processes to use for multi-processing, by default None.
        """
        self.num_proc = num_proc

        fh_logger = logging.FileHandler(f"{self.name}_debug.log")
        fh_logger.setLevel(logging.DEBUG)
        self.logger.addHandler(fh_logger)

        self.logger.info(f"VAMPIRES DPP: v{vpp.__version__}")
        ## configure astrometry
        self.get_frame_centers()
        self.get_coordinate()

        self.table = header_table(filenames, quiet=True)
        if self.products is not None:
            self.products.output_directory.mkdir(parents=True, exist_ok=True)
            if self.products.header_table:
                self.table.to_csv(self.products.output_directory / f"{self.name}_headers.csv")

        ## For each file do
        self.logger.info("Starting file-by-file processing")
        self.output_files = []
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for row in self.table.itertuples(index=False):
                jobs.append(pool.apply_async(self.process_one, args=(row._asdict(),)))

            for job in tqdm(jobs, desc="Processing files"):
                result, tripwire = job.get()
                if isinstance(result, Path):
                    self.output_files.append(result)
                else:
                    self.output_files.extend(result)
        self.output_table = header_table(self.output_files, quiet=True)
        ## products
        if self.products is not None:
            # self.console.print("Saving ADI products")
            if self.products.adi_cubes:
                self.save_adi_cubes(force=tripwire)
        ## polarimetry
        if self.polarimetry:
            # self.console.print("Doing PDI")
            self._polarimetry(tripwire=tripwire)

        # self.console.print("Finished running pipeline")

    def process_one(self, fileinfo):
        # fix headers and calibrate
        if self.calibrate is not None:
            cur_file, tripwire = self.calibrate_one(fileinfo["path"], fileinfo)
            if not isinstance(cur_file, Path):
                file_flc1, file_flc2 = cur_file
                path1, tripwire = self.process_post_calib(file_flc1, fileinfo, tripwire)
                path2, tripwire = self.process_post_calib(file_flc2, fileinfo, tripwire)
                return (path1, path2), tripwire
            else:
                path, tripwire = self.process_post_calib(cur_file, fileinfo, tripwire)
                return path, tripwire

    def process_post_calib(self, path, fileinfo, tripwire=False):
        ## Step 2: Frame selection
        if self.frame_select is not None:
            path, tripwire = self.frame_select_one(path, fileinfo, tripwire)
        ## 3: Image registration
        if self.register is not None:
            path, tripwire = self.register_one(path, fileinfo, tripwire)
        ## Step 4: collapsing
        if self.collapse is not None:
            path, tripwire = self.collapse_one(path, fileinfo, tripwire)

        return path, tripwire

    def get_frame_centers(self):
        self.centers = {"cam1": None, "cam2": None}
        if self.frame_centers is not None:
            if self.frame_centers.cam1 is not None:
                self.centers["cam1"] = np.array(self.frame_centers.cam1)[::-1]
            if self.frame_centers.cam2 is not None:
                self.centers["cam2"] = np.array(self.frame_centers.cam2)[::-1]
        self.logger.debug(f"Cam 1 frame center is {self.centers['cam1']} (y, x)")
        self.logger.debug(f"Cam 2 frame center is {self.centers['cam2']} (y, x)")

    def get_center(self, fileinfo):
        if fileinfo["U_CAMERA"] == 2:
            return self.centers["cam2"]
        # for cam 1 data, need to flip coordinate about x-axis
        Ny = fileinfo["NAXIS2"]
        ctr = np.asarray((Ny - 1 - self.centers["cam1"][0], self.centers["cam1"][1]))
        return ctr

    def get_coordinate(self):
        self.pxscale = PIXEL_SCALE
        self.pupil_offset = PUPIL_OFFSET
        self.coord = None
        if self.coordinate is not None:
            self.coord = self.coordinate.get_coord()

    def calibrate_one(self, path, fileinfo, tripwire=False):
        self.logger.info("Starting data calibration")
        config = self.calibrate
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving calibrated data to {outdir.absolute()}")
        if config.distortion is not None:
            transform_filename = config.distortion.transform_filename
        else:
            transform_filename = None
        tripwire |= config.force
        ext = 1 if ".fits.fz" in path.name else 0
        if fileinfo["U_CAMERA"] == 1:
            dark_filename = config.master_darks.cam1
            flat_filename = config.master_flats.cam1
        elif fileinfo["U_CAMERA"] == 2:
            dark_filename = config.master_darks.cam2
            flat_filename = config.master_flats.cam2
        calib_file = calibrate_file(
            path,
            dark_filename=dark_filename,
            flat_filename=flat_filename,
            transform_filename=transform_filename,
            deinterleave=config.deinterleave,
            bpfix=config.fix_bad_pixels,
            coord=self.coord,
            output_directory=outdir,
            force=tripwire,
            hdu=ext,
        )
        self.logger.info("Data calibration completed")
        return calib_file, tripwire

    def frame_select_one(self, path, fileinfo, tripwire=False):
        self.logger.info("Performing frame selection")
        config = self.frame_select
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        tripwire |= config.force
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving selected data to {outdir.absolute()}")
        ctr = self.get_center(fileinfo)
        if self.coronagraph is not None:
            metric_file = measure_metric_file(
                path,
                center=ctr,
                coronagraphic=True,
                window=config.window_size,
                radius=self.satspots.radius,
                theta=self.satspots.angle,
                metric=config.metric,
                output_directory=outdir,
                force=tripwire,
            )
        else:
            metric_file = measure_metric_file(
                path,
                center=ctr,
                window=config.window_size,
                metric=config.metric,
                output_directory=outdir,
                force=tripwire,
            )

        selected_file = frame_select_file(
            path, metric_file, q=config.cutoff, output_directory=outdir, force=tripwire
        )

        self.logger.info("Data calibration completed")
        return selected_file, tripwire

    def register_one(self, path, fileinfo, tripwire=False):
        self.logger.info("Performing image registration")
        config = self.register
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving registered data to {outdir.absolute()}")
        tripwire |= config.force
        ctr = self.get_center(fileinfo)
        if self.coronagraph is not None:
            offsets_file = measure_offsets_file(
                path,
                method=config.method,
                window=config.window_size,
                output_directory=outdir,
                force=tripwire,
                center=ctr,
                coronagraphic=True,
                upample_factor=config.dft_factor,
                radius=self.satspots.radius,
                theta=self.satspots.angle,
            )
        else:
            offsets_file = measure_offsets_file(
                path,
                method=config.method,
                window=config.window_size,
                output_directory=outdir,
                force=tripwire,
                center=ctr,
                upsample_factor=config.dft_factor,
            )

        reg_file = register_file(
            path,
            offsets_file,
            output_directory=outdir,
            force=tripwire,
        )
        self.logger.info("Data calibration completed")
        return reg_file, tripwire

    def collapse_one(self, path, fileinfo, tripwire=False):
        self.logger.info("Starting data calibration")
        config = self.collapse
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving collapsed data to {outdir.absolute()}")
        tripwire |= config.force
        calib_file = collapse_cube_file(
            path,
            method=config.method,
            output_directory=outdir,
            force=tripwire,
        )
        self.logger.info("Data calibration completed")
        return calib_file, tripwire

    def save_adi_cubes(self, force: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
        # preset values
        outname1 = outname2 = None

        # save cubes for each camera
        cam1_table = self.output_table.query("U_CAMERA == 1").sort_values(["MJD", "U_FLCSTT"])
        if len(cam1_table) > 0:
            outname1 = self.products.output_directory / f"{self.name}_adi_cube_cam1.fits"
            combine_frames_files(cam1_table["path"], output=outname1, force=force)
            derot_angles1 = np.asarray(cam1_table["PARANG"])
            fits.writeto(
                outname1.with_stem(f"{outname1.stem}_angles"),
                derot_angles1.astype("f4"),
                overwrite=True,
            )

        cam2_table = self.output_table.query("U_CAMERA == 2").sort_values(["MJD", "U_FLCSTT"])
        if len(cam2_table) > 0:
            outname2 = self.products.output_directory / f"{self.name}_adi_cube_cam2.fits"
            combine_frames_files(cam2_table["path"], output=outname2, force=force)
            derot_angles2 = np.asarray(cam2_table["PARANG"])
            fits.writeto(
                outname2.with_stem(f"{outname2.stem}_angles"),
                derot_angles2.astype("f4"),
                overwrite=True,
            )

    def _polarimetry(self, tripwire=False):
        if self.collapse is None:
            raise ValueError("Cannot do PDI without collapsing data.")
        config = self.polarimetry
        self.logger.info("Performing polarimetric calibration")
        if config.output_directory is not None:
            outdir = config.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving Stokes data to {outdir.absolute()}")
        tripwire |= config.force

        # 1. Make diff images
        self.make_diff_images(outdir, force=tripwire)

        # 2. Correct IP + crosstalk
        if config.ip is not None:
            tripwire |= config.ip.force
            self.polarimetry_ip_correct(outdir, force=tripwire)
        else:
            self.diff_files_ip = self.diff_files.copy()
        # 3. Do higher-order correction
        if self.products is not None:
            self.polarimetry_triplediff(
                force=tripwire,
                N_per_hwp=config.N_per_hwp,
                order=config.order,
                derotate_pa=config.derotate_pa,
            )

        self.logger.info("Finished PDI")

    def make_diff_images(self, outdir, force: bool = False):
        self.logger.info("Making difference frames")
        # table should still be sorted by MJD
        groups = self.output_table.groupby("MJD")
        # filter groups without full camera/FLC states
        filesets = []
        cam1_paths = []
        cam2_paths = []
        for mjd, group in groups:
            fileset = FileSet(group["path"])
            if len(group) == 4:
                filesets.append(fileset)
                for flc in (1, 2):
                    cam1_paths.append(fileset.paths[(1, flc)])
                    cam2_paths.append(fileset.paths[(2, flc)])
                continue
            miss = set([(1, 1), (1, 2), (2, 1), (2, 2)]) - set(fileset.keys)
            self.logger.warn(f"Discarding group for missing {miss} camera, FLC state pairs")

        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for cam1_file, cam2_file in zip(cam1_paths, cam2_paths):
                stem = re.sub("_cam[12]", "", cam1_file.name)
                outname = outdir / stem.replace(".fits", "_diff.fits")
                self.logger.debug(f"loading cam1 image from {cam1_file.absolute()}")
                self.logger.debug(f"loading cam2 image from {cam2_file.absolute()}")
                kwds = dict(outname=outname, force=force)
                jobs.append(
                    pool.apply_async(make_diff_image, args=(cam1_file, cam2_file), kwds=kwds)
                )

            self.diff_files = [job.get() for job in tqdm(jobs, desc="Making diff images")]
        self.logger.info("Done making difference frames")
        return self.diff_files

    def polarimetry_ip_correct(self, outdir, force=False):
        match self.polarimetry.ip.method:
            case "photometry":
                func = self.polarimetry_ip_correct_center
            case "satspots":
                func = self.polarimetry_ip_correct_satspot
            case "mueller":
                func = self.polarimetry_ip_correct_mueller

        self.diff_files_ip = []
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for filename in self.diff_files:
                outname = outdir / filename.name.replace(".fits", "_ip.fits")
                self.diff_files_ip.append(outname)
                if not force and outname.is_file():
                    continue
                jobs.append(pool.apply_async(func, args=(filename, outname)))
            if len(jobs) == 0:
                return
            for job in tqdm(jobs, desc="Correcting IP"):
                job.get()

        self.logger.info(f"Done correcting instrumental polarization")

    def polarimetry_ip_correct_center(self, filename, outname):
        stack, header = fits.getdata(
            filename,
            header=True,
        )
        aper_rad = self.polarimetry.ip.aper_rad
        pX = measure_instpol(
            stack[0],
            stack[1],
            r=aper_rad,
        )
        stack_corr = stack.copy()
        stack_corr[1] -= pX * stack[0]

        stokes = HWP_POS_STOKES[header["U_HWPANG"]]
        header[f"DPP_P{stokes}"] = pX, f"I -> {stokes} IP correction"
        fits.writeto(
            outname,
            stack_corr,
            header=header,
            overwrite=True,
        )

    def polarimetry_ip_correct_satspot(self, filename, outname):
        stack, header = fits.getdata(
            filename,
            header=True,
        )
        aper_rad = self.polarimetry.ip.aper_rad
        pX = measure_instpol_satellite_spots(
            stack[0],
            stack[1],
            r=aper_rad,
            radius=self.satspots.radius,
            angle=self.satspots.angle,
        )
        stack_corr = stack.copy()
        stack_corr[1] -= pX * stack[0]

        stokes = HWP_POS_STOKES[header["U_HWPANG"]]
        header[f"DPP_P{stokes}"] = pX, f"I -> {stokes} IP correction"
        fits.writeto(
            outname,
            stack_corr,
            header=header,
            overwrite=True,
        )

    def polarimetry_ip_correct_mueller(self, filename, outname):
        raise NotImplementedError()

    # logger.warning("Mueller matrix calibration is extremely experimental.")
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

    def polarimetry_triplediff(self, force=False, N_per_hwp=1, derotate_pa=False, **kwargs):
        # sort table
        table = header_table(self.diff_files_ip, quiet=True)
        inds = pol_inds(table["U_HWPANG"], 2 * N_per_hwp, **kwargs)
        if len(inds) == 0:
            raise ValueError(f"Could not correctly order the HWP angles")
        table_filt = table.loc[inds]
        self.logger.info(
            f"using {len(table_filt)}/{len(table)} files for triple-differential processing"
        )

        outname = self.products.output_directory / f"{self.name}_stokes_cube.fits"
        outname_coll = outname.with_name(f"{outname.stem}_collapsed.fits")
        if (
            force
            or not outname.is_file()
            or not outname_coll.is_file()
            or any_file_newer(table_filt["path"], outname)
        ):
            polarization_calibration_triplediff(
                table_filt["path"], outname=outname, force=True, N_per_hwp=N_per_hwp
            )
            self.logger.debug(f"saved Stokes cube to {outname.absolute()}")
            stokes_angles = triplediff_average_angles(table_filt["path"])
            stokes_cube, header = fits.getdata(
                outname,
                header=True,
            )
            stokes_cube_collapsed, header = collapse_stokes_cube(
                stokes_cube, stokes_angles, header=header, derotate_pa=derotate_pa
            )
            write_stokes_products(
                stokes_cube_collapsed,
                outname=outname_coll,
                header=header,
                force=True,
            )
            self.logger.debug(f"saved collapsed Stokes cube to {outname_coll.absolute()}")
