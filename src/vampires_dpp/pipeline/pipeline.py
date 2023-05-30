import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from tqdm.auto import tqdm

import vampires_dpp as dpp
from vampires_dpp.calibration import calibrate_file
from vampires_dpp.constants import PIXEL_SCALE, PUPIL_OFFSET
from vampires_dpp.frame_selection import frame_select_file, measure_metric_file
from vampires_dpp.image_processing import FileSet, combine_frames_files
from vampires_dpp.image_registration import (
    lucky_image_file,
    measure_offsets_file,
    register_file,
)
from vampires_dpp.organization import header_table
from vampires_dpp.pipeline.config import PipelineOptions
from vampires_dpp.polarization import (
    polarization_calibration_leastsq,
    collapse_stokes_cube,
    instpol_correct,
    measure_instpol,
    measure_instpol_satellite_spots,
    pol_inds,
    polarization_calibration_triplediff,
    triplediff_average_angles,
    write_stokes_products,
)
from vampires_dpp.util import any_file_newer


class Pipeline(PipelineOptions):
    __doc__ = PipelineOptions.__doc__

    def __post_init__(self):
        super().__post_init__()
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        # self.console = Console()
        self.logger = logging.getLogger("DPP")

    def run(self, filenames, num_proc: int = None, quiet: bool = False):
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

        self.logger.info(f"VAMPIRES DPP: v{dpp.__version__}")
        ## configure astrometry
        self.get_frame_centers()
        self.get_coordinate()

        self.table = header_table(filenames, quiet=True)
        if self.products is not None:
            self.products.output_directory.mkdir(parents=True, exist_ok=True)
            if self.products.header_table:
                table_path = self.products.output_directory / f"{self.name}_headers.csv"
                self.table.to_csv(table_path)
                print(f"Saved header values to: {table_path}")
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
        metric_file = offsets_file = None
        ## Step 2: Frame selection
        if self.frame_select is not None:
            metric_file, tripwire = self.frame_select_one(path, fileinfo, tripwire=tripwire)
        ## 3: Image registration
        if self.register is not None:
            offsets_file, tripwire = self.register_one(path, fileinfo, tripwire=tripwire)
        ## Step 4: collapsing
        if self.collapse is not None:
            path, tripwire = self.collapse_one(
                path,
                fileinfo,
                tripwire=tripwire,
                metric_file=metric_file,
                offsets_file=offsets_file,
            )

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
        if self.centers["cam1"] is None:
            return self.centers["cam1"]
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
            back_filename = config.master_backgrounds.cam1
            flat_filename = config.master_flats.cam1
        elif fileinfo["U_CAMERA"] == 2:
            back_filename = config.master_backgrounds.cam2
            flat_filename = config.master_flats.cam2
        calib_file = calibrate_file(
            path,
            back_filename=back_filename,
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

    def frame_select_one(self, path, fileinfo, save_intermediate=False, tripwire=False):
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

        if save_intermediate:
            frame_select_file(
                path, metric_file, q=config.cutoff, output_directory=outdir, force=tripwire
            )

        self.logger.info("Frame selection completed")
        return metric_file, tripwire

    def register_one(self, path, fileinfo, save_intermediate=False, tripwire=False):
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

        if save_intermediate:
            register_file(
                path,
                offsets_file,
                output_directory=outdir,
                force=tripwire,
            )
        self.logger.info("Image registration completed")
        return offsets_file, tripwire

    def collapse_one(self, path, fileinfo, metric_file=None, offsets_file=None, tripwire=False):
        self.logger.info("Starting data calibration")
        config = self.collapse
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving collapsed data to {outdir.absolute()}")
        tripwire |= config.force

        kwargs = dict(
            method=config.method,
            metric_file=metric_file,
            offsets_file=offsets_file,
            output_directory=outdir,
            force=tripwire,
        )
        if self.frame_select is not None:
            kwargs["q"] = self.frame_select.cutoff

        calib_file = lucky_image_file(path, **kwargs)
        self.logger.info("Data calibration completed")
        return calib_file, tripwire

    def save_adi_cubes(self, force: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
        # preset values
        outname1 = outname2 = None

        # save cubes for each camera
        if "U_FLCSTT" in self.output_table.keys():
            sort_keys = ["MJD", "U_FLCSTT"]
        else:
            sort_keys = "MJD"
        cam1_table = self.output_table.query("U_CAMERA == 1").sort_values(sort_keys)
        if len(cam1_table) > 0:
            outname1 = self.products.output_directory / f"{self.name}_adi_cube_cam1.fits"
            outname1_angles = outname1.with_stem(f"{outname1.stem}_angles")
            combine_frames_files(cam1_table["path"], output=outname1, force=force)
            derot_angles1 = np.asarray(cam1_table["PARANG"])
            fits.writeto(
                outname1_angles,
                derot_angles1.astype("f4"),
                overwrite=True,
            )
            print(f"Saved ADI cube (cam1) to: {outname1}")
            print(f"Saved derotation angles (cam1) to: {outname1_angles}")
        cam2_table = self.output_table.query("U_CAMERA == 2").sort_values(["MJD", "U_FLCSTT"])
        if len(cam2_table) > 0:
            outname2 = self.products.output_directory / f"{self.name}_adi_cube_cam2.fits"
            outname2_angles = outname2.with_stem(f"{outname2.stem}_angles")
            combine_frames_files(cam2_table["path"], output=outname2, force=force)
            derot_angles2 = np.asarray(cam2_table["PARANG"])
            fits.writeto(
                outname2_angles,
                derot_angles2.astype("f4"),
                overwrite=True,
            )
            print(f"Saved ADI cube (cam2) to: {outname2}")
            print(f"Saved derotation angles (cam2) to: {outname2_angles}")

    def _polarimetry(self, tripwire=False):
        if self.products is None:
            return
        if self.collapse is None:
            raise ValueError("Cannot do PDI without collapsing data.")
        config = self.polarimetry
        self.logger.info("Performing polarimetric calibration")
        if config.output_directory is not None:
            outdir = config.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving Stokes data to {outdir.absolute()}")
        tripwire |= config.force

        # create variables
        self.stokes_cube_file = self.stokes_angles_file = self.stokes_collapsed_file = None

        # 1. Make Stokes cube
        if config.method == "difference":
            self.polarimetry_triplediff(
                force=tripwire,
                N_per_hwp=config.N_per_hwp,
                order=config.order,
                adi_sync=config.adi_sync,
            )
        elif config.method == "leastsq":
            self.polarimetry_leastsq(force=tripwire, adi_sync=config.adi_sync)

        # 2. 2nd order IP correction
        if config.ip is not None:
            tripwire |= config.ip.force
            self.polarimetry_ip_correct(outdir, force=tripwire)

        self.logger.info("Finished PDI")

    def polarimetry_ip_correct(self, outdir, force=False):
        match self.polarimetry.ip.method:
            case "photometry":
                opts = dict(r=self.polarimetry.ip.aper_rad)
                func = measure_instpol
            case "satspots":
                opts = dict(
                    r=self.polarimetry.ip.aper_rad,
                    radius=self.satspots.radius,
                    angle=self.satspots.angle,
                )
                func = measure_instpol_satellite_spots

        # for triple diff correct each frame and collapse
        if self.stokes_cube_file is not None:
            ip_file = self.stokes_cube_file.with_name(
                self.stokes_cube_file.name.replace(".fits", "_ipcorr.fits")
            )
            ip_coll_file = ip_file.with_name(ip_file.name.replace("cube", "cube_collapsed"))
            if (
                force
                or not ip_file.is_file()
                or not ip_coll_file.is_file()
                or any_file_newer(self.stokes_cube_file, ip_file)
            ):
                stokes_cube, header = fits.getdata(self.stokes_cube_file, header=True)
                # average cQ and cU for header
                ave_cQ = ave_cU = 0
                for i in range(stokes_cube.shape[1]):
                    cQ = func(stokes_cube[0, i], stokes_cube[1, i], **opts)
                    cU = func(stokes_cube[0, i], stokes_cube[2, i], **opts)
                    ave_cQ += cQ
                    ave_cU += cU
                    stokes_cube[:3, i] = instpol_correct(stokes_cube[:3, i], cQ, cU)
                header["DPP_PQ"] = ave_cQ / stokes_cube.shape[1], "I -> Q IP correction value"
                header["DPP_PU"] = ave_cU / stokes_cube.shape[1], "I -> U IP correction value"
                write_stokes_products(stokes_cube, header=header, outname=ip_file, force=True)
                self.logger.debug(f"saved Stokes cube to: {ip_file.absolute()}")
                print(f"Saved IP-corrected Stokes cube to: {ip_file}")

                stokes_cube_collapsed, header = collapse_stokes_cube(stokes_cube, header=header)
                write_stokes_products(
                    stokes_cube_collapsed,
                    outname=ip_coll_file,
                    header=header,
                    force=True,
                )
                self.logger.debug(f"saved collapsed Stokes cube to: {ip_coll_file.absolute()}")
                print(f"Saved IP-corrected collapsed Stokes cube to: {ip_coll_file}")
        else:  # no cube means least-square reduction
            ip_file = self.stokes_collapsed_file.with_name(
                self.stokes_collapsed_file.name.replace(".fits", "_ipcorr.fits")
            )
            if (
                force
                or not ip_file.is_file()
                or any_file_newer(self.stokes_collapsed_file, ip_file)
            ):
                stokes_frame, header = fits.getdata(self.stokes_collapsed_file, header=True)
                cQ = func(stokes_frame[0], stokes_frame[1], **opts)
                cU = func(stokes_frame[0], stokes_frame[2], **opts)
                stokes_frame = instpol_correct(stokes_frame, cQ, cU)
                header["DPP_PQ"] = cQ, "I -> Q IP correction value"
                header["DPP_PU"] = cU, "I -> U IP correction value"
                write_stokes_products(stokes_frame, header=header, outname=ip_file, force=True)
                self.logger.debug(f"saved ip corrected file to {ip_file.absolute()}")
                print(f"Saved IP-corrected file to {ip_file}")

        self.logger.info(f"Done correcting instrumental polarization")

    def polarimetry_triplediff(self, force=False, N_per_hwp=1, adi_sync=True, **kwargs):
        # sort table
        inds = pol_inds(self.output_table["U_HWPANG"], 4 * N_per_hwp, **kwargs)
        if len(inds) == 0:
            raise ValueError(f"Could not correctly order the HWP angles")
        table_filt = self.output_table.loc[inds]
        self.logger.info(
            f"using {len(table_filt)}/{len(self.output_table)} files for triple-differential processing"
        )

        self.stokes_cube_file = self.products.output_directory / f"{self.name}_stokes_cube.fits"
        self.stokes_angles_file = (
            self.products.output_directory / f"{self.name}_stokes_cube_angles.fits"
        )
        self.stokes_collapsed_file = self.stokes_cube_file.with_name(
            f"{self.stokes_cube_file.stem}_collapsed.fits"
        )
        if (
            force
            or not self.stokes_cube_file.is_file()
            or not self.stokes_angles_file.is_file()
            or not self.stokes_collapsed_file.is_file()
            or any_file_newer(table_filt["path"], self.stokes_cube_file)
        ):
            # create stokes cube
            polarization_calibration_triplediff(
                table_filt["path"],
                outname=self.stokes_cube_file,
                force=True,
                N_per_hwp=N_per_hwp,
                adi_sync=adi_sync,
            )
            self.logger.debug(f"saved Stokes cube to {self.stokes_cube_file.absolute()}")
            print(f"Saved Stokes cube to: {self.stokes_cube_file}")
            # get average angles for each HWP set, save to disk
            stokes_angles = triplediff_average_angles(table_filt["path"])
            fits.writeto(self.stokes_angles_file, stokes_angles, overwrite=True)
            self.logger.debug(f"saved Stokes angles to {self.stokes_angles_file.absolute()}")
            print(f"Saved derotation angles to: {self.stokes_angles_file}")
            # collapse the stokes cube
            stokes_cube, header = fits.getdata(
                self.stokes_cube_file,
                header=True,
            )
            stokes_cube_collapsed, header = collapse_stokes_cube(stokes_cube, header=header)
            write_stokes_products(
                stokes_cube_collapsed,
                outname=self.stokes_collapsed_file,
                header=header,
                force=True,
            )
            self.logger.debug(
                f"saved collapsed Stokes cube to {self.stokes_collapsed_file.absolute()}"
            )
            print(f"Saved collapsed Stokes cube to: {self.stokes_collapsed_file}")

    def polarimetry_leastsq(self, force=False, adi_sync=True, **kwargs):
        self.stokes_collapsed_file = (
            self.products.output_directory / f"{self.name}_stokes_cube_collapsed.fits"
        )
        if (
            force
            or not self.stokes_collapsed_file.is_file()
            or any_file_newer(self.output_table["path"], self.stokes_cube_file)
        ):
            # create stokes cube
            polarization_calibration_leastsq(
                self.output_table["path"],
                outname=self.stokes_collapsed_file,
                force=True,
                adi_sync=adi_sync,
            )
