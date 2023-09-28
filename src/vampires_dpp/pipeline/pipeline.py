import logging
import multiprocessing as mp
import re
from pathlib import Path
from typing import Optional, Tuple
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm.auto import tqdm

import vampires_dpp as dpp
from vampires_dpp.analysis import analyze_file
from vampires_dpp.calibration import calibrate_file
from vampires_dpp.constants import PIXEL_SCALE, PUPIL_OFFSET
from vampires_dpp.frame_selection import frame_select_file, measure_metric_file
from vampires_dpp.image_processing import (
    FileSet,
    collapse_frames_files,
    combine_frames_files,
    make_diff_image,
)
from vampires_dpp.image_registration import (
    lucky_image_file,
    measure_offsets_file,
    register_file,
)
from vampires_dpp.organization import header_table
from vampires_dpp.pipeline.config import PipelineOptions
from vampires_dpp.polarization import (
    collapse_stokes_cube,
    doublediff_average_angles,
    instpol_correct,
    measure_instpol,
    measure_instpol_satellite_spots,
    pol_inds,
    polarization_calibration_doublediff,
    polarization_calibration_leastsq,
    polarization_calibration_triplediff,
    triplediff_average_angles,
    write_stokes_products,
)
from vampires_dpp.util import any_file_newer, get_paths


class Pipeline(PipelineOptions):
    __doc__ = PipelineOptions.__doc__

    def __post_init__(self):
        super().__post_init__()
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        self.logger = logging.getLogger("DPP")
        fh_logger = logging.FileHandler(f"{self.name}_debug.log")
        fh_logger.setLevel(logging.DEBUG)
        self.logger.addHandler(fh_logger)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

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
                self.logger.info(f"Saved header values to: {table_path}")
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

        # for row in tqdm(self.table.itertuples(index=False), total=len(self.table), desc="Processing files"):
        #     result, tripwire = self.process_one(row._asdict())
        #     if isinstance(result, Path):
        #         self.output_files.append(result)
        #     else:
        #         self.output_files.extend(result)
        self.output_table = header_table(self.output_files, quiet=True)
        ## products
        if self.products is not None:
            # self.console.print("Saving ADI products")
            if self.products.adi_cubes:
                self.save_adi_cubes(force=tripwire)
        ## diff images
        if self.diff is not None:
            self.make_diff_images(force=tripwire)
        ## polarimetry
        if self.polarimetry:
            # self.console.print("Doing PDI")
            self._polarimetry(tripwire=tripwire)

        # self.console.print("Finished running pipeline")

    def process_one(self, row):
        path, outpath = get_paths(
            row["path"], output_directory=self.collapse.output_directory, suffix="collapsed"
        )
        tripwire = False
        if outpath.exists():
            return outpath, tripwire
        # tripwire = not outpath.exists()
        # print(f"{path} loaded") DEBUG
        # fix headers and calibrate
        if self.calibrate is not None:
            # print("Calibrating...", end="") DEBUG
            t1 = time.time()
            tripwire |= self.calibrate.force
            data, header, tripwire = self.calibrate_one(path, row)
            t2 = time.time()
            # print(f" done calibrating (took {t2 - t1} s).") DEBUG
        metric_file = offsets_file = None
        ## Step 2: Frame selection
        if self.frame_select is not None:
            tripwire |= self.frame_select.force
            metric_file, tripwire = self.frame_select_one(
                data, header, filename=path, tripwire=tripwire
            )
        ## 3: Image registration
        if self.register is not None:
            # print("Registering...", end="") DEBUG
            t1 = time.time()
            tripwire |= self.register.force
            offsets_file, tripwire = self.register_one(
                data, header, filename=path, tripwire=tripwire
            )
            t2 = time.time()
            # print(f" done registering (took {t2 - t1} s).") DEBUG
        ## Step 4: collapsing
        if self.collapse is not None:
            # print("Collapsing...", end="") DEBUG
            t1 = time.time()
            tripwire |= self.collapse.force
            data, header, tripwire = self.collapse_one(
                data,
                header=header,
                filename=path,
                tripwire=tripwire,
                metric_file=metric_file,
                offsets_file=offsets_file,
            )
            t2 = time.time()
            # print(f" done collapsing (took {t2 - t1} s).") DEBUG

        ## Step 5: PSF analysis
        if self.analysis is not None:
            data, header, tripwire = self.analyze_one(
                data,
                header,
                filename=path,
                tripwire=tripwire,
            )

        ## finally, save to file
        fits.writeto(outpath, data, header=header, overwrite=True)

        return outpath, tripwire

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
        # if config.output_directory is not None:
        #     outdir = config.output_directory
        # else:
        #     outdir = self.output_directory
        # outdir.mkdir(parents=True, exist_ok=True)
        # self.logger.debug(f"Saving calibrated data to {outdir.absolute()}")
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
        calib_cube, header = calibrate_file(
            path,
            back_filename=back_filename,
            flat_filename=flat_filename,
            transform_filename=transform_filename,
            bpfix=config.fix_bad_pixels,
            coord=self.coord,
            # output_directory=outdir,
            force=tripwire,
            hdu=ext,
        )
        self.logger.info("Data calibration completed")
        return calib_cube, header, tripwire

    def frame_select_one(
        self, cube, header=None, filename=None, save_intermediate=False, tripwire=False
    ):
        self.logger.info("Performing frame selection")
        config = self.frame_select
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        tripwire |= config.force
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving selected data to {outdir.absolute()}")
        ctr = self.get_center(header)
        if self.use_satspots:
            metric_file = measure_metric_file(
                cube,
                header=header,
                filename=filename,
                center=ctr,
                coronagraphic=True,
                window=config.window_size,
                radius=header["X_GRDSEP"],
                theta=-4,
                metric=config.metric,
                output_directory=outdir,
                force=tripwire,
            )
        else:
            metric_file = measure_metric_file(
                cube,
                header=header,
                filename=filename,
                center=ctr,
                window=config.window_size,
                metric=config.metric,
                output_directory=outdir,
                force=tripwire,
            )

        if save_intermediate:
            frame_select_file(
                cube,
                header=header,
                filename=filename,
                metric_file=metric_file,
                q=config.cutoff,
                output_directory=outdir,
                force=tripwire,
            )

        self.logger.info("Frame selection completed")
        return metric_file, tripwire

    def register_one(self, cube, header, filename, save_intermediate=False, tripwire=False):
        self.logger.info("Performing image registration")
        config = self.register
        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving registered data to {outdir.absolute()}")
        tripwire |= config.force
        ctr = self.get_center(header)
        if self.use_satspots:
            offsets_file = measure_offsets_file(
                cube,
                header,
                filename=filename,
                method=config.method,
                window=config.window_size,
                output_directory=outdir,
                force=tripwire,
                center=ctr,
                coronagraphic=True,
                upample_factor=config.dft_factor,
                refmethod=config.dft_ref,
                radius=header["X_GRDSEP"],
                theta=-4,
            )
        else:
            offsets_file = measure_offsets_file(
                cube,
                header,
                filename=filename,
                method=config.method,
                window=config.window_size,
                output_directory=outdir,
                force=tripwire,
                center=ctr,
                upsample_factor=config.dft_factor,
                refmethod=config.dft_ref,
            )

        if save_intermediate:
            register_file(
                cube,
                header,
                offsets_file,
                filename=filename,
                output_directory=outdir,
                force=tripwire,
            )
        self.logger.info("Image registration completed")
        return offsets_file, tripwire

    def collapse_one(
        self, cube, header, filename, metric_file=None, offsets_file=None, tripwire=False
    ):
        self.logger.info("Starting data collapsing")
        # print("starting data collapsing") DEBUG
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
            filename=filename,
            output_directory=outdir,
            force=tripwire,
        )
        if self.frame_select is not None:
            kwargs["q"] = self.frame_select.cutoff
        # print("starting lucky imaging") DEBUG
        calib_frame, header = lucky_image_file(cube, header, **kwargs)
        self.logger.info("Data calibration completed")
        return calib_frame, header, tripwire

    def analyze_one(self, cube, header, filename, tripwire=False):
        self.logger.info("Starting PSF analysis")
        config = self.analysis

        if config.output_directory is not None:
            outdir = config.output_directory
        else:
            outdir = self.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Saving analyzed data to {outdir.absolute()}")
        tripwire |= config.force

        if not self.use_satspots:
            outcube, header = analyze_file(
                cube,
                header=header,
                filename=filename,
                aper_rad=config.photometry.aper_rad,
                ann_rad=config.photometry.ann_rad,
                model=config.model,
                output_directory=outdir,
                recenter=config.recenter,
                force=tripwire,
                window=config.window_size,
            )
        else:
            outcube, header = analyze_file(
                cube,
                header=header,
                filename=filename,
                aper_rad=config.photometry.aper_rad,
                ann_rad=config.photometry.ann_rad,
                model=config.model,
                output_directory=outdir,
                recenter=config.recenter,
                force=tripwire,
                coronagraphic=True,
                radius=header["X_GRDSEP"],
                theta=-4,
                window=config.window_size,
            )

        return outcube, header, tripwire

    def save_adi_cubes(self, force: bool = False) -> Tuple[Optional[Path], Optional[Path]]:
        # preset values
        self.cam1_cube_path = self.cam2_cube_path = None

        # save cubes for each camera
        sort_keys = ["MJD"]
        cam1_table = self.output_table.query("U_CAMERA == 1").sort_values(sort_keys)
        if len(cam1_table) > 0:
            self.cam1_cube_path = self.products.output_directory / f"{self.name}_adi_cube_cam1.fits"
            self.cam1_angles_path = self.cam1_cube_path.with_stem(
                f"{self.cam1_cube_path.stem}_angles"
            )
            combine_frames_files(cam1_table["path"], output=self.cam1_cube_path, force=force)
            self.cam1_angles = np.asarray(cam1_table["DEROTANG"])
            fits.writeto(
                self.cam1_angles_path,
                self.cam1_angles.astype("f4"),
                overwrite=True,
            )
            print(f"Saved ADI cube (cam1) to: {self.cam1_cube_path}")
            print(f"Saved derotation angles (cam1) to: {self.cam1_angles_path}")
        cam2_table = self.output_table.query("U_CAMERA == 2").sort_values(sort_keys)
        if len(cam2_table) > 0:
            self.cam2_cube_path = self.products.output_directory / f"{self.name}_adi_cube_cam2.fits"
            self.cam2_angles_path = self.cam2_cube_path.with_stem(
                f"{self.cam2_cube_path.stem}_angles"
            )
            combine_frames_files(cam2_table["path"], output=self.cam2_cube_path, force=force)
            self.cam2_angles = np.asarray(cam2_table["DEROTANG"])
            fits.writeto(
                self.cam2_angles_path,
                self.cam2_angles.astype("f4"),
                overwrite=True,
            )
            print(f"Saved ADI cube (cam2) to: {self.cam2_cube_path}")
            print(f"Saved derotation angles (cam2) to: {self.cam2_angles_path}")

    def make_diff_images(self, force=False):
        self.logger.info("Making difference frames")
        if self.diff.output_directory is not None:
            outdir = self.diff.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"saving difference images to {outdir.absolute()}")
        force |= self.diff.force
        # table should still be sorted by MJD
        groups = self.output_table.groupby("MJD")
        # filter groups without full camera/FLC states
        filesets = []
        cam1_paths = []
        cam2_paths = []
        for _, group in groups:
            fileset = FileSet(group["path"])
            if len(group) == 4:
                filesets.append(fileset)
                for flc in ("A", "B"):
                    cam1_paths.append(fileset.paths[(1, flc)])
                    cam2_paths.append(fileset.paths[(2, flc)])
                continue
            elif len(group) == 2:
                cam1_paths.extend(fileset.cam1_paths)
                cam2_paths.extend(fileset.cam2_paths)
                continue
            miss = set([(1, "A"), (1, "B"), (2, "A"), (2, "B")]) - set(fileset.keys)
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
            # see if FLC value has been set, if not do doublediff
            if self.output_table["U_FLC"].iloc[0] == "NA":
                self.polarimetry_doublediff(
                    force=tripwire,
                    N_per_hwp=config.N_per_hwp,
                    order=config.order,
                )
            else:
                self.polarimetry_triplediff(
                    force=tripwire,
                    N_per_hwp=config.N_per_hwp,
                    order=config.order,
                )
        elif config.method == "leastsq":
            self.polarimetry_leastsq(force=tripwire)

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

    def polarimetry_triplediff(self, force=False, N_per_hwp=1, order="QQUU", **kwargs):
        # sort table
        # filt_table = self.output_table.loc[self.output_table["U_FLC"] == ""]
        self.output_table.sort_values(["MJD", "U_FLC", "U_CAMERA"], inplace=True)
        HWPANGS = {"QQUU": [0, 45, 22.5, 67.5], "QUQU": [0, 22.5, 45, 67.5]}[order.upper()]
        ordered_sets = {ang: [] for ang in HWPANGS}
        hwp_idx = 0
        cycle_idx = 0
        buffer = []
        outdir = self.polarimetry.output_directory
        outdir.mkdir(exist_ok=True, parents=True)
        for _, row in self.output_table.iterrows():
            if row["RET-ANG1"] == HWPANGS[hwp_idx]:
                buffer.append(row["path"])
            elif row["RET-ANG1"] == HWPANGS[(hwp_idx + 1) % 4]:
                output = outdir / f"{self.name}_{cycle_idx:03d}.fits"
                coll_frame = collapse_frames_files(buffer, output)
                buffer = []
                ordered_sets[HWPANGS[hwp_idx]].append(coll_frame)

        inds = pol_inds(self.output_table["RET-ANG1"], 4 * N_per_hwp, **kwargs)
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

    def polarimetry_doublediff(self, force=False, N_per_hwp=1, order="QQUU", **kwargs):
        outdir = self.polarimetry.output_directory
        outdir.mkdir(exist_ok=True, parents=True)
        # sort table
        # filt_table = self.output_table.loc[self.output_table["U_FLC"] == ""]
        self.output_table.sort_values(["MJD", "U_CAMERA"], inplace=True)
        HWPANGS = {"QQUU": [0, 45, 22.5, 67.5], "QUQU": [0, 22.5, 45, 67.5]}[order.upper()]
        stokes_files = []
        for cam_num, group in self.output_table.groupby("U_CAMERA"):
            hwp_idx = 0
            cycle_idx = 0
            buffer = []
            for _, row in tqdm(
                group.iterrows(), desc=f"Coadding cam {cam_num} frames", total=len(group)
            ):
                if row["P_RTAGL1"] == -1:
                    row["P_RTAGL1"] = 0
                if row["P_RTAGL1"] == HWPANGS[hwp_idx]:
                    buffer.append(row["path"])
                elif row["P_RTAGL1"] == HWPANGS[(hwp_idx + 1) % 4]:
                    output = (
                        outdir
                        / f"{self.name}_{cycle_idx:03d}_{hwp_idx:01d}_cam{cam_num:01.0f}.fits"
                    )
                    coll_frame = collapse_frames_files(filenames=buffer, output=output)
                    stokes_files.append(coll_frame)
                    buffer = [row["path"]]
                    hwp_idx += 1
                    if hwp_idx == 4:
                        hwp_idx = 0
                        cycle_idx += 1
        stokes_table = header_table(stokes_files)
        groups = stokes_table.sort_values("path").groupby("U_CAMERA")
        sorted_idxs = []
        for idx1, idx2 in zip(groups.get_group(1).index, groups.get_group(2).index):
            sorted_idxs.append(idx1)
            sorted_idxs.append(idx2)
        stokes_table.loc[sorted_idxs]
        inds = pol_inds(stokes_table["RET-ANG1"].loc[sorted_idxs], n=2 * N_per_hwp, **kwargs)
        if len(inds) == 0:
            raise ValueError(f"Could not correctly order the HWP angles")
        table_filt = stokes_table.iloc[inds]
        self.logger.info(
            f"using {len(table_filt)}/{len(stokes_table)} files for double-differential processing"
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
            polarization_calibration_doublediff(
                table_filt["path"],
                outname=self.stokes_cube_file,
                force=True,
                N_per_hwp=N_per_hwp,
            )
            self.logger.debug(f"saved Stokes cube to {self.stokes_cube_file.absolute()}")
            print(f"Saved Stokes cube to: {self.stokes_cube_file}")
            # get average angles for each HWP set, save to disk
            stokes_angles = doublediff_average_angles(table_filt["path"])
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

    def polarimetry_leastsq(self, force=False, **kwargs):
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
            )
