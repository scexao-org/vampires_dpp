import multiprocessing as mp
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tomli
import tomli_w
from astropy.io import fits
from loguru import logger
from tqdm.auto import tqdm

import vampires_dpp as dpp
from vampires_dpp.analysis import analyze_file
from vampires_dpp.calibration import calibrate_file
from vampires_dpp.constants import PIXEL_SCALE, PUPIL_OFFSET
from vampires_dpp.image_processing import (
    FileSet,
    collapse_frames_files,
    combine_frames_files,
    make_diff_image,
)
from vampires_dpp.image_registration import lucky_image_file
from vampires_dpp.organization import header_table
from vampires_dpp.pdi.processing import (
    pol_inds,
    polarization_calibration_leastsq,
    polarization_calibration_triplediff,
)
from vampires_dpp.pdi.utils import (
    collapse_stokes_cube,
    instpol_correct,
    measure_instpol,
    triplediff_average_angles,
    write_stokes_products,
)
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.util import any_file_newer

from .modules import get_psf_centroids_mpl

# logger.add(level="INFO")


class Pipeline:
    def __init__(self, config: PipelineConfig, workdir=Path.cwd()):
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        self.diff_files = None
        self.config = config
        self.workdir = Path(workdir)
        self._conn = None
        self._prepare_directories()
        self.db_file = self.workdir / f"{self.config.name}_db.csv"

    def _prepare_directories(self):
        if self.config.calibrate.save_intermediate:
            self.calib_dir = self.workdir / self.config.calibrate.output_directory
            self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir = self.workdir / self.config.analysis.output_directory
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_dir = self.workdir / self.config.preproc_directory
        self.preproc_dir.mkdir(parents=True, exist_ok=True)
        self.product_dir = self.workdir / self.config.product_directory
        self.product_dir.mkdir(parents=True, exist_ok=True)

        if self.config.collapse:
            self.collapse_dir = self.workdir / self.config.collapse.output_directory
            self.collapse_dir.mkdir(parents=True, exist_ok=True)

        if self.config.make_diff_images:
            self.diff_dir = self.workdir / "diff"
            self.diff_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_adi_cubes:
            self.adi_dir = self.workdir / self.config.product_directory / "adi"
            self.adi_dir.mkdir(parents=True, exist_ok=True)

        if self.config.polarimetry:
            self.polarimetry_dir = self.workdir / self.config.polarimetry.output_directory
            self.stokes_dir = self.polarimetry_dir / "stokes"
            self.stokes_dir.mkdir(parents=True, exist_ok=True)
            if self.config.polarimetry.mm_correct:
                self.mm_dir = self.polarimetry_dir / "mm"
                self.mm_dir.mkdir(parents=True, exist_ok=True)

    def create_input_table(self, filenames) -> pd.DataFrame:
        input_table = header_table(filenames, quiet=True)
        table_path = self.preproc_dir / f"{self.config.name}_headers.csv"
        input_table.to_csv(table_path)
        logger.info(f"Saved header table to: {table_path}")
        return input_table

    def merge_input_table_with_db(self, input_table: pd.DataFrame):
        if self.db_file.exists():
            self.working_db = pd.read_csv(self.db_file, index_col=0)
        else:
            input_table.to_csv(self.db_file)
            self.working_db = input_table

        mask = self.working_db["path"] == input_table["path"].values
        self.working_db = self.working_db.loc[mask]
        return self.working_db

    def create_raw_input_psf(self, max_files=50) -> list[Path]:
        # group by cameras
        outfiles = {}
        for cam_num, group in self.working_db.groupby("U_CAMERA"):
            paths = group["path"].sample(n=max_files)
            outpath = (
                self.workdir
                / self.config.preproc_directory
                / f"{self.config.name}_raw_psf_cam{cam_num:.0f}.fits"
            )
            outfile = collapse_frames_files(paths, output=outpath)
            outfiles[f"cam{cam_num:.0f}"] = outfile
            logger.info(f"Saved raw PSF frame to {outpath.absolute()}")
        return outfiles

    def get_centroids(self):
        self.centroids = {"cam1": None, "cam2": None}
        for key in self.centroids.keys():
            path = self.preproc_dir / f"{self.config.name}_centroids_{key}.toml"
            if path.exists():
                with path.open("rb") as fh:
                    centroids = tomli.load(fh)
                self.centroids[key] = {}
                for field, ctrs in centroids.items():
                    self.centroids[key][field] = np.fliplr(np.atleast_2d(ctrs))
            else:
                raw_psf_filenames = self.create_raw_input_psf()
                im = fits.getdata(raw_psf_filenames[key])
                npsfs = 4 if self.config.coronagraphic else 1
                ctrs = get_psf_centroids_mpl(im, npsfs=npsfs)
                field_keys = (f"field_{i}" for i in range(1))
                ctrs_as_dict = dict(zip(field_keys, ctrs.tolist()))
                with path.open("wb") as fh:
                    tomli_w.dump(ctrs_as_dict, fh)
                self.centroids[key] = dict(zip(field_keys, np.fliplr(ctrs)))

        logger.debug(f"Cam 1 frame center is {self.centroids['cam1']} (y, x)")
        logger.debug(f"Cam 2 frame center is {self.centroids['cam2']} (y, x)")
        return self.centroids

    def add_paths_to_db(self):
        input_paths = self.working_db["path"].apply(Path)
        # figure out which metrics need to be calculated, which is necessary to collapse files
        metric_files = [
            (self.workdir / self.config.analysis.get_output_path(f)).absolute() for f in input_paths
        ]
        self.working_db["metric_file"] = metric_files

        # figure out which files need to be re-collapsed, if any
        if self.config.collapse is not None:
            collapse_files = [
                (self.workdir / self.config.collapse.get_output_path(f)).absolute()
                for f in input_paths
            ]
            self.working_db["collapse_file"] = collapse_files

        if self.config.calibrate.save_intermediate:
            calib_files = [
                (self.workdir / self.config.calibrate.get_output_path(f)).absolute()
                for f in input_paths
            ]
            self.working_db["calib_file"] = calib_files

        self.working_db.to_csv(self.db_file)

    def determine_execution(self, force=False):
        if force:
            return self.working_db

        files_to_calibrate = self.working_db["metric_file"].apply(lambda p: not Path(p).exists())
        if self.config.collapse is not None:
            files_to_calibrate |= self.working_db["collapse_file"].apply(
                lambda p: not Path(p).exists()
            )

        subset = self.working_db.loc[files_to_calibrate]
        return subset

    def run(self, filenames, num_proc: Optional[int] = None, force=False):
        """Run the pipeline

        Parameters
        ----------
        filenames : Iterable[PathLike]
            Input filenames to process
        num_proc : Optional[int]
            Number of processes to use for multi-processing, by default None.
        """
        logger.debug(f"VAMPIRES DPP: v{dpp.__version__}")
        conf_copy_path = self.preproc_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        self.num_proc = num_proc
        input_table = self.create_input_table(filenames=filenames)
        self.merge_input_table_with_db(input_table)
        self.add_paths_to_db()
        self.get_centroids()
        self.get_coordinate()

        self.subset_db = self.determine_execution(force=force)
        logger.info(
            f"Processing {len(self.subset_db)} files using {len(self.working_db) - len(self.subset_db)} cached files"
        )

        ## For each file do
        logger.info("Starting file-by-file processing")
        self.output_files = []
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for row in self.subset_db.itertuples(index=False):
                jobs.append(pool.apply_async(self.process_one, args=(row._asdict(),)))

            for job in tqdm(jobs, desc="Processing files"):
                result = job.get()
                self.output_files.append(result)
        self.output_table = header_table(self.output_files, quiet=True)
        ## products
        if self.config.save_adi_cubes:
            self.save_adi_cubes(force=force)
        ## diff images
        if self.config.make_diff_images:
            self.make_diff_images(force=force)

    def process_one(self, fileinfo):
        # fix headers and calibrate
        metric_file = offsets_file = None
        cur_hdu = self.calibrate_one(fileinfo["path"], fileinfo)
        ## Step 2: Frame analysis
        self.analyze_one(cur_hdu, fileinfo)
        ## Step 4: collapsing
        path = self.collapse_one(
            cur_hdu,
            fileinfo,
        )
        return path

    def get_coordinate(self):
        self.pxscale = PIXEL_SCALE
        self.pupil_offset = PUPIL_OFFSET
        self.coord = None
        if self.config.object is not None:
            self.coord = self.config.object.get_coord()

    def calibrate_one(self, path, fileinfo, force=False):
        logger.info("Starting data calibration")
        config = self.config.calibrate
        if config.save_intermediate:
            outpath = Path(fileinfo["calib_file"])
            if not force and outpath.exists():
                hdul = fits.open(outpath)
                return hdul[0]
        if fileinfo["U_CAMERA"] == 1:
            back_filename = config.master_backgrounds.cam1
            flat_filename = config.master_flats.cam1
        elif fileinfo["U_CAMERA"] == 2:
            back_filename = config.master_backgrounds.cam2
            flat_filename = config.master_flats.cam2
        calib_hdu = calibrate_file(
            path,
            back_filename=back_filename,
            flat_filename=flat_filename,
            transform_filename=config.distortion_file,
            bpfix=config.fix_bad_pixels,
            coord=self.coord,
            force=force,
        )
        if config.save_intermediate:
            calib_hdu.writeto(fileinfo["calib_file"], overwrite=True)
            logger.debug(f"Calibrated data saved to {fileinfo['calib_file']}")
        logger.info("Data calibration completed")
        return calib_hdu

    def analyze_one(self, hdu: fits.PrimaryHDU, fileinfo, force=False):
        logger.info("Starting frame analysis")
        config = self.config.analysis

        key = f"cam{fileinfo['U_CAMERA']:.0f}"
        outpath = analyze_file(
            hdu,
            centroids=self.centroids[key],
            subtract_radprof=config.subtract_radprof,
            aper_rad=config.aper_rad,
            ann_rad=config.ann_rad,
            model=config.model,
            outpath=fileinfo["metric_file"],
            force=force,
            window=config.window_size,
        )
        return outpath

    def collapse_one(self, hdu, fileinfo, force=False):
        logger.info("Starting data calibration")
        config = self.config.collapse
        outpath = Path(fileinfo["collapse_file"])
        kwargs = dict(
            method=config.method,
            frame_select=config.frame_select,
            select_cutoff=config.select_cutoff,
            register=config.centroid,
            metric_file=fileinfo["metric_file"],
            outpath=outpath,
            force=force,
        )
        lucky_image_file(hdu, **kwargs)
        logger.info("Data calibration completed")
        logger.debug(f"Saved collapsed data to {outpath}")
        return outpath

    # def reanalyze_one(self, hdu, fileinfo, force=False):
    #     config = self.config.analysis

    #     key = f"cam{fileinfo['U_CAMERA']:.0f}"
    #     outpath = analyze_file(
    #         hdu,
    #         centroids=self.centroids[key],
    #         aper_rad=config.aper_rad,
    #         ann_rad=config.ann_rad,
    #         model=config.model,
    #         outpath=fileinfo["metric_file"],
    #         force=force,
    #         window=config.window_size,
    #     )

    def save_output_header(self):
        outpath = self.product_dir / f"{self.config.name}_table.csv"
        self.output_table.to_csv(outpath)
        return outpath

    def save_adi_cubes(self, force: bool = False):
        # preset values
        self.cam1_cube_path = self.cam2_cube_path = None

        # save cubes for each camera
        if "U_FLCSTT" in self.output_table.keys():
            sort_keys = ["MJD", "U_FLCSTT"]
        else:
            sort_keys = "MJD"

        for cam_num, group in self.output_table.sort_values(sort_keys).groupby("U_CAMERA"):
            cube_path = self.product_dir / f"{self.config.name}_adi_cube_cam{cam_num:.0f}.fits"
            combine_frames_files(group["path"], output=cube_path, force=force)
            logger.info(f"Saved cam {cam_num} ADI cube to {cube_path}")
            angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
            angles = np.asarray(group["DEROTANG"], dtype="f4")
            fits.writeto(angles_path, angles, overwrite=True)
            logger.info(f"Saved cam {cam_num} ADI angles to {angles_path}")

    # def save_sdi_products(self, force: bool = False):
    #     # preset values
    #     sdi_frames = []
    #     derot_angs = []
    #     headers = []
    #     if self.diff_files is None:
    #         print("Skipping SDI products because no difference images were made.")
    #         return
    #     for filename in self.diff_files:
    #         frames, hdr = fits.getdata(filename, header=True)
    #         diff_frame, summ_frame = frames
    #         sdi_frames.append(diff_frame)
    #         headers.append(hdr)
    #         if hdr["U_PLSTIT"] == 2:
    #             # mask state 2: Cont / Halpha
    #             diff_frame *= -1
    #         sdi_frames.append(diff_frame)
    #         derot_angs.append(hdr["DEROTANG"])

    #     output_header_cube = combine_frames_headers(headers)
    #     sdi_frames = np.array(sdi_frames).astype("f4")

    #     outname = self.product_dir / f"{self.config.name}_sdi_cube.fits"
    #     outname_angles = self.product_dir / f"{self.config.name}_sdi_angles.fits"
    #     outname_frame = self.product_dir / f"{self.config.name}_sdi_frame.fits"

    #     fits.writeto(outname, sdi_frames, overwrite=True, header=output_header_cube)
    #     fits.writeto(
    #         outname_angles,
    #         np.array(derot_angs).astype("f4"),
    #         overwrite=True,
    #     )
    #     print(f"Saved SDI cube to: {outname}")
    #     print(f"Saved SDI derotation angles to: {outname_angles}")

    #     collapsed_frame, frame_header = collapse_frames(sdi_frames, headers=headers)
    #     fits.writeto(outname_frame, collapsed_frame, header=frame_header, overwrite=True)
    #     print(f"Saved collapsed SDI frame to: {outname_frame}")

    def make_diff_images(self, force=False):
        logger.info("Making difference frames")
        if self.diff.output_directory is not None:
            outdir = self.diff.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"saving difference images to {outdir.absolute()}")
        force |= self.diff.force
        # table should still be sorted by MJD
        groups = self.output_table.groupby(["MJD", "U_CAMERA"])
        cam1_paths = []
        cam2_paths = []
        for key, group in groups:
            if key[1] == 1:
                cam1_paths.append(group["path"].iloc[0])
            elif key[1] == 2:
                cam2_paths.append(group["path"].iloc[0])

        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for cam1_file, cam2_file in zip(cam1_paths, cam2_paths):
                stem = re.sub("_cam[12]", "", cam1_file.name)
                outname = outdir / stem.replace(".fits", "_diff.fits")
                logger.debug(f"loading cam1 image from {cam1_file.absolute()}")
                logger.debug(f"loading cam2 image from {cam2_file.absolute()}")
                kwds = dict(outname=outname, force=force)
                jobs.append(
                    pool.apply_async(make_diff_image, args=(cam1_file, cam2_file), kwds=kwds)
                )

            self.diff_files = [job.get() for job in tqdm(jobs, desc="Making diff images")]
        logger.info("Done making difference frames")
        return self.diff_files

    def _polarimetry(self, tripwire=False):
        if self.products is None:
            return
        if self.collapse is None:
            raise ValueError("Cannot do PDI without collapsing data.")
        config = self.polarimetry
        logger.info("Performing polarimetric calibration")
        if config.output_directory is not None:
            outdir = config.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"saving Stokes data to {outdir.absolute()}")
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
                mm_correct=config.mm_correct,
            )
        elif config.method == "leastsq":
            self.polarimetry_leastsq(force=tripwire, adi_sync=config.adi_sync)

        # 2. 2nd order IP correction
        if config.ip is not None:
            tripwire |= config.ip.force
            self.polarimetry_ip_correct(outdir, force=tripwire)

        logger.info("Finished PDI")

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
                logger.debug(f"saved Stokes cube to: {ip_file.absolute()}")
                print(f"Saved IP-corrected Stokes cube to: {ip_file}")

                stokes_cube_collapsed, header = collapse_stokes_cube(stokes_cube, header=header)
                write_stokes_products(
                    stokes_cube_collapsed,
                    outname=ip_coll_file,
                    header=header,
                    force=True,
                )
                logger.debug(f"saved collapsed Stokes cube to: {ip_coll_file.absolute()}")
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
                logger.debug(f"saved ip corrected file to {ip_file.absolute()}")
                print(f"Saved IP-corrected file to {ip_file}")

        logger.info(f"Done correcting instrumental polarization")

    def polarimetry_triplediff(self, force=False, N_per_hwp=1, adi_sync=True, **kwargs):
        # sort table
        inds = pol_inds(self.output_table["U_HWPANG"], 4 * N_per_hwp, **kwargs)
        if len(inds) == 0:
            raise ValueError(f"Could not correctly order the HWP angles")
        table_filt = self.output_table.loc[inds]
        logger.info(
            f"using {len(table_filt)}/{len(self.output_table)} files for triple-differential processing"
        )

        self.stokes_cube_file = self.product_dir / f"{self.config.name}_stokes_cube.fits"
        self.stokes_angles_file = self.product_dir / f"{self.config.name}_stokes_cube_angles.fits"
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
                mm_correct=self.polarimetry.mm_correct,
            )
            logger.info(f"saved Stokes cube to {self.stokes_cube_file.absolute()}")
            # get average angles for each HWP set, save to disk
            stokes_angles = triplediff_average_angles(table_filt["path"])
            fits.writeto(self.stokes_angles_file, stokes_angles, overwrite=True)
            logger.info(f"saved Stokes angles to {self.stokes_angles_file.absolute()}")
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
            logger.info(f"saved collapsed Stokes cube to {self.stokes_collapsed_file.absolute()}")

    def polarimetry_leastsq(self, force=False, adi_sync=True, **kwargs):
        self.stokes_collapsed_file = (
            self.product_dir / f"{self.config.name}_stokes_cube_collapsed.fits"
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
