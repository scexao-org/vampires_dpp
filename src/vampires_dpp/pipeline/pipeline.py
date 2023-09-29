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
from vampires_dpp.image_processing import (
    FileSet,
    collapse_frames,
    collapse_frames_files,
    combine_frames_files,
    make_diff_image,
)
from vampires_dpp.organization import header_table
from vampires_dpp.pdi.mueller_matrices import mueller_matrix_from_file
from vampires_dpp.pdi.processing import (
    get_doublediff_set,
    get_triplediff_set,
    make_stokes_image,
    polarization_calibration_leastsq,
)
from vampires_dpp.pdi.utils import write_stokes_products
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.util import any_file_newer

from ..lucky_imaging import lucky_image_file
from .modules import get_psf_centroids_mpl


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
        self.output_table_path = self.product_dir / f"{self.config.name}_table.csv"

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
        input_table = header_table(filenames, quiet=True).sort_values("MJD")
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

    def create_raw_input_psf(self, table, max_files=50) -> dict[str, Path]:
        # group by cameras
        outfiles = {}
        for cam_num, group in table.groupby("U_CAMERA"):
            paths = group["path"].sample(n=max_files)
            outpath = (
                self.workdir
                / self.config.preproc_directory
                / f"{self.config.name}_raw_psf_cam{cam_num:.0f}.fits"
            )
            outfile = collapse_frames_files(paths, output=outpath, cubes=True)
            outfiles[f"cam{cam_num:.0f}"] = outfile
            logger.info(f"Saved raw PSF frame to {outpath.absolute()}")
        return outfiles

    def save_centroids(self, table):
        self.centroids = {"cam1": None, "cam2": None}
        raw_psf_filenames = self.create_raw_input_psf(table)
        for key in self.centroids.keys():
            path = self.preproc_dir / f"{self.config.name}_centroids_{key}.toml"

            im, hdr = fits.getdata(raw_psf_filenames[key], header=True)
            npsfs = 4 if self.config.coronagraphic else 1
            outpath = self.preproc_dir / f"{self.config.name}_{key}.png"
            if "MBI" in hdr.get("OBS-MOD", ""):
                field_keys = "F610", "F670", "F720", "F760"
                nfields = 4
            else:
                field_keys = ("PSF",)
                nfields = 1
            ctrs = get_psf_centroids_mpl(
                np.squeeze(im), npsfs=npsfs, nfields=nfields, suptitle=key, outpath=outpath
            )
            ctrs_as_dict = dict(zip(field_keys, ctrs.tolist()))
            with path.open("wb") as fh:
                tomli_w.dump(ctrs_as_dict, fh)
            logger.debug(f"Saved {key} centroids to {path}")

    def get_centroids(self):
        self.centroids = {"cam1": None, "cam2": None}
        for key in self.centroids.keys():
            path = self.preproc_dir / f"{self.config.name}_centroids_{key}.toml"
            if not path.exists():
                raise RuntimeError(
                    f"Could not locate centroid file for {key}, expected it to be at {path}"
                )
            with path.open("rb") as fh:
                centroids = tomli.load(fh)
            self.centroids[key] = {}
            for field, ctrs in centroids.items():
                self.centroids[key][field] = np.flip(np.atleast_2d(ctrs), axis=-1)

        logger.debug(f"Cam 1 frame center is {self.centroids['cam1']} (y, x)")
        logger.debug(f"Cam 2 frame center is {self.centroids['cam2']} (y, x)")
        return self.centroids

    def add_paths_to_db(self):
        input_paths = self.working_db["path"].apply(Path)
        # figure out which metrics need to be calculated, which is necessary to collapse files
        func = lambda p: self.workdir / self.config.analysis.get_output_path(p).absolute()
        self.working_db["metric_file"] = input_paths.apply(func)

        if self.config.collapse is not None:
            func = lambda p: self.workdir / self.config.collapse.get_output_path(p).absolute()
            self.working_db["collapse_file"] = input_paths.apply(func)

        if self.config.calibrate.save_intermediate:
            func = lambda p: self.workdir / self.config.calibrate.get_output_path(p).absolute()
            self.working_db["calib_file"] = input_paths.apply(func)

        if self.config.polarimetry:
            func = lambda p: self.workdir / self.config.polarimetry.get_output_path(p).absolute()
            self.working_db["mm_file"] = input_paths.apply(func)

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
        if len(self.subset_db) == 0:
            logger.success("Finished processing files (No files to process)")
            return
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
        self.save_output_header()
        ## products
        if self.config.save_adi_cubes:
            self.save_adi_cubes(force=force)
        ## diff images
        if self.config.make_diff_images:
            self.make_diff_images(force=force)

        logger.success("Finished processing files")

    def process_one(self, fileinfo):
        # fix headers and calibrate
        cur_hdu = self.calibrate_one(fileinfo["path"], fileinfo)
        ## Step 2: Frame analysis
        self.analyze_one(cur_hdu, fileinfo)
        ## Step 4: collapsing
        path = self.collapse_one(cur_hdu, fileinfo)
        return path

    def get_coordinate(self):
        if self.config.object is None:
            self.coord = None
        else:
            self.coord = self.config.object.get_coord()

    def calibrate_one(self, path, fileinfo, force=False):
        logger.debug("Starting data calibration")
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
        logger.debug("Data calibration completed")
        return calib_hdu

    def analyze_one(self, hdu: fits.PrimaryHDU, fileinfo, force=False):
        logger.debug("Starting frame analysis")
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
        logger.debug("Starting data calibration")
        config = self.config.collapse
        outpath = Path(fileinfo["collapse_file"])
        kwargs = dict(
            method=config.method,
            frame_select=config.frame_select,
            select_cutoff=config.select_cutoff,
            register=config.centroid,
            metric_file=fileinfo["metric_file"],
            recenter=config.recenter,
            centroids=self.centroids,
            outpath=outpath,
            force=force,
        )
        lucky_image_file(hdu, **kwargs)
        logger.debug("Data calibration completed")
        logger.debug(f"Saved collapsed data to {outpath}")
        return outpath

    def save_output_header(self):
        self.output_table.to_csv(self.output_table_path)
        return self.output_table_path

    def save_adi_cubes(self, force: bool = False):
        # preset values
        self.cam1_cube_path = self.cam2_cube_path = None

        # save cubes for each camera
        if "U_FLC" in self.output_table.keys():
            sort_keys = ["MJD", "U_FLC"]
        else:
            sort_keys = "MJD"

        for cam_num, group in self.output_table.sort_values(sort_keys).groupby("U_CAMERA"):
            cube_path = self.adi_dir / f"{self.config.name}_adi_cube_cam{cam_num:.0f}.fits"
            combine_frames_files(group["path"], output=cube_path, force=force)
            logger.info(f"Saved cam {cam_num:.0f} ADI cube to {cube_path}")
            angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
            angles = np.asarray(group["DEROTANG"], dtype="f4")
            fits.writeto(angles_path, angles, overwrite=True)
            logger.info(f"Saved cam {cam_num:.0f} ADI angles to {angles_path}")

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

    def create_stokes_table(self, table):
        keys = ["path", "MJD", "RET-ANG1", "U_CAMERA"]
        sort_keys = ["MJD", "RET-ANG1", "U_CAMERA"]
        if "U_FLC" in table.keys():
            keys.append("U_FLC")
            sort_keys.append("U_FLC")
        table = table.sort_values(sort_keys)
        subset = table[keys]
        return subset

    def make_mueller_mats(self, force=False):
        logger.info("Creating Mueller matrices")
        mm_paths = []
        kwds = dict(
            adi_sync=self.config.polarimetry.hwp_adi_sync,
            ideal=self.config.polarimetry.use_ideal_mm,
            force=force,
        )
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for row in self.working_db.itertuples(index=False):
                outpath = self.workdir / self.config.polarimetry.get_output_path(Path(row.path))
                jobs.append(
                    pool.apply_async(mueller_matrix_from_file, args=(row.path, outpath), kwds=kwds)
                )

            for job in tqdm(jobs, desc="Making Mueller matrices"):
                mm_paths.append(job.get())

        return mm_paths

    def run_polarimetry(self, num_proc, force=False):
        logger.debug(f"VAMPIRES DPP: v{dpp.__version__}")
        conf_copy_path = self.preproc_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        self.num_proc = num_proc

        if not self.output_table_path.exists():
            raise RuntimeError(f"Output table {self.output_table_path} cannot be found")

        self.working_db = pd.read_csv(self.output_table_path, index_col=0)

        if self.config.polarimetry.mm_correct or self.config.polarimetry.method == "leastsq":
            self.working_db["mm_file"] = self.make_mueller_mats()

        logger.info("Performing polarimetric calibration")
        logger.debug(f"Saving Stokes data to {self.polarimetry_dir.absolute()}")
        if self.config.polarimetry.method.endswith("diff"):
            self.polarimetry_difference(method=self.config.polarimetry.method, force=force)
        elif self.config.polarimetry.method == "leastsq":
            self.polarimetry_leastsq(force=force)
        logger.success("Finished PDI")

    def polarimetry_difference(self, method, force=False):
        config = self.config.polarimetry
        full_paths = []
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for _, row in self.working_db.iterrows():
                if method == "triplediff":
                    jobs.append(pool.apply_async(get_triplediff_set, args=(self.working_db, row)))
                else:
                    jobs.append(pool.apply_async(get_doublediff_set, args=(self.working_db, row)))

            for job in tqdm(jobs, desc="Forming Stokes sets"):
                stokes_set = job.get()
                full_paths.append(tuple(sorted(stokes_set.values())))

        full_path_set = list(set(paths for paths in full_paths))

        stokes_files = [
            self.stokes_dir / f"{self.config.name}_stokes_{i:03d}.fits"
            for i in range(len(full_path_set))
        ]
        stokes_data = []
        stokes_hdrs = []
        kwds = dict(
            method=method,
            mm_correct=config.mm_correct,
            ip_correct=config.ip_correct,
            ip_method=config.ip_method,
            ip_radius=config.ip_radius,
            force=force,
        )
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for outpath, path_set in zip(stokes_files, full_path_set):
                if config.mm_correct:
                    mm_paths = self.working_db.loc[
                        self.working_db["path"].apply(lambda p: p in path_set), "mm_file"
                    ]
                else:
                    mm_paths = None
                if len(path_set) not in (8, 16):
                    continue
                jobs.append(
                    pool.apply_async(
                        make_stokes_image, args=(path_set, outpath, mm_paths), kwds=kwds
                    )
                )

            for job in tqdm(jobs, desc="Creating Stokes images"):
                data, header = job.get()
                stokes_data.append(data)
                stokes_hdrs.append(header)

                full_paths.append(tuple(sorted(stokes_set.values())))
        remain_files = filter(lambda f: Path(f).exists(), stokes_files)
        ## Save CSV of Stokes values
        stokes_tbl = header_table(remain_files, quiet=True)
        stokes_tbl_path = self.polarimetry_dir / f"{self.config.name}_stokes_table.csv"
        stokes_tbl.to_csv(stokes_tbl_path)
        logger.info(f"Saved table of Stokes file headers to {stokes_tbl_path}")

        ## Collapse outputs
        collapse_frame, coll_hdr = collapse_frames(stokes_data, headers=stokes_hdrs)
        stokes_coll_path = self.polarimetry_dir / f"{self.config.name}_stokes_coll.fits"
        write_stokes_products(collapse_frame, header=coll_hdr, outname=stokes_coll_path, force=True)
        logger.info(f"Saved collapsed Stokes cube to {stokes_coll_path}")

    def polarimetry_leastsq(self, force=False):
        self.stokes_collapsed_file = self.polarimetry_dir / f"{self.config.name}_stokes_coll.fits"
        if (
            force
            or not self.stokes_collapsed_file.is_file()
            or any_file_newer(self.working_db["path"], self.stokes_collapsed_file)
        ):
            # create stokes cube
            polarization_calibration_leastsq(
                self.working_db["path"],
                self.working_db["mm_file"],
                outname=self.stokes_collapsed_file,
                force=True,
            )
