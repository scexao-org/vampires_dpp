import multiprocessing as mp
import warnings
from functools import partial
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
from vampires_dpp.calibration import calibrate_file, match_calib_files
from vampires_dpp.image_processing import (
    collapse_frames,
    collapse_frames_files,
    combine_frames_files,
    combine_frames_headers,
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

from ..lucky_imaging import lucky_image_file
from ..paths import Paths, any_file_newer, get_paths, make_dirs
from ..synthpsf import create_synth_psf
from ..util import load_fits
from .modules import get_psf_centroids_mpl


class Pipeline:
    def __init__(self, config: PipelineConfig, workdir=Path.cwd()):
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        self.diff_files = None
        self.config = config
        self.workdir = workdir
        self.paths = Paths(workdir=self.workdir)
        self.output_table_path = self.paths.products_dir / f"{self.config.name}_table.csv"
        # cache for PSFs
        self.synthpsf_cache = {}

    def create_input_table(self, filenames) -> pd.DataFrame:
        input_table = header_table(filenames, quiet=True).sort_values("MJD")
        table_path = self.paths.preproc_dir / f"{self.config.name}_headers.csv"
        input_table.to_csv(table_path)
        logger.info(f"Saved input header table to: {table_path}")
        return input_table

    def create_raw_input_psf(self, table, max_files=5) -> dict[str, Path]:
        # group by cameras
        outfiles = {}
        for cam_num, group in table.groupby("U_CAMERA"):
            paths = group["path"].sample(n=max_files)
            outpath = self.paths.preproc_dir / f"{self.config.name}_raw_psf_cam{cam_num:.0f}.fits"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outfile = collapse_frames_files(paths, output=outpath, cubes=True, quiet=False)
            outfiles[f"cam{cam_num:.0f}"] = outfile
            logger.info(f"Saved raw PSF frame to {outpath.absolute()}")
        return outfiles

    def save_centroids(self, table):
        raw_psf_filenames = self.create_raw_input_psf(table)
        for key in ("cam1", "cam2"):
            path = self.paths.preproc_dir / f"{self.config.name}_centroids_{key}.toml"

            im, hdr = load_fits(raw_psf_filenames[key], header=True)
            npsfs = 4 if self.config.coronagraphic else 1
            outpath = self.paths.preproc_dir / f"{self.config.name}_{key}.png"
            outpath.parent.mkdir(parents=True, exist_ok=True)
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
        self.centroids = {}
        for key in ("cam1", "cam2"):
            path = self.paths.preproc_dir / f"{self.config.name}_centroids_{key}.toml"
            if not path.exists():
                logger.warning(
                    f"Could not locate centroid file for {key}, expected it to be at {path}. Using center of image as default."
                )
                continue
            with path.open("rb") as fh:
                centroids = tomli.load(fh)
            self.centroids[key] = {}
            for field, ctrs in centroids.items():
                self.centroids[key][field] = np.flip(np.atleast_2d(ctrs), axis=-1)

            logger.debug(f"{key} frame center is {self.centroids['cam1']} (y, x)")
        return self.centroids

    def add_paths_to_db(self, table):
        input_paths = table["path"].apply(Path)
        # figure out which metrics need to be calculated, which is necessary to collapse files
        func = lambda p: get_paths(
            p, suffix="metrics", filetype=".npz", output_directory=self.paths.metrics_dir
        )[1]
        table["metric_file"] = input_paths.apply(func)

        if self.config.collapse is not None:
            func = lambda p: get_paths(
                p, suffix="coll", filetype=".fits", output_directory=self.paths.collapsed_dir
            )[1]
            table["collapse_file"] = input_paths.apply(func)

        if self.config.calibrate.save_intermediate:
            func = lambda p: get_paths(
                p, suffix="calib", filetype=".fits", output_directory=self.paths.calibrated_dir
            )[1]
            table["calib_file"] = input_paths.apply(func)

        return table

    def determine_execution(self, table, force=False):
        if force:
            return table
        file_doesnt_exist = lambda p: not Path(p).exists()
        files_to_calibrate = table["metric_file"].apply(file_doesnt_exist)
        if self.config.collapse is not None:
            files_to_calibrate |= table["collapse_file"].apply(file_doesnt_exist)

        subset = table.loc[files_to_calibrate]
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
        make_dirs(self.paths, self.config)
        logger.debug(f"VAMPIRES DPP: v{dpp.__version__}")
        conf_copy_path = self.paths.preproc_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        self.num_proc = num_proc
        input_table = self.create_input_table(filenames=filenames)
        self.add_paths_to_db(input_table)
        self.get_centroids()
        self.get_coordinate()

        working_table = self.determine_execution(input_table, force=force)
        if len(working_table) == 0:
            logger.success("Finished processing files (No files to process)")
            return
        logger.info(
            f"Processing {len(working_table)} files using {len(input_table) - len(working_table)} cached files"
        )
        working_table["backfile"] = None
        working_table["flatfile"] = None
        if self.config.calibrate.calib_directory is not None:
            cal_files = list(self.config.calibrate.calib_directory.glob("**/[!.]*.fits"))
            if len(cal_files) > 0:
                calib_table = match_calib_files(working_table["path"], cal_files)
                working_table["backfile"] = calib_table["backfile"]
                working_table["flatfile"] = calib_table["flatfile"]

        ## For each file do
        logger.info("Starting file-by-file processing")
        self.output_files = []
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for row in working_table.itertuples(index=False):
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
            raise NotImplementedError("Making diff images needs reworking")
            # self.make_diff_images(force=force)

        logger.success("Finished processing files")

    def process_one(self, fileinfo):
        # fix headers and calibrate
        logger.debug(f"Processing {fileinfo['path']}")
        cur_hdul = self.calibrate_one(fileinfo["path"], fileinfo)
        ## Step 2: Frame analysis
        self.analyze_one(cur_hdul, fileinfo)
        ## Step 3: collapsing
        path = self.collapse_one(cur_hdul, fileinfo)
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
        back_filename = flat_filename = None
        if config.back_subtract:
            back_filename = fileinfo["backfile"]
        if config.flat_correct:
            flat_filename = fileinfo["flatfile"]
        calib_hdul = calibrate_file(
            path,
            back_filename=back_filename,
            flat_filename=flat_filename,
            transform_filename=config.distortion_file,
            bpfix=config.fix_bad_pixels,
            coord=self.coord,
            force=force,
        )
        if config.save_intermediate:
            calib_hdul.writeto(fileinfo["calib_file"], overwrite=True)
            logger.debug(f"Calibrated data saved to {fileinfo['calib_file']}")
        logger.debug("Data calibration completed")
        return calib_hdul

    def analyze_one(self, hdul: fits.HDUList, fileinfo, force=False):
        logger.debug("Starting frame analysis")
        config = self.config.analysis

        mod = hdul[0].header["OBS-MOD"]
        if mod.endswith("MBIR"):
            filts = ("F670", "F720", "F760")
        elif mod.endswith("MBI"):
            filts = ("F610", "F670", "F720", "F760")
        elif mod.endswith("SDI"):
            filts = (hdul[0].header["FILTER02"],)
        else:
            filts = (hdul[0].header["FILTER01"],)

        psfs = []
        for filt in filts:
            if filt not in self.synthpsf_cache:
                psf = create_synth_psf(fileinfo, filt, output_directory=self.paths.preproc_dir)
                self.synthpsf_cache[filt] = psf
            psfs.append(self.synthpsf_cache[filt])

        key = f"cam{fileinfo['U_CAMERA']:.0f}"
        outpath = analyze_file(
            hdul,
            centroids=self.centroids.get(key, None),
            psfs=psfs,
            subtract_radprof=config.subtract_radprof,
            aper_rad=config.aper_rad,
            ann_rad=config.ann_rad,
            outpath=fileinfo["metric_file"],
            force=force,
            window=config.window_size,
        )
        return outpath

    def collapse_one(self, hdul, fileinfo, force=False):
        logger.debug("Starting data collapsing")
        config = self.config.collapse
        outpath = Path(fileinfo["collapse_file"])
        lucky_image_file(
            hdul,
            method=config.method,
            frame_select=config.frame_select,
            select_cutoff=config.select_cutoff,
            register=config.centroid,
            metric_file=fileinfo["metric_file"],
            recenter=config.recenter,
            centroids=self.centroids,
            outpath=outpath,
            force=force,
            specphot=self.config.specphot,
            preproc_dir=self.paths.preproc_dir,
        )
        logger.debug("Data collapsing completed")
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
            self.output_table.sort_values(["MJD", "U_FLC"], inplace=True)
        else:
            self.output_table.sort_values("MJD", inplace=True)

        for cam_num, group in self.output_table.groupby("U_CAMERA"):
            cube_path = self.paths.adi_dir / f"{self.config.name}_adi_cube_cam{cam_num:.0f}.fits"
            combine_frames_files(group["path"], output=cube_path, force=force)
            logger.info(f"Saved cam {cam_num:.0f} ADI cube to {cube_path}")
            angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
            angles = np.asarray(group["DEROTANG"], dtype="f4")
            fits.writeto(angles_path, angles, overwrite=True)
            logger.info(f"Saved cam {cam_num:.0f} ADI angles to {angles_path}")

    # def make_diff_images(self, force=False):
    #     logger.info("Making difference frames")
    #     # table should still be sorted by MJD
    #     groups = self.output_table.groupby(["MJD", "U_CAMERA"])
    #     cam1_paths = []
    #     cam2_paths = []
    #     for key, group in groups:
    #         if key[1] == 1:
    #             cam1_paths.append(group["path"].iloc[0])
    #         elif key[1] == 2:
    #             cam2_paths.append(group["path"].iloc[0])

    #     with mp.Pool(self.num_proc) as pool:
    #         jobs = []
    #         for cam1_file, cam2_file in zip(cam1_paths, cam2_paths):
    #             stem = re.sub("_cam[12]", "", cam1_file.name)
    #             outname = outdir / stem.replace(".fits", "_diff.fits")
    #             logger.debug(f"loading cam1 image from {cam1_file.absolute()}")
    #             logger.debug(f"loading cam2 image from {cam2_file.absolute()}")
    #             kwds = dict(outname=outname, force=force)
    #             jobs.append(
    #                 pool.apply_async(make_diff_image, args=(cam1_file, cam2_file), kwds=kwds)
    #             )

    #         self.diff_files = [job.get() for job in tqdm(jobs, desc="Making diff images")]
    #     logger.info("Done making difference frames")
    #     return self.diff_files

    def run_polarimetry(self, num_proc, force=False):
        make_dirs(self.paths, self.config)
        logger.debug(f"VAMPIRES DPP: v{dpp.__version__}")
        conf_copy_path = self.paths.preproc_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        self.num_proc = num_proc

        if not self.output_table_path.exists():
            raise RuntimeError(f"Output table {self.output_table_path} cannot be found")

        working_table = pd.read_csv(self.output_table_path, index_col=0)

        if self.config.polarimetry.mm_correct or self.config.polarimetry.method == "leastsq":
            working_table["mm_file"] = self.make_mueller_mats(working_table)

        logger.info("Performing polarimetric calibration")
        logger.debug(f"Saving Stokes data to {self.paths.pdi_dir.absolute()}")
        match self.config.polarimetry.method:
            case "doublediff" | "triplediff":
                self.polarimetry_difference(
                    working_table, method=self.config.polarimetry.method, force=force
                )
            case "leastsq":
                self.polarimetry_leastsq(working_table, force=force)
        logger.success("Finished PDI")

    def make_mueller_mats(self, table, force=False):
        logger.info("Creating Mueller matrices")
        mm_paths = []
        kwds = dict(
            adi_sync=self.config.polarimetry.hwp_adi_sync,
            ideal=self.config.polarimetry.use_ideal_mm,
            force=force,
        )
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for row in table.itertuples(index=False):
                _, outpath = get_paths(row.path, suffix="mm", output_directory=self.paths.mm_dir)
                jobs.append(
                    pool.apply_async(mueller_matrix_from_file, args=(row.path, outpath), kwds=kwds)
                )

            for job in tqdm(jobs, desc="Making Mueller matrices"):
                mm_paths.append(job.get())

        return mm_paths

    def polarimetry_difference(self, table, method, force=False):
        config = self.config.polarimetry
        full_paths = []
        match method.lower():
            case "triplediff":
                pol_func = partial(get_triplediff_set, table)
            case "doublediff":
                pol_func = partial(get_doublediff_set, table)
            case _:
                raise ValueError(f"Invalid polarimetric difference method '{method}'")

        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for _, row in table.iterrows():
                jobs.append(pool.apply_async(pol_func, args=(row,)))
            for job in tqdm(jobs, desc="Forming Stokes sets"):
                stokes_set = job.get()
                full_paths.append(tuple(sorted(stokes_set.values())))

        full_path_set = list(set(paths for paths in full_paths))

        stokes_files = [
            self.paths.stokes_dir / f"{self.config.name}_stokes_{i:03d}.fits"
            for i in range(len(full_path_set))
        ]
        # peek to get nfields
        with fits.open(full_path_set[0][0]) as hdul:
            nfields = hdul[0].shape[0]

        stokes_data = []
        stokes_hdrs = [[] for _ in range(nfields + 1)]
        stokes_err = [[] for _ in range(nfields)]
        kwds = dict(
            method=method,
            mm_correct=config.mm_correct,
            ip_correct=config.ip_correct,
            ip_method=config.ip_method,
            ip_radius=config.ip_radius,
            ip_radius2=config.ip_radius2,
            force=force,
        )
        # TODO this is kind of ugly
        with mp.Pool(self.num_proc) as pool:
            jobs = []
            for outpath, path_set in zip(stokes_files, full_path_set):
                if config.mm_correct:
                    mm_paths = table.loc[table["path"].apply(lambda p: p in path_set), "mm_file"]
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
                outpath = job.get()
                with fits.open(outpath) as hdul:
                    stokes_data.append(hdul[0].data)
                    for i in range(nfields + 1):
                        stokes_hdrs[i].append(hdul[i].header)
                    for i in range(nfields):
                        err_extname = f"{hdul[i + 1].header['FIELD']}ERR"
                        stokes_err[i].append(hdul[err_extname].data)
        remain_files = filter(lambda f: Path(f).exists(), stokes_files)
        ## Save CSV of Stokes values
        stokes_tbl = header_table(remain_files, fix=False, quiet=True)
        stokes_tbl_path = self.paths.pdi_dir / f"{self.config.name}_stokes_table.csv"
        stokes_tbl.to_csv(stokes_tbl_path)
        logger.info(f"Saved table of Stokes file headers to {stokes_tbl_path}")
        logger.info(f"Collapsing {len(stokes_tbl)} Stokes files...")
        ## Collapse outputs
        coll_frame, coll_hdr = collapse_frames(stokes_data, headers=stokes_hdrs[0])
        coll_hdrs = [
            combine_frames_headers(stokes_hdrs[i], wcs=True) for i in range(1, nfields + 1)
        ]
        coll_errs = []
        prim_hdu = fits.PrimaryHDU(coll_frame, header=coll_hdr)
        hdul = fits.HDUList(prim_hdu)
        for i in range(nfields):
            hdu = fits.ImageHDU(coll_frame[i], header=coll_hdrs[i], name=coll_hdrs[i]["FIELD"])
            hdul.append(hdu)
        for i in range(nfields):
            err_list = stokes_err[i]
            hdr = coll_hdrs[i]
            coll_err = np.sqrt(collapse_frames(np.power(err_list, 2))[0])
            coll_errs.append(coll_err)
            hdu_err = fits.ImageHDU(coll_err, header=hdr, name=f"{hdr['FIELD']}ERR")
            hdul.append(hdu_err)
        # In the case we have multi-wavelength data, save 4D Stokes cube
        if nfields > 1:
            stokes_cube_path = self.paths.pdi_dir / f"{self.config.name}_stokes_cube.fits"
            write_stokes_products(hdul, outname=stokes_cube_path, force=True)
            logger.info(f"Saved Stokes cube to {stokes_cube_path}")

            # now collapse wavelength axis and clobber
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wave_coll_frame = np.nansum(coll_frame, axis=0, keepdims=True)
                wave_err_frame = np.sqrt(np.nanmean(np.power(coll_errs, 2), axis=0, keepdims=True))
            wave_coll_hdr = combine_frames_headers(coll_hdrs, wcs=True)
            # TODO some fits keywords here are screwed up
            hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(wave_coll_frame, header=wave_coll_hdr),
                    fits.ImageHDU(wave_err_frame, header=wave_coll_hdr, name="ERR"),
                ]
            )
        # save single-wavelength (or wavelength-collapsed) Stokes cube
        stokes_coll_path = self.paths.pdi_dir / f"{self.config.name}_stokes_coll.fits"
        write_stokes_products(hdul, outname=stokes_coll_path, force=True)
        logger.info(f"Saved collapsed Stokes cube to {stokes_coll_path}")

    def polarimetry_leastsq(self, table, force=False):
        raise NotImplementedError("Need to rewrite this, sorry.")
        # self.stokes_collapsed_file = self.paths.pdi_dir / f"{self.config.name}_stokes_coll.fits"
        # if (
        #     force
        #     or not self.stokes_collapsed_file.is_file()
        #     or any_file_newer(self.working_db["path"], self.stokes_collapsed_file)
        # ):
        #     # create stokes cube
        #     polarization_calibration_leastsq(
        #         self.working_db["path"],
        #         self.working_db["mm_file"],
        #         outname=self.stokes_collapsed_file,
        #         force=True,
        #     )
