import multiprocessing as mp
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tomli
from astropy.io import fits
from loguru import logger
from tqdm.auto import tqdm

from vampires_dpp.analysis import analyze_file
from vampires_dpp.calib.calib_files import match_calib_files
from vampires_dpp.calib.calibration import calibrate_file
from vampires_dpp.image_processing import (
    collapse_frames,
    combine_frames_files,
    combine_frames_headers,
)
from vampires_dpp.lucky_imaging import lucky_image_file
from vampires_dpp.organization import header_table
from vampires_dpp.paths import Paths, get_paths, make_dirs
from vampires_dpp.pdi.diff_images import (
    doublediff_images,
    get_doublediff_sets,
    get_singlediff_sets,
    singlediff_images,
)
from vampires_dpp.pdi.models import mueller_matrix_from_file
from vampires_dpp.pdi.processing import get_doublediff_set, get_triplediff_set, make_stokes_image
from vampires_dpp.pdi.utils import write_stokes_products
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.synthpsf import create_synth_psf
from vampires_dpp.wcs import apply_wcs


class Pipeline:
    def __init__(self, config: PipelineConfig, workdir: Path | None = None):
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        self.diff_files = None
        self.config = config
        self.workdir = workdir if workdir is not None else Path.cwd()
        self.paths = Paths(workdir=self.workdir)
        self.output_table_path = self.paths.products_dir / f"{self.config.name}_table.csv"

    def create_input_table(self, filenames, num_proc) -> pd.DataFrame:
        input_table = header_table(filenames, quiet=True, num_proc=num_proc).sort_values("MJD")
        table_path = self.paths.aux_dir / f"{self.config.name}_headers.csv"
        input_table.to_csv(table_path)
        logger.info(f"Saved input header table to: {table_path}")
        return input_table

    def get_centroids(self):
        self.centroids = {}
        for key in ("cam1", "cam2"):
            path = self.paths.aux_dir / f"{self.config.name}_centroids_{key}.toml"
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

            logger.debug(f"{key} frame center is {self.centroids[key]} (y, x)")
        return self.centroids

    def add_paths_to_db(self, table):
        input_paths = table["path"].apply(Path)

        # figure out which metrics need to be calculated, which is necessary to collapse files
        def func(p):
            return get_paths(
                p, suffix="metrics", filetype=".npz", output_directory=self.paths.metrics_dir
            )[1]

        table["metric_file"] = input_paths.apply(func)

        if self.config.collapse is not None:

            def func(p):
                return get_paths(
                    p, suffix="coll", filetype=".fits", output_directory=self.paths.collapsed_dir
                )[1]

            table["collapse_file"] = input_paths.apply(func)

        if self.config.calibrate.save_intermediate:

            def func(p):
                return get_paths(
                    p, suffix="calib", filetype=".fits", output_directory=self.paths.calibrated_dir
                )[1]

            table["calib_file"] = input_paths.apply(func)

        return table

    def determine_execution(self, table, force=False):
        if force:
            return table

        def file_doesnt_exist(p):
            return not Path(p).exists()

        files_to_calibrate = table["metric_file"].apply(file_doesnt_exist)
        if self.config.collapse is not None:
            files_to_calibrate |= table["collapse_file"].apply(file_doesnt_exist)

        subset = table.loc[files_to_calibrate]
        return subset.copy()

    def _determine_filts_from_header(self, header):
        mod = header["OBS-MOD"]
        if mod.endswith("MBIR"):
            filts = ("F670", "F720", "F760")
        elif mod.endswith("MBI"):
            filts = ("F610", "F670", "F720", "F760")
        elif mod.endswith("SDI"):
            filts = (header["FILTER02"],)
        else:
            filts = (header["FILTER01"],)
        return filts

    def make_synth_psfs(self, input_table):
        # make PSFs ahead of time so they don't overwhelm
        # during multiprocessing
        filters = {}
        for _, row in input_table.iterrows():
            for filt in self._determine_filts_from_header(row):
                filters[filt] = row

        self.synth_psfs = {}
        for filt, row in filters.items():
            psf = create_synth_psf(
                row,
                filt,
                npix=self.config.analysis.window_size,
                output_directory=self.paths.aux_dir,
            )
            self.synth_psfs[filt] = psf

    def run(self, filenames, num_proc: int | None = None, force=False):
        """Run the pipeline

        Parameters
        ----------
        filenames : Iterable[PathLike]
            Input filenames to process
        num_proc : Optional[int]
            Number of processes to use for multi-processing, by default None.
        """
        make_dirs(self.paths, self.config)
        conf_copy_path = self.paths.aux_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        input_table = self.create_input_table(filenames=filenames, num_proc=num_proc)
        self.add_paths_to_db(input_table)
        self.get_centroids()
        self.get_coordinate()
        self.make_synth_psfs(input_table)

        working_table = self.determine_execution(input_table, force=force)
        if len(working_table) == 0:
            logger.success("Finished processing files (No files to process)")
        else:
            logger.info(
                f"Processing {len(working_table)} files using {len(input_table) - len(working_table)} cached files"
            )

            if self.config.calibrate.calib_directory is not None:
                cal_files = list(self.config.calibrate.calib_directory.glob("**/[!.]*.fits"))
                if len(cal_files) > 0:
                    calib_table = match_calib_files(working_table["path"], cal_files)
                    working_table = pd.merge(working_table, calib_table, on="path")

            ## For each file do
            logger.info("Starting file-by-file processing")
            with mp.Pool(num_proc) as pool:
                jobs = []
                for row in working_table.itertuples(index=False):
                    jobs.append(pool.apply_async(self.process_one, args=(row._asdict(),)))

                for job in tqdm(jobs, desc="Processing files"):
                    job.get()
        logger.info("Creating table from collapsed headers")
        self.output_files = input_table["collapse_file"]
        self.output_table = header_table(self.output_files, num_proc=num_proc, quiet=True)
        self.save_output_header()
        ## products
        if self.config.save_adi_cubes:
            self.save_adi_cubes(force=force)
        ## diff images
        if self.config.diff_images is not None:
            self.make_diff_images(self.output_table, force=force)

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
        if self.config.target is None:
            self.coord = None
        else:
            self.coord = self.config.target.get_coord()

    def calibrate_one(self, path, fileinfo, force=False):
        logger.debug("Starting data calibration")
        config = self.config.calibrate
        if config.save_intermediate:
            outpath = Path(fileinfo["calib_file"])
            if not force and outpath.exists():
                return fits.open(outpath)
        back_filename = flat_filename = None
        if config.back_subtract and "backfile" in fileinfo:
            back_filename = fileinfo["backfile"]
        if config.flat_correct and "flatfile" in fileinfo:
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

        psfs = [self.synth_psfs[filt] for filt in self._determine_filts_from_header(fileinfo)]

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
            window_size=config.window_size,
        )
        return outpath

    def collapse_one(self, hdul, fileinfo, force=False):
        logger.debug("Starting data collapsing")
        config = self.config.collapse
        outpath = Path(fileinfo["collapse_file"])
        if config.reproject:
            with Path(self.paths.aux_dir / f"{self.config.name}_astrometry.toml").open("rb") as fh:
                astrometry = tomli.load(fh)
        else:
            astrometry = None
        lucky_image_file(
            hdul,
            method=config.method,
            frame_select=config.frame_select,
            select_cutoff=config.select_cutoff,
            register=config.centroid,
            metric_file=fileinfo["metric_file"],
            recenter=config.recenter,
            reproject=config.reproject,
            astrometry=astrometry,
            refsep=config.satspot_reference["separation"],
            refang=config.satspot_reference["angle"],
            centroids=self.centroids,
            outpath=outpath,
            force=force,
            specphot=self.config.specphot,
            aux_dir=self.paths.aux_dir,
            window=self.config.analysis.window_size,
            psfs=self.psfs,
        )
        logger.debug("Data collapsing completed")
        logger.debug(f"Saved collapsed data to {outpath}")
        return outpath

    def save_output_header(self):
        self.output_table.to_csv(self.output_table_path)
        logger.info(f"Saved output header table to {self.output_table_path}")
        return self.output_table_path

    def save_adi_cubes(self, force: bool = False):
        # preset values
        self.cam1_cube_path = self.cam2_cube_path = None

        # save cubes for each camera
        if "U_FLC" in self.output_table:
            self.output_table.sort_values(["MJD", "U_FLC"], inplace=True)
        else:
            self.output_table.sort_values("MJD", inplace=True)

        hduls = []
        cam_groups = self.output_table.groupby("U_CAMERA")
        for cam_num, group in cam_groups:
            cube_path = self.paths.adi_dir / f"{self.config.name}_adi_cube_cam{cam_num:.0f}.fits"
            combine_frames_files(group["path"], output=cube_path, force=force, crop=False)
            hduls.append(fits.open(cube_path, memmap=False))
            logger.info(f"Saved cam {cam_num:.0f} ADI cube to {cube_path}")
            angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
            angles = np.asarray(group["DEROTANG"], dtype="f4")
            fits.writeto(angles_path, angles, overwrite=True)
            logger.info(f"Saved cam {cam_num:.0f} ADI angles to {angles_path}")
        # take cam 1 as refernce, why not
        # if len(hduls) > 1:
        #     # cam1_cube, cam1_hdr, cam2_cube, cam2_hdr = reproject_data(
        #     #     hduls[0][0].data, hduls[0][0].header, hduls[1][0].data, hduls[1][0].header
        #     # )
        #     cam1_cube = hduls[0][0].data
        #     cam1_hdr = hduls[0][0].header
        #     cam2_cube = hduls[1][0].data
        #     # cam2_hdr = hduls[1][0].header

        #     mean_cube = 0.5 * (cam1_cube + cam2_cube)
        #     cube_path = self.paths.adi_dir / f"{self.config.name}_adi_cube_comb.fits"
        #     del cam1_hdr["U_CAMERA"]
        #     fits.writeto(cube_path, mean_cube, header=cam1_hdr, overwrite=True)
        #     logger.info(f"Saved combined ADI cube to {cube_path}")
        #     angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
        #     angles = np.asarray(cam_groups.get_group(1)["DEROTANG"])
        #     fits.writeto(angles_path, angles, overwrite=True)
        #     logger.info(f"Saved combined ADI angles to {angles_path}")

    def make_diff_images(self, table, num_proc=None, force=False):
        logger.info("Making difference frames")
        self.diff_files = []
        if self.config.diff_images == "singlediff":
            path_sets = get_singlediff_sets(table)
            diff_func = partial(singlediff_images, force=force)
            outdir = self.paths.diff_dir / "single"
        elif self.config.diff_images == "doublediff":
            path_sets = get_doublediff_sets(table)
            diff_func = partial(doublediff_images, force=force)
            outdir = self.paths.diff_dir / "double"
        elif self.config.diff_images == "both":
            # do singlediff first, then deliberate to doublediff
            path_sets = get_singlediff_sets(table)
            diff_func = partial(singlediff_images, force=force)
            outdir = self.paths.diff_dir / "single"
            outdir.mkdir(exist_ok=True)
            with mp.Pool(num_proc) as pool:
                jobs = []
                for i, paths in enumerate(path_sets):
                    outpath = outdir / f"{self.config.name}_diff_{i:04d}.fits"
                    jobs.append(
                        pool.apply_async(diff_func, args=(paths,), kwds=dict(outpath=outpath))
                    )
                self.diff_files.extend(
                    job.get() for job in tqdm(jobs, desc="Making single diff images")
                )
            # now set for double-diff
            path_sets = get_doublediff_sets(table)
            diff_func = partial(doublediff_images, force=force)
            outdir = self.paths.diff_dir / "double"
        outdir.mkdir(exist_ok=True)
        with mp.Pool(num_proc) as pool:
            jobs = []
            for i, paths in enumerate(path_sets):
                outpath = outdir / f"{self.config.name}_diff_{i:04d}.fits"
                jobs.append(pool.apply_async(diff_func, args=(paths,), kwds=dict(outpath=outpath)))

            self.diff_files.extend(job.get() for job in tqdm(jobs, desc="Making diff images"))
        logger.info("Done making difference frames")
        return self.diff_files

    def run_polarimetry(self, num_proc, force=False):
        make_dirs(self.paths, self.config)
        conf_copy_path = self.paths.aux_dir / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        if not self.output_table_path.exists():
            msg = f"Output table {self.output_table_path} cannot be found"
            raise RuntimeError(msg)

        working_table = pd.read_csv(self.output_table_path, index_col=0).sort_values("MJD")

        if self.config.polarimetry.mm_correct or self.config.polarimetry.method == "leastsq":
            working_table["mm_file"] = self.make_mueller_mats(working_table, num_proc=num_proc)

        logger.info("Performing polarimetric calibration")
        logger.debug(f"Saving Stokes data to {self.paths.pdi_dir.absolute()}")
        match self.config.polarimetry.method:
            case "doublediff" | "triplediff":
                self.polarimetry_difference(
                    working_table,
                    method=self.config.polarimetry.method,
                    force=force,
                    num_proc=num_proc,
                )
            case "leastsq":
                self.polarimetry_leastsq(working_table, force=force, num_proc=num_proc)
        logger.success("Finished PDI")

    def make_mueller_mats(self, table, num_proc=None, force=False):
        logger.info("Creating Mueller matrices")
        mm_paths = []
        kwds = dict(
            hwp_adi_sync=self.config.polarimetry.hwp_adi_sync,
            ideal=self.config.polarimetry.use_ideal_mm,
            force=force,
        )
        with mp.Pool(num_proc) as pool:
            jobs = []
            for row in table.itertuples(index=False):
                _, outpath = get_paths(row.path, suffix="mm", output_directory=self.paths.mm_dir)
                jobs.append(
                    pool.apply_async(mueller_matrix_from_file, args=(row.path, outpath), kwds=kwds)
                )

            for job in tqdm(jobs, desc="Making Mueller matrices"):
                mm_paths.append(job.get())

        return mm_paths

    def polarimetry_difference(self, table, method, num_proc=None, force=False):
        config = self.config.polarimetry
        full_paths = []
        match method.lower():
            case "triplediff":
                pol_func = partial(get_triplediff_set, table)
            case "doublediff":
                pol_func = partial(get_doublediff_set, table)
            case _:
                msg = f"Invalid polarimetric difference method '{method}'"
                raise ValueError(msg)

        with mp.Pool(num_proc) as pool:
            jobs = []
            for _, row in table.iterrows():
                jobs.append(pool.apply_async(pol_func, args=(row,)))
            for job in tqdm(jobs, desc="Determing Stokes frame combinations"):
                stokes_set = job.get()
                if stokes_set is not None:
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
        stokes_err = []
        stokes_hdrs = []
        stokes_func = partial(
            make_stokes_image,
            method=method,
            mm_correct=config.mm_correct,
            hwp_adi_sync=config.hwp_adi_sync,
            ip_correct=config.ip_correct,
            ip_method=config.ip_method,
            ip_radius=config.ip_radius,
            ip_radius2=config.ip_radius2,
            force=force,
        )
        # TODO this is kind of ugly
        with mp.Pool(num_proc) as pool:
            jobs = []
            for outpath, path_set in zip(stokes_files, full_path_set, strict=True):
                if config.mm_correct:
                    mask = [p in path_set for p in table["path"]]
                    subset = table.loc[mask]
                    mm_paths = subset["mm_file"]
                else:
                    mm_paths = None
                if len(path_set) not in (8, 16):
                    continue
                jobs.append(pool.apply_async(stokes_func, args=(path_set, outpath, mm_paths)))

            for job in tqdm(jobs, desc="Creating Stokes images"):
                outpath = job.get()
                # use memmap=False to avoid "too many files open" effects
                # another way would be to set ulimit -n <MAX_FILES>
                with fits.open(outpath, memmap=False) as hdul:
                    stokes_data.append(hdul[0].data)
                    stokes_err.append(hdul["ERR"].data)
                    hdrs = [hdul[i].header for i in range(3, len(hdul))]
                    stokes_hdrs.append(hdrs)

        remain_files = filter(lambda f: Path(f).exists(), stokes_files)
        ## Save CSV of Stokes values
        stokes_tbl = header_table(remain_files, fix=False, quiet=True)
        stokes_tbl_path = self.paths.pdi_dir / f"{self.config.name}_stokes_table.csv"
        stokes_tbl.to_csv(stokes_tbl_path)
        logger.info(f"Saved table of Stokes file headers to {stokes_tbl_path}")
        logger.info(f"Collapsing {len(stokes_tbl)} Stokes files...")
        ## Collapse outputs
        stokes_data = np.array(stokes_data)
        stokes_err = np.array(stokes_err)
        coll_frame, _ = collapse_frames(stokes_data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coll_err = np.sqrt(np.nansum(stokes_err**2, axis=0)) / stokes_err.shape[0]
        coll_hdrs = []
        for i in range(nfields):
            hdrs = [hdr[i] for hdr in stokes_hdrs]
            hdr = apply_wcs(coll_frame, combine_frames_headers(hdrs), angle=0)
            coll_hdrs.append(hdr)

        # correct TINT to account for actual number of files used
        unique_files = []
        for paths in full_path_set:
            unique_files.extend(paths)
        tints = [fits.getval(path, "TINT") for path in np.unique(unique_files)]
        tint = np.sum(tints)
        for hdr in coll_hdrs:
            hdr["NCOADD"] = len(tints)
            hdr["TINT"] = tint
        prim_hdr = apply_wcs(coll_frame, combine_frames_headers(coll_hdrs), angle=0)
        prim_hdr["NCOADD"] /= len(coll_hdrs)
        prim_hdr["TINT"] /= len(coll_hdrs)
        prim_hdu = fits.PrimaryHDU(coll_frame, header=prim_hdr)
        err_hdu = fits.ImageHDU(coll_err, header=prim_hdr, name="ERR")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr = prim_hdu.data / err_hdu.data
        snr_hdu = fits.ImageHDU(snr, header=prim_hdr, name="SNR")
        hdul = fits.HDUList([prim_hdu, err_hdu, snr_hdu])
        hdul.extend([fits.ImageHDU(header=hdr, name=hdr["FIELD"]) for hdr in coll_hdrs])
        # In the case we have multi-wavelength data, save 4D Stokes cube
        if nfields > 1:
            stokes_cube_path = self.paths.pdi_dir / f"{self.config.name}_stokes_cube.fits"
            write_stokes_products(
                hdul, outname=stokes_cube_path, force=True, planetary=config.planetary
            )
            logger.info(f"Saved Stokes cube to {stokes_cube_path}")

            # now collapse wavelength axis and clobber
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wave_coll_frame = np.nansum(coll_frame, axis=0, keepdims=True)
                wave_err_frame = np.sqrt(np.nansum(coll_err**2, axis=0, keepdims=True))
            wave_coll_hdr = apply_wcs(wave_coll_frame, combine_frames_headers(coll_hdrs), angle=0)
            wave_coll_hdr["NCOADD"] /= len(coll_hdrs)
            wave_coll_hdr["TINT"] /= len(coll_hdrs)
            wave_coll_hdr["FIELD"] = "COMB"
            # TODO some fits keywords here are screwed up
            prim_hdu = fits.PrimaryHDU(wave_coll_frame, header=wave_coll_hdr)
            err_hdu = fits.ImageHDU(wave_err_frame, header=wave_coll_hdr, name="ERR")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                snr = prim_hdu.data / err_hdu.data
            snr_hdu = fits.ImageHDU(snr, header=wave_coll_hdr, name="SNR")
            dummy_hdu = fits.ImageHDU(header=wave_coll_hdr, name="COMB")
            hdul = fits.HDUList([prim_hdu, err_hdu, snr_hdu, dummy_hdu])
        # save single-wavelength (or wavelength-collapsed) Stokes cube
        stokes_coll_path = self.paths.pdi_dir / f"{self.config.name}_stokes_coll.fits"
        write_stokes_products(
            hdul, outname=stokes_coll_path, force=True, planetary=config.planetary
        )
        logger.info(f"Saved collapsed Stokes cube to {stokes_coll_path}")

    def polarimetry_leastsq(self, table, force=False):
        msg = "Need to rewrite this, sorry."
        raise NotImplementedError(msg)
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
