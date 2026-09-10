import multiprocessing as mp
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tomli
from astropy.io import fits
from loguru import logger
from skimage import transform
from tqdm.auto import tqdm

from vampires_dpp.analysis import (
    add_coadd_metrics_to_header,
    add_metrics_to_header,
    analyze_coadded_hdul,
    analyze_file,
)
from vampires_dpp.calib.calib_files import match_calib_file
from vampires_dpp.calib.calibration import calibrate_file
from vampires_dpp.coadd import coadd_hdul, collapse_frames
from vampires_dpp.combine_frames import (
    combine_frames_headers,
    combine_hduls,
    generate_frame_combinations,
)
from vampires_dpp.constants import NBS_INSTALL_MJD
from vampires_dpp.frame_select import frame_select_hdul
from vampires_dpp.headers import sort_headers_hdul
from vampires_dpp.logging_utils import configure_subprocesslogging_utils
from vampires_dpp.nrm.extraction import extract_observables
from vampires_dpp.nrm.pdi import process_nrm_polarimetry
from vampires_dpp.nrm.plotting import make_nrm_plots
from vampires_dpp.organization import dict_from_header, header_table
from vampires_dpp.paths import (
    Paths,
    any_file_newer,
    get_nrm_paths,
    get_paths,
    get_reduced_path,
    make_dirs,
)
from vampires_dpp.pdi.diff_images import (
    doublediff_images,
    get_doublediff_sets,
    get_singlediff_sets,
    singlediff_images,
)
from vampires_dpp.pdi.models import mueller_matrix_from_file
from vampires_dpp.pdi.processing import (
    get_doublediff_set,
    get_triplediff_set,
    make_stokes_image,
    optimize_uphi_offsets,
)
from vampires_dpp.pdi.utils import rotate_stokes, write_stokes_products
from vampires_dpp.pipeline.config import PipelineConfig
from vampires_dpp.registration import intersect_point, recenter_hdul, register_hdul
from vampires_dpp.specphot.filters import determine_filterset_from_header
from vampires_dpp.specphot.specphot import (
    read_prior_specphot_factors,
    shape_factor_for_data,
    specphot_cal_hdul,
)
from vampires_dpp.synthpsf import create_synth_psf
from vampires_dpp.util import add_timestamp_hdul, get_center
from vampires_dpp.wcs import apply_wcs

PIPELINE_STAGES = (
    "calibrate",
    "combine",
    "metrics",
    "select",
    "align",
    "specphot",
    "coadd",
    "adi",
    "diff",
    "pdi",
    "all",
)

# Cap concurrent workers when num_proc is unspecified. The per-group working set
# (calibrated cubes, aligned cubes, registration scratch) can easily reach tens
# of GB; fanning out to cpu_count() workers on a large server explodes RAM.
DEFAULT_NUM_PROC = 8


def _resolve_num_proc(num_proc: int | None) -> int:
    if num_proc is not None:
        return num_proc
    cpus = mp.cpu_count()
    return min(DEFAULT_NUM_PROC, cpus)


class Pipeline:
    def __init__(self, config: PipelineConfig, workdir: Path | None = None, verbose: bool = False):
        self.master_backgrounds = {1: None, 2: None}
        self.master_flats = {1: None, 2: None}
        self.diff_files = None
        self.calib_table = None
        self.centroids = None
        self.reproject_tforms = None
        self.config = config
        self.workdir = workdir if workdir is not None else Path.cwd()
        self.paths = Paths(workdir=self.workdir)
        self.output_table_path = self.paths.aux / f"{self.config.name}_table.csv"
        self.verbose = verbose

    def run(self, filenames, num_proc: int | None = None, redo: str | None = None):
        """Run the pipeline

        Parameters
        ----------
        filenames : Iterable[PathLike]
            Input filenames to process
        num_proc : Optional[int]
            Number of processes to use for multi-processing. When ``None``, the
            pipeline caps the worker count at ``DEFAULT_NUM_PROC`` to keep peak
            RAM bounded on large machines.
        redo : str, optional
            Force a specific stage to rerun. Downstream stages cascade via the dirty flag.
            One of: "calibrate", "combine", "metrics", "select", "align", "specphot", "coadd",
            "adi", "diff", "pdi", "all". "specphot" is an alias for "align" (they share the
            aligned intermediate). "all" reruns every stage from the start.
        """
        num_proc = _resolve_num_proc(num_proc)
        # calibrate/combine/metrics/align/specphot are always active; select and coadd are optional.
        # Only force process stages that are actually enabled in the config.
        _enabled_process_stages = {"calibrate", "combine", "metrics", "align", "specphot"}
        if self.config.frame_select.frame_select:
            _enabled_process_stages.add("select")
        if self.config.coadd.coadd:
            _enabled_process_stages.add("coadd")
        force_process = redo == "all" or redo in _enabled_process_stages
        force_adi = redo in ("adi", "all")
        force_diff = redo in ("diff", "all")

        make_dirs(self.paths, self.config)
        conf_copy_path = self.paths.aux / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        input_table = self.create_input_table(filenames=filenames, num_proc=num_proc)
        self.get_centroids()
        if self.config.nrm is not None:
            self.get_uv_thetas()
        if self.config.align.reproject:
            self.get_reproject_tforms()
        self.get_coordinate()
        self.make_synth_psfs(input_table)
        combinations = generate_frame_combinations(input_table, method=self.config.combine.method)
        combinations_path = self.paths.aux / f"{self.config.name}_file_combinations.csv"
        combinations.to_csv(combinations_path, index=False)
        logger.info(f"Saved file combination table to {combinations_path.absolute()}")
        input_table["GROUP_KEY"] = combinations["GROUP_KEY"]

        if self.config.calibrate.calib_directory is not None:
            self.calib_table = header_table(
                self.config.calibrate.calib_directory.glob("**/[!.]*.fits"), quiet=True
            )
            if len(self.calib_table) == 0:
                msg = f"Could not find any FITS files in {self.config.calibrate.calib_directory} double-check config or set `calib_directory` to False"
                raise ValueError(msg)

        self.output_paths = []
        with mp.Pool(num_proc) as pool:
            jobs = []
            for group_key, group in input_table.groupby("GROUP_KEY"):
                output_path = get_reduced_path(self.paths, self.config, group_key)
                needs_run = (
                    force_process
                    or not output_path.exists()
                    or any_file_newer(group["path"], output_path)
                )
                if not needs_run:
                    logger.debug(f"Skipping processing for group {output_path}")
                    self.output_paths.append(output_path)
                else:
                    jobs.append(
                        pool.apply_async(
                            self.process_group,
                            args=(group, group_key, output_path),
                            kwds={"redo_stage": redo if force_process else None},
                        )
                    )

            for job in tqdm(jobs, desc="Processing files", leave=False):
                self.output_paths.append(job.get())

        self.output_paths.sort()

        logger.info("Creating table from output headers")
        self.output_table = header_table(self.output_paths, num_proc=num_proc, quiet=True)
        self.save_output_header()

        ## products — adi, diff, and pdi are independent leaves; skip when targeting another
        if self.config.save_adi_cubes and redo not in ("diff", "pdi"):
            self.save_adi_cubes(force=force_adi)

        if self.config.diff_images.make_diff and redo not in ("adi", "pdi"):
            self.make_diff_images(self.output_table, force=force_diff)

        logger.success("Finished processing files")

    def run_polarimetry(self, num_proc, redo: str | None = None):
        # pdi covers both MM computation and Stokes reduction;
        # MM→Stokes cascade is handled via file mtimes in make_stokes_image
        num_proc = _resolve_num_proc(num_proc)
        force_pdi = redo in ("pdi", "all")

        make_dirs(self.paths, self.config)
        conf_copy_path = self.paths.aux / f"{self.config.name}.bak.toml"
        self.config.save(conf_copy_path)
        logger.debug(f"Saved copy of config to {conf_copy_path}")

        if not self.output_table_path.exists():
            msg = f"Output table {self.output_table_path} cannot be found"
            raise RuntimeError(msg)

        working_table = pd.read_csv(self.output_table_path, index_col=0).sort_values("MJD")

        if self.config.polarimetry.mm_correct or self.config.polarimetry.method == "leastsq":
            working_table["mm_file"] = self.make_mueller_mats(
                working_table, num_proc=num_proc, force=force_pdi
            )

        logger.info("Performing polarimetric calibration")
        logger.debug(f"Saving Stokes data to {self.paths.pdi.absolute()}")
        match self.config.polarimetry.method:
            case "doublediff" | "triplediff":
                if self.config.nrm is None:
                    self.polarimetry_difference(
                        working_table,
                        method=self.config.polarimetry.method,
                        force=force_pdi,
                        num_proc=num_proc,
                    )
                else:
                    self.polarimetry_nrm(working_table, force=force_pdi, num_proc=num_proc)
            case "leastsq":
                self.polarimetry_leastsq(working_table, force=force_pdi, num_proc=num_proc)
        logger.success("Finished PDI")

    def create_input_table(self, filenames, num_proc) -> pd.DataFrame:
        logger.debug("Creating input header table")
        input_table = header_table(filenames, quiet=False, num_proc=num_proc).sort_values("MJD")
        table_path = self.paths.aux / f"{self.config.name}_input_headers.csv"
        input_table.to_csv(table_path)
        logger.info(f"Saved input header table to: {table_path}")
        return input_table

    def get_centroids(self):
        self.centroids = {}
        for key in ("cam1", "cam2"):
            path = self.paths.aux / f"{self.config.name}_centroids_{key}.toml"
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

    def get_uv_thetas(self):
        self.uv_thetas = {}
        for key in ("cam1", "cam2"):
            path = self.paths.aux / f"{self.config.name}_uv_theta_{key}.toml"
            if not path.exists():
                logger.warning(
                    f"Could not locate uv_theta file for {key}, expected it to be at {path}."
                )
                continue
            with path.open("rb") as fh:
                uv_thetas = tomli.load(fh)
            self.uv_thetas[key] = uv_thetas
        return self.uv_thetas

    def get_reproject_tforms(self):
        if not ("cam1" in self.centroids and "cam2" in self.centroids):
            self.reproject_tforms = None
            return self.reproject_tforms
        cam1_centroids = self.centroids["cam1"].copy()
        cam2_centroids = self.centroids["cam2"].copy()
        # flip cam1 on y!!
        for key in cam1_centroids:
            cam1_middle = intersect_point(cam1_centroids[key][:, 0], cam1_centroids[key][:, 1])
            cam1_offs = cam1_centroids[key] - cam1_middle
            cam1_offs[:, 1] *= -1
            cam1_centroids[key] = cam1_offs + cam1_middle

        # sort both sets by x-index
        self.reproject_tforms = {}
        cam1_offsets = recenter_centroids(cam1_centroids)
        cam2_offsets = recenter_centroids(cam2_centroids)
        for key in cam1_offsets:
            # fit similarity transform (scale + rotation + translation) from cam2 centroids to cam1 centroids
            tform = transform.SimilarityTransform()
            success = tform.estimate(cam2_offsets[key], cam1_offsets[key])
            assert success, (
                "Determining scale+rot transformation between cameras failed, check input centroids!"
            )
            # only save the rotation and scaling portions-- the translation will be handled during image registration
            self.reproject_tforms[key] = transform.SimilarityTransform(
                scale=tform.scale, rotation=tform.rotation
            )

        return self.reproject_tforms

    def make_synth_psfs(self, input_table):
        # make PSFs ahead of time so they don't overwhelm
        # during multiprocessing
        filters = {}
        for _, row in input_table.iterrows():
            for filt in determine_filterset_from_header(row):
                filters[filt] = row

        self.synth_psfs = {}
        for filt, row in filters.items():
            psf = create_synth_psf(
                row, filt, npix=self.config.analysis.window_size, output_directory=self.paths.aux
            )
            self.synth_psfs[filt] = psf

    def process_group(
        self, group, group_key: str, output_path: Path, redo_stage: str | None = None
    ):
        # Child process: file-only logging; main process owns stderr
        logger = configure_subprocesslogging_utils(self.workdir)

        force_all = redo_stage == "all"
        force_calibrate = force_all or redo_stage == "calibrate"
        force_combine = force_all or redo_stage == "combine"
        force_metrics = force_all or redo_stage == "metrics"
        force_select = force_all or redo_stage == "select"
        # specphot is bundled with align (they share aligned_path); rerunning either invalidates both.
        force_align = force_all or redo_stage in ("align", "specphot")
        # force_coadd: coadd always runs when process_group is called; the top-level skip
        # in run() is the only coadd checkpoint, so no per-stage force needed here.
        # dirty: once any stage reruns, all downstream stages must also rerun regardless
        # of whether intermediate files were saved (avoids stale-mtime false cache hits).
        dirty = False

        # ── Resolve intermediate checkpoint paths ──
        combined_path = (
            get_paths(output_path, suffix="comb", output_directory=self.paths.combined)[1]
            if self.config.combine.save_intermediate
            else None
        )
        metric_file = self.paths.metrics / f"{self.config.name}_{group_key}_metrics.npz"
        selected_path = (
            get_paths(output_path, output_directory=self.paths.selected)[1]
            if self.config.frame_select.save_intermediate
            else None
        )
        selected_metrics_path = (
            selected_path.with_suffix(".npz") if selected_path is not None else None
        )
        # when coadd is disabled the aligned file *is* the final output (see get_reduced_path)
        aligned_path = None
        if self.config.align.save_intermediate:
            if self.config.coadd.coadd:
                _, _ap = get_paths(output_path, output_directory=self.paths.aligned)
                aligned_path = _ap.with_name(_ap.name.replace("_coll", "_reg"))
            else:
                aligned_path = output_path

        # ── Stages 1+2: Calibrate + Combine ──
        if (
            not force_calibrate
            and not force_combine
            and combined_path is not None
            and combined_path.exists()
            and not any_file_newer(group["path"], combined_path)
        ):
            logger.debug(f"[{group_key}] loading combined data from cache")
            hdul = fits.open(combined_path)
        else:
            hdul_list = []
            for _, row in group.iterrows():
                logger.debug(f"Calibrating {row['path']}")
                cur_hdul = self.calibrate_one(row["path"], row, force=force_calibrate)
                hdul_list.append(cur_hdul)
            logger.debug(f"Combining {len(group)} calibrated files")
            hdul = combine_hduls(hdul_list)
            if combined_path is not None:
                hdul = add_timestamp_hdul(hdul)
                hdul = sort_headers_hdul(hdul)
                hdul.writeto(combined_path, overwrite=True)
                logger.debug(f"Saved combined HDU list to {combined_path.absolute()}")
            dirty = True

        # ── Stage 3: Metrics ──
        metrics_ref = (
            combined_path
            if (combined_path is not None and combined_path.exists())
            else group["path"].tolist()
        )
        if (
            not force_metrics
            and not dirty
            and metric_file.exists()
            and not any_file_newer(metrics_ref, metric_file)
        ):
            logger.debug(f"[{group_key}] loading metrics from cache")
            metrics = np.load(metric_file)
        else:
            metrics = self.analyze_one(
                hdul, metric_file, source_paths=group["path"].tolist(), force=True
            )
            dirty = True

        # ── Stage 4: Frame Select ──
        if self.config.frame_select.frame_select:
            if (
                not force_select
                and not dirty
                and selected_path is not None
                and selected_path.exists()
                and selected_metrics_path.exists()
                and not any_file_newer(metric_file, selected_path)
            ):
                logger.debug(f"[{group_key}] loading selected data from cache")
                hdul = fits.open(selected_path)
                metrics = np.load(selected_metrics_path)
            else:
                logger.debug(f"Frame selecting group {group_key}")
                hdul, metrics = frame_select_hdul(
                    hdul,
                    metrics,
                    metric=self.config.frame_select.metric,
                    quantile=self.config.frame_select.cutoff,
                )
                if selected_path is not None:
                    hdul = add_timestamp_hdul(hdul)
                    hdul = sort_headers_hdul(hdul)
                    hdul.writeto(selected_path, overwrite=True)
                    logger.debug(f"Saved selected HDU list to {selected_path.absolute()}")
                    np.savez_compressed(selected_metrics_path, metrics)
                    logger.debug(f"Saved selected metrics to {selected_metrics_path.absolute()}")
                dirty = True

        # ── Stage 5: Align + Specphot ──
        # Aligned intermediate (when it exists) includes specphot, so both are skipped together.
        align_input_refs = [
            p
            for p in (selected_path, metric_file, combined_path)
            if p is not None and Path(p).exists()
        ]
        if (
            not force_align
            and not dirty
            and aligned_path is not None
            and aligned_path.exists()
            and (not align_input_refs or not any_file_newer(align_input_refs, aligned_path))
        ):
            logger.debug(f"[{group_key}] loading aligned data from cache")
            hdul = fits.open(aligned_path)
        else:
            # note: register_hdul also handles MBI frame cropping even when align=False
            logger.debug(f"Aligning group {group_key}")
            reproject_tforms = self.reproject_tforms if self.config.align.reproject else None
            hdul = register_hdul(
                hdul,
                metrics,
                init_centroids=self.centroids.get(f"cam{hdul[0].header['U_CAMERA']:.0f}", None),
                align=self.config.align.align,
                pad=self.config.align.pad,
                method=self.config.align.method,
                crop_width=self.config.align.crop_width,
                reproject_tforms=reproject_tforms,
            )
            hdul = add_metrics_to_header(hdul, metrics)
            logger.debug(f"Running specphot calibration for group {group_key}")
            hdul = specphot_cal_hdul(hdul, config=self.config, metrics=metrics)
            if aligned_path is not None:
                hdul = add_timestamp_hdul(hdul)
                hdul = sort_headers_hdul(hdul)
                hdul.writeto(aligned_path, overwrite=True)
                logger.debug(f"Saved aligned HDU list to {aligned_path.absolute()}")
            dirty = True  # noqa: F841

        # ── Stage 6: Coadd ──
        if self.config.coadd.coadd:
            logger.debug(f"Coadding group {group_key}")
            _hdul = coadd_hdul(hdul, method=self.config.coadd.method)
            cam_num = int(hdul[0].header["U_CAMERA"])
            cam_key = f"cam{cam_num}"
            psfs = [
                self.synth_psfs[filt] for filt in determine_filterset_from_header(hdul[0].header)
            ]
            if self.config.coadd.recenter:
                logger.debug(f"Recentering group {group_key}")
                window_centers = self.centroids[cam_key]
                nbs_flag = hdul[0].header["MJD"] > NBS_INSTALL_MJD
                for key in window_centers:
                    for idx in range(window_centers[key].shape[0]):
                        window_centers[key][idx] = get_center(
                            hdul[0].data, window_centers[key][idx], cam_num, nbs_flag=nbs_flag
                        )
                _hdul = recenter_hdul(
                    _hdul, window_centers, method=self.config.coadd.recenter_method, psfs=psfs
                )

            # Undo the stage-5 specphot scaling so we can measure raw-unit metrics
            # for the next conv_factor pass. specphot_cal_hdul will re-derive and
            # re-apply, so the saved data ends up in BUNIT again.
            prior = read_prior_specphot_factors(_hdul)
            if prior is not None:
                undo = shape_factor_for_data(prior, _hdul[0].data)
                _hdul[0].data /= undo
                _hdul["ERR"].data /= undo
                for hdu in _hdul[2:]:
                    _hdul[0].header.pop(f"hierarch DPP SPECPHOT FACTOR {hdu.header['FIELD']}", None)

            cfg_a = self.config.analysis
            analyze_kwargs = dict(
                psfs=psfs,
                aper_rad=cfg_a.phot_aper_rad,
                ann_rad=cfg_a.phot_ann_rad or None,
                window_size=cfg_a.window_size,
                do_phot=cfg_a.photometry,
                do_strehl=cfg_a.strehl,
                do_psf_model=cfg_a.fit_psf_model,
                psf_model=cfg_a.psf_model,
            )
            logger.debug(f"Remeasuring coadd metrics for group {group_key}")
            coadd_metrics = analyze_coadded_hdul(_hdul, self.centroids[cam_key], **analyze_kwargs)
            logger.debug(f"Re-running specphot with coadd metrics for group {group_key}")
            _hdul = specphot_cal_hdul(_hdul, config=self.config, metrics=coadd_metrics)
            # Re-measure on the post-specphot cube so DPP headers carry BUNIT-scaled values.
            coadd_metrics = analyze_coadded_hdul(_hdul, self.centroids[cam_key], **analyze_kwargs)
            _hdul = add_coadd_metrics_to_header(_hdul, coadd_metrics)

            _hdul = sort_headers_hdul(_hdul)
            _hdul = add_timestamp_hdul(_hdul)
            logger.debug(f"Saving coadded output to {output_path.absolute()}")
            _hdul.writeto(output_path, overwrite=True)

        ## NRM analysis
        if self.config.nrm is not None:
            logger.debug(f"Starting NRM extraction for group {group_key}")
            subfolder = self.paths.nrm / "observables"
            subfolder.mkdir(parents=True, exist_ok=True)
            h5_path = subfolder / f"{self.config.name}_{group_key}_vis.h5"
            cam_num = int(hdul[0].header["U_CAMERA"])
            cam_key = f"cam{cam_num}"
            uv_thetas = self.uv_thetas[cam_key]
            h5_output_paths = extract_observables(
                config=self.config,
                input_hdul=hdul,
                output_path=h5_path,
                uv_thetas=uv_thetas,
                force=False,
            )
            for path in h5_output_paths:
                logger.debug(f"Saved observables to {path.absolute()}")
            logger.debug(f"Finished NRM extraction for group {group_key}")

        return output_path

    def get_coordinate(self):
        if self.config.target is None:
            self.coord = None
        else:
            self.coord = self.config.target.get_coord()

    def calibrate_one(self, path, fileinfo, force=False):
        logger.debug("Starting data calibration")
        config = self.config.calibrate
        if config.save_intermediate:
            outpath = get_paths(
                path, suffix="calib", filetype=".fits", output_directory=self.paths.calibrated
            )[1]
            if not force and outpath.exists() and not any_file_newer(path, outpath):
                return fits.open(outpath)

        back_filename = None
        flat_filename = None
        if self.calib_table is not None:
            calib_match = match_calib_file(path, self.calib_table)
            if config.back_subtract:
                back_filename = calib_match["backfile"]
            if config.flat_correct:
                flat_filename = calib_match["flatfile"]
        calib_hdul = calibrate_file(
            path,
            back_filename=back_filename,
            flat_filename=flat_filename,
            bpfix=config.fix_bad_pixels,
            coord=self.coord,
            force=force,
        )
        if config.save_intermediate:
            calib_hdul.writeto(outpath, overwrite=True)
            logger.debug(f"Calibrated data saved to {outpath}")
        logger.debug("Data calibration completed")
        return calib_hdul

    def analyze_one(self, hdul: fits.HDUList, metric_file, source_paths=None, force=False):
        logger.debug("Starting frame analysis")
        if (
            not force
            and metric_file.exists()
            and (source_paths is None or not any_file_newer(source_paths, metric_file))
        ):
            return np.load(metric_file)
        config = self.config.analysis
        hdr = hdul[0].header
        if not self.config.planetary:
            psfs = [self.synth_psfs[filt] for filt in determine_filterset_from_header(hdr)]
        else:
            psfs = None
        key = f"cam{hdr['U_CAMERA']:.0f}"
        outpath = analyze_file(
            hdul,
            centroids=self.centroids.get(key, None),
            window_size=config.window_size,
            aper_rad=config.phot_aper_rad,
            ann_rad=config.phot_ann_rad,
            psfs=psfs,
            do_phot=config.photometry,
            fit_psf_model=config.fit_psf_model,
            psf_model=config.psf_model,
            do_strehl=config.strehl,
            outpath=metric_file,
            force=force,
        )
        return np.load(outpath)

    def save_output_header(self):
        self.output_table.to_csv(self.output_table_path)
        logger.info(f"Saved output header table to {self.output_table_path}")
        return self.output_table_path

    def save_adi_cubes(self, force: bool = False):
        output_path = self.paths.adi / f"{self.config.name}_adi_cube.fits"
        angles_path = output_path.with_stem(output_path.stem.replace("_cube", "_angles"))
        if (
            not force
            and output_path.exists()
            and not any_file_newer(self.output_paths, output_path)
        ):
            if not angles_path.exists():
                group_keys = ["MJD", "U_FLC"]
                mask = self.output_table["U_FLC"].isna()
                self.output_table.loc[mask, "U_FLC"] = "NA"
                time_groups = self.output_table.sort_values(group_keys).groupby(group_keys)
                angs = [group["DEROTANG"].mean() for _, group in time_groups]
                fits.writeto(angles_path, np.array(angs, dtype="f4"), overwrite=True)
            return
        group_keys = ["MJD", "U_FLC"]
        mask = self.output_table["U_FLC"].isna()
        self.output_table.loc[mask, "U_FLC"] = "NA"
        time_groups = self.output_table.sort_values(group_keys).groupby(group_keys)
        cubes = []
        headers = []
        logger.info("Stacking output files into ADI cubes")
        time_groups = list(time_groups)
        for _key, group in tqdm(time_groups, desc="Stacking ADI frames", leave=False):
            hduls = [fits.open(path) for path in group["path"]]
            cube = np.mean([hdul[0].data for hdul in hduls], axis=0)
            cubes.append(cube)
            header = combine_frames_headers([hdul[0].header for hdul in hduls])
            headers.append(header)
        angs = np.array([hdr["DEROTANG"] for hdr in headers])
        # stacked_hdul = combine_hduls(hduls)
        prim_hdr = combine_frames_headers(headers)
        stacked_hdul = fits.PrimaryHDU(np.array(cubes), header=prim_hdr)
        stacked_hdul = add_timestamp_hdul(stacked_hdul)
        stacked_hdul = sort_headers_hdul(stacked_hdul)
        stacked_hdul.writeto(output_path, overwrite=True)
        logger.info(f"Saved ADI cube to {output_path}")
        fits.writeto(angles_path, np.array(angs, dtype="f4"), overwrite=True)
        # paths = []
        # for cam_num, group in cam_groups:
        #     cube_path = self.paths.adi / f"{self.config.name}_adi_cube_cam{cam_num:.0f}.fits"
        #     paths.append(cube_path)
        #     combine_frames_files(group["path"], output=cube_path, force=True, crop=False)
        #     logger.info(f"Saved cam {cam_num:.0f} ADI cube to {cube_path}")
        #     angles_path = cube_path.with_stem(f"{cube_path.stem}_angles")
        #     angles = np.asarray(group["DEROTANG"], dtype="f4")
        #     fits.writeto(angles_path, angles, overwrite=True)
        #     logger.info(f"Saved cam {cam_num:.0f} ADI angles to {angles_path}")

    def make_diff_images(self, table, num_proc=None, force=False):
        num_proc = _resolve_num_proc(num_proc)
        logger.info("Making difference frames")
        self.diff_files = []
        # do singlediff first, then deliberate to doublediff
        path_sets = get_singlediff_sets(table)
        diff_func = partial(singlediff_images, force=force)
        outdir = self.paths.diff / "single"
        outdir.mkdir(exist_ok=True)
        with mp.Pool(num_proc) as pool:
            jobs = []
            for i, paths in enumerate(path_sets):
                outpath = outdir / f"{self.config.name}_single_diff_{i:04d}.fits"
                jobs.append(pool.apply_async(diff_func, args=(paths,), kwds=dict(outpath=outpath)))
            for job in tqdm(jobs, desc="Making single-diff images", leave=False):
                self.diff_files.append(job.get())
        if self.config.diff_images.save_double:
            # now set for double-diff
            path_sets = get_doublediff_sets(table)
            diff_func = partial(doublediff_images, force=force)
            outdir = self.paths.diff / "double"
            outdir.mkdir(exist_ok=True)

            with mp.Pool(num_proc) as pool:
                jobs = []
                for i, paths in enumerate(path_sets):
                    outpath = outdir / f"{self.config.name}_double_diff_{i:04d}.fits"
                    jobs.append(
                        pool.apply_async(diff_func, args=(paths,), kwds=dict(outpath=outpath))
                    )
                for job in tqdm(jobs, desc="Making double-diff images", leave=False):
                    self.diff_files.append(job.get())
        logger.info("Done making difference frames")
        return self.diff_files

    def make_mueller_mats(self, table, num_proc=None, force=False):
        num_proc = _resolve_num_proc(num_proc)
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
                _, outpath = get_paths(row.path, suffix="mm", output_directory=self.paths.mm)
                jobs.append(
                    pool.apply_async(mueller_matrix_from_file, args=(row.path, outpath), kwds=kwds)
                )

            for job in tqdm(jobs, desc="Making Mueller matrices", leave=False):
                mm_paths.append(job.get())

        return mm_paths

    def polarimetry_difference(self, table, method, num_proc=None, force=False):
        num_proc = _resolve_num_proc(num_proc)
        config = self.config.polarimetry
        stokes_sets_path = self.paths.pdi / f"{self.config.name}_stokes_sets.csv"
        if stokes_sets_path.exists():
            stokes_sets = pd.read_csv(stokes_sets_path)
            logger.info(f"Loaded HWP cycle combinations from {stokes_sets_path}")
        else:
            match method.lower():
                case "triplediff":
                    stokes_sets = get_triplediff_set(table)
                case "doublediff":
                    stokes_sets = get_doublediff_set(table)
                case _:
                    msg = f"Invalid polarimetric difference method '{method}'"
                    raise ValueError(msg)
            stokes_sets.to_csv(stokes_sets_path, index=False)
            logger.info(f"Saved HWP cycle combinations to {stokes_sets_path}")

        stokes_data = []
        stokes_err = []
        prim_hdrs = []
        stokes_hdrs = []
        stokes_func = partial(
            make_stokes_image,
            method=method,
            coadded=self.config.coadd.coadd,
            derotate=config.derotate,
            mm_correct=config.mm_correct,
            hwp_adi_sync=config.hwp_adi_sync,
            ip_correct=config.ip_correct,
            ip_method=config.ip_method,
            ip_radius=config.ip_radius,
            ip_radius2=config.ip_radius2,
            coronagraphic=self.config.coronagraphic,
            pol_aper_rad=self.config.analysis.phot_aper_rad,
            pol_ann_rad=self.config.analysis.phot_ann_rad,
            mask_satspots=config.mask_satspots,
            force=force,
        )
        # TODO this is kind of ugly
        with mp.Pool(num_proc) as pool:
            jobs = []
            for set_idx, group in stokes_sets.query("STOKES_IDX != -1").groupby("STOKES_IDX"):
                paths = group["path"]
                outpath = self.paths.stokes / f"{self.config.name}_stokes_{set_idx:03d}.fits"
                if config.mm_correct:
                    mask = [p in paths.values for p in table["path"]]
                    subset = table.loc[mask]
                    mm_paths = subset["mm_file"]
                else:
                    mm_paths = None
                if len(paths) != (16 if method == "triplediff" else 8):
                    continue
                jobs.append(pool.apply_async(stokes_func, args=(paths, outpath, mm_paths)))

            for job in tqdm(jobs, desc="Creating Stokes images", leave=False):
                outpath = job.get()
                # use memmap=False to avoid "too many files open" effects
                # another way would be to set ulimit -n <MAX_FILES>
                with fits.open(outpath, memmap=False) as hdul:
                    stokes_data.append(hdul[0].data)
                    stokes_err.append(hdul["ERR"].data)
                    prim_hdrs.append(hdul[0].header)
                    hdrs = [hdul[i].header for i in range(2, len(hdul))]
                    stokes_hdrs.append(hdrs)

        ## Save CSV of Stokes values
        stokes_tbl = pd.DataFrame(
            [dict_from_header(hdr, fix=False) for hdr in prim_hdrs]
        ).sort_values("MJD")
        stokes_tbl_path = self.paths.pdi / f"{self.config.name}_stokes_table.csv"
        stokes_tbl.to_csv(stokes_tbl_path, index=False)
        logger.info(f"Saved table of Stokes file headers to {stokes_tbl_path}")
        ## Collapse outputs
        logger.info(f"Collapsing {stokes_sets['STOKES_IDX'].max() + 1} Stokes files...")
        stokes_data = np.array(stokes_data)
        stokes_err = np.array(stokes_err)
        coll_frame, _ = collapse_frames(np.nan_to_num(stokes_data))
        footprint = np.mean(np.isfinite(stokes_data).astype("f4"), axis=0)
        fits.writeto(
            self.paths.pdi / f"{self.config.name}_footprint.fits", footprint, overwrite=True
        )
        # coll_frame *= footprint
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coll_err = np.sqrt(np.nansum(stokes_err**2, axis=0)) / stokes_err.shape[0]
        nfields = coll_frame.shape[0]
        coll_hdrs = []
        for i in range(nfields):
            hdrs = [hdr[i] for hdr in stokes_hdrs]
            hdr = apply_wcs(coll_frame, combine_frames_headers(hdrs), angle=0)
            coll_hdrs.append(hdr)

        # correct TINT to account for actual number of files used
        tints = [fits.getval(path, "TINT") for path in np.unique(stokes_sets["path"])]
        tint = np.sum(tints)
        for hdr in coll_hdrs:
            hdr["NCOADD"] = len(tints)
            hdr["TINT"] = tint
        # optionally optimize the Qphi/Uphi offset angle per wavelength by rotating Q/U directly
        # so that all downstream products (Qphi, Uphi, LP_I, AoLP) stay self-consistent
        if config.optimize_uphi:
            offsets = optimize_uphi_offsets(
                coll_frame,
                method=config.uphi_method,
                radius=config.uphi_radius,
                radius2=config.uphi_radius2,
                max_angle=config.uphi_max_angle,
            )
            for hdr, offset, frame, frame_err in zip(
                coll_hdrs, offsets, coll_frame, coll_err, strict=True
            ):
                frame[:] = rotate_stokes(frame, -offset)
                frame_err[:] = rotate_stokes(frame_err, -offset)
                field = hdr["FIELD"]
                hdr[f"hierarch DPP PDI UPHI_OFF {field}"] = (
                    offset,
                    "[deg] Uphi-optimized AoLP offset angle",
                )
                logger.info(f"Optimized AoLP offset angle for {field}: {offset:.02f}°")
        prim_hdr = apply_wcs(coll_frame, combine_frames_headers(coll_hdrs), angle=0)
        prim_hdr["NCOADD"] /= len(coll_hdrs)
        prim_hdr["TINT"] /= len(coll_hdrs)
        prim_hdu = fits.PrimaryHDU(coll_frame, header=prim_hdr)
        err_hdu = fits.ImageHDU(coll_err, header=prim_hdr, name="ERR")
        hdul = fits.HDUList([prim_hdu, err_hdu])
        hdul.extend([fits.ImageHDU(header=hdr, name=hdr["FIELD"]) for hdr in coll_hdrs])
        # In the case we have multi-wavelength data, save 4D Stokes cube
        hdul = add_timestamp_hdul(hdul)
        hdul = sort_headers_hdul(hdul)
        if nfields > 1:
            stokes_cube_path = self.paths.pdi / f"{self.config.name}_stokes_cube.fits"
            write_stokes_products(
                hdul, outname=stokes_cube_path, force=True, planetary=config.cyl_stokes == "radial"
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
            prim_hdu = fits.PrimaryHDU(wave_coll_frame[:4], header=wave_coll_hdr)
            err_hdu = fits.ImageHDU(wave_err_frame[:4], header=wave_coll_hdr, name="ERR")
            dummy_hdu = fits.ImageHDU(header=wave_coll_hdr, name="COMB")
            hdul = fits.HDUList([prim_hdu, err_hdu, dummy_hdu])
            hdul = add_timestamp_hdul(hdul)
            hdul = sort_headers_hdul(hdul)
        # save single-wavelength (or wavelength-collapsed) Stokes cube
        stokes_coll_path = self.paths.pdi / f"{self.config.name}_stokes_coll.fits"
        write_stokes_products(
            hdul, outname=stokes_coll_path, force=True, planetary=config.cyl_stokes == "radial"
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

    def polarimetry_nrm(self, table, force: bool = False, num_proc=None):
        num_proc = _resolve_num_proc(num_proc)
        subfolder = self.paths.nrm / "observables"
        subfolder.mkdir(parents=True, exist_ok=True)
        path_list = []
        for idx, row in table.iterrows():
            _output_name = self.output_paths[idx].name
            if "reg.fits" in _output_name:
                _output_path = _output_name.replace("_reg.fits", "_vis.h5")
            else:
                _output_path = _output_name.replace("_coll.fits", "_vis.h5")

            h5_path = subfolder / _output_path
            h5_real_paths = get_nrm_paths(h5_path, row)
            path_list.append(h5_real_paths)
        table["nrm_paths"] = path_list

        outpath = self.paths.nrm / f"{self.config.name}_nrm_results.npz"
        if not force and outpath.exists():
            logger.info(f"Loading final NRM observables from {outpath.absolute()}")
            result_dict = np.load(outpath)
        else:
            result_dict = process_nrm_polarimetry(table, nbootstrap=self.config.nrm.nbootstrap)
            np.savez(outpath, **result_dict)
            logger.info(f"Saved final NRM observables to {outpath.absolute()}")
        make_nrm_plots(result_dict, self.paths.nrm, self.config.name)


def recenter_centroids(centroids: dict) -> dict:
    output = {}
    for key, value in centroids.items():
        arr = np.array(value)
        offs = arr - intersect_point(arr[:, 0], arr[:, 1])
        sorted_inds = np.argsort(offs[:, 0], axis=0)
        output[key] = offs[sorted_inds]
    return output
