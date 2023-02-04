from multiprocessing import Pool
from pathlib import Path

from tqdm.auto import tqdm

from vampires_dpp.calibration import make_dark_file, make_flat_file
from vampires_dpp.cli.dpp import subparser
from vampires_dpp.image_processing import collapse_frames_files
from vampires_dpp.pipeline.config import FileInput, MasterDarkOptions, MasterFlatOptions


def make_master_dark(args):
    # prepare input filenames
    config = MasterDarkOptions(
        filenames=args.filenames,
        collapse=args.collapse,
        cam1=f"{args.name}_cam1.fits",
        cam2=f"{args.name}_cam2.fits",
        output_directory=args.directory,
        force=args.force,
    )
    config.process()
    # make darks for each camera
    if config.output_directory is not None:
        outdir = config.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()
    with Pool(args.num_proc) as pool:
        jobs = []
        for file_info, path in zip(config.file_infos, config.paths):
            kwds = dict(
                output_directory=outdir,
                force=args.force,
                method=args.collapse,
            )
            job = pool.apply_async(make_dark_file, args=(path,), kwds=kwds)
            jobs.append((file_info.camera, job))

        cam1_darks = []
        cam2_darks = []
        for cam, job in tqdm(jobs, desc="Collapsing dark frames"):
            filename = job.get()
            if cam == 1:
                cam1_darks.append(filename)
            else:
                cam2_darks.append(filename)

    if len(cam1_darks) > 0:
        collapse_frames_files(
            cam1_darks, method=config.collapse, outname=config.cam1, force=config.force
        )
    if len(cam2_darks) > 0:
        collapse_frames_files(
            cam1_darks, method=config.collapse, outname=config.cam2, force=config.force
        )


dark_parser = subparser.add_parser("make_dark")
dark_parser.add_argument("filenames", nargs="+", help="Input FITS files")
dark_parser.add_argument(
    "-n", "--name", default="master_dark", help="File name base for generated data"
)
dark_parser.add_argument(
    "-c", "--collapse", default="median", choices=("median", "mean", "varmean")
)


def make_master_flat(args):
    dark_files = FileInput(args.darks)
    dark_files.process()
    # prepare input filenames
    config = MasterFlatOptions(
        filenames=args.filenames,
        collapse=args.collapse,
        cam1=f"{args.name}_cam1.fits",
        cam2=f"{args.name}_cam2.fits",
        cam1_dark=dark_files.cam1_paths[0],
        cam2_dark=dark_files.cam2_paths[0],
        output_directory=args.directory,
        force=args.force,
    )
    config.process()
    if config.output_directory is not None:
        outdir = config.output_directory
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path.cwd()
    with Pool(args.num_proc) as pool:
        jobs = []
        for file_info, path in zip(config.file_infos, config.paths):
            if file_info.camera == 1:
                dark = config.cam1_dark
            else:
                dark = config.cam2_dark
            kwds = dict(
                output_directory=outdir,
                dark_filename=dark,
                force=config.force,
                method=config.collapse,
            )
            job = pool.apply_async(make_flat_file, args=(path,), kwds=kwds)
            jobs.append((file_info.camera, job))

        cam1_flats = []
        cam2_flats = []
        for cam, job in tqdm(jobs, desc="Collapsing flat frames"):
            filename = job.get()
            if cam == 1:
                cam1_flats.append(filename)
            else:
                cam2_flats.append(filename)

    if len(cam1_flats) > 0:
        self.master_flats[1] = outdir / f"master_flat_cam1.fits"
        collapse_frames_files(
            cam1_flats, method=config.collapse, output=self.master_flats[1], force=config.force
        )
    if len(cam2_flats) > 0:
        self.master_flats[2] = outdir / f"master_flat_cam2.fits"
        collapse_frames_files(
            cam2_flats, method=config.collapse, output=self.master_flats[2], force=config.force
        )
