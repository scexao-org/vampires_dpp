from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional

from astropy.io import fits
from serde import SerdeSkip, field, serialize
from tqdm.auto import tqdm

from vampires_dpp.calibration import make_dark_file, make_flat_file
from vampires_dpp.image_processing import collapse_frames_files
from vampires_dpp.util import FileInfo, FileType


## Some base classes for repeated functionality
@serialize
@dataclass(kw_only=True)
class OutputDirectory:
    output_directory: Optional[Path] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        if self.output_directory is not None:
            self.output_directory = Path(self.output_directory)


@serialize
@dataclass
class FileInput:

    filenames: Path | List[Path]

    def __post_init__(self):
        if isinstance(self.filenames, list):
            self.filenames = map(Path, self.filenames)
        else:
            self.filenames = Path(self.filenames)

    def process(self):
        if isinstance(self.filenames, Path):
            if self.filenames.is_file():
                # is a file with a list of filenames
                with self.filenames.open("r") as fh:
                    self.filenames = (Path(f.strip()) for f in fh.readlines())
            else:
                # is a globbing expression
                paths = self.filenames.parent.glob(pattern=self.filenames.name)
        else:
            paths = self.filenames
        # only accept FITS files as inputs
        self.paths = list(filter(lambda p: ".fit" in p.name, paths))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"Could not find any FITS files from the following expression '{self.filenames}'"
            )

        # split into cam1 and cam2
        self.file_infos = []
        self.cam1_paths = []
        self.cam2_paths = []
        for path in self.paths:
            file_info = FileInfo.from_file(path)
            self.file_infos.append(file_info)
            if file_info.camera == 1:
                self.cam1_paths.append(path)
            else:
                self.cam2_paths.append(path)


@serialize
@dataclass
class CamFileInput:
    cam1: Optional[Path] = field(default=None, skip_if_default=True)
    cam2: Optional[Path] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        if self.cam1 is not None:
            self.cam1 = Path(self.cam1)

        if self.cam2 is not None:
            self.cam2 = Path(self.cam2)


## Define classes for each configuration block
@serialize
@dataclass
class DistortionOptions:
    transform_filename: Path

    def __post_init__(self):
        self.transform_filename = Path(self.transform_filename)


def cam_dict_factory():
    return


@serialize
@dataclass
class CalibrateOptions(OutputDirectory):
    master_darks: Optional[CamFileInput] = field(default=CamFileInput())
    master_flats: Optional[CamFileInput] = field(default=CamFileInput())
    distortion: Optional[DistortionOptions] = field(default=None, skip_if_default=True)
    deinterleave: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.master_darks is not None and isinstance(self.master_darks, dict):
            self.master_darks = CamFileInput(**self.master_darks)
        if self.master_flats is not None and isinstance(self.master_flats, dict):
            self.master_flats = CamFileInput(**self.master_flats)
        if self.distortion is not None and isinstance(self.distortion, dict):
            self.distortion = DistortionOptions(**self.distortion)


@serialize
@dataclass(frozen=True)
class CoronagraphOptions:
    iwa: float
    satspot_radius: float = field(default=15.9)
    satspot_angle: float = field(default=-4)


## Define classes for each config block
@serialize
@dataclass
class MasterDark(FileInput, OutputDirectory):
    collapse: str = "median"
    force: bool = False

    def process(self):
        super().process()

        # make darks for each camera
        with Pool(self.num_proc) as pool:
            jobs = []
            for path in self.paths:
                kwds = dict(
                    output_directory=self.output_directory, force=self.force, method=self.collapse
                )
                jobs.append(pool.apply_async(make_dark_file, args=(path,), kwds=kwds))

            self.cam1_darks = []
            self.cam2_darks = []
            for job in tqdm(jobs, desc="Collapsing dark frames"):
                filename = job.get()
                file_info = FileInfo.read(filename)
                if file_info.camera == 1:
                    self.cam1_darks.append(filename)
                else:
                    self.cam2_darks.append(filename)

        self.master_darks = {1: None, 2: None}
        if len(self.cam1_darks) > 0:
            self.master_darks[1] = self.output_directory / f"master_dark_cam1.fits"
            collapse_frames_files(
                self.cam1_darks, method=self.collapse, output=self.master_darks[1], force=self.force
            )
        if len(self.cam2_darks) > 0:
            self.master_darks[2] = self.output_directory / f"master_dark_cam2.fits"
            collapse_frames_files(
                self.cam2_darks, method=self.collapse, output=self.master_darks[2], force=self.force
            )


@serialize
@dataclass
class MasterFlat(FileInput, OutputDirectory):
    cam1: PathLike | List[PathLike]
    cam2: Optional[PathLike | List[PathLike]]
    cam1_dark: Optional[PathLike]
    cam2_dark: Optional[PathLike]
    collapse: str = "median"

    def process(self):
        super().process()
        # make darks for each camera
        with Pool(self.num_proc) as pool:
            jobs = []
            for file_info, path in zip(self.file_infos, self.paths):
                if file_info.camera == 1:
                    dark = self.cam1_dark
                else:
                    dark = self.cam2_dark
                kwds = dict(
                    output_directory=self.output_directory,
                    dark_filename=dark,
                    force=self.force,
                    method=self.collapse,
                )

                jobs.append(pool.apply_async(make_flat_file, args=(path,), kwds=kwds))

            self.cam1_flats = []
            self.cam2_flats = []
            for job in tqdm(jobs, desc="Collapsing dark frames"):
                filename = job.get()
                file_info = FileInfo.read(filename)
                if file_info.camera == 1:
                    self.cam1_flats.append(filename)
                else:
                    self.cam2_flats.append(filename)

        self.master_flats = {1: None, 2: None}
        if len(self.cam1_flats) > 0:
            self.master_flats[1] = self.output_directory / f"master_dark_cam1.fits"
            collapse_frames_files(
                self.cam1_flats, method=self.collapse, output=self.master_flats[1], force=self.force
            )
        if len(self.cam2_flats) > 0:
            self.master_flats[2] = self.output_directory / f"master_dark_cam2.fits"
            collapse_frames_files(
                self.cam2_flats, method=self.collapse, output=self.master_flats[2], force=self.force
            )
