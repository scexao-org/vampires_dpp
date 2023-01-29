from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import List, Optional

from astropy.io import fits
from serde import field, serde
from tqdm.auto import tqdm

from vampires_dpp.calibration import make_dark_file, make_flat_file
from vampires_dpp.image_processing import collapse_frames_files

## Some base classes for repeated functionality


@serde
@dataclass(kw_only=True)
class OutputDirectory:
    output_directory: Optional[Path] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        if self.output_directory is not None:
            self.output_directory = Path(self.output_directory)


@dataclass
class FileInput:

    filenames: str | List[str]

    def process(self):
        if isinstance(self.filenames, str):
            path = Path(self.filenames)
            if path.is_file():
                # is a file with a list of filenames
                with path.open("r") as fh:
                    paths = (Path(f.strip()) for f in fh.readlines())
            else:
                # is a globbing expression
                path = Path(self.filenames)
                paths = path.parent.glob(pattern=path.name)
        else:
            # is a list of filenames
            paths = map(Path, self.filenames)
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
            file_info = FileInfo.read(path)
            self.file_infos.append(file_info)
            if file_info.camera == 1:
                self.cam1_paths.append(path)
            else:
                self.cam2_paths.append(path)


@dataclass(kw_only=True)
class MultiProcessing:
    num_proc: Optional[int] = None

    def __post_init__(self):
        if self.num_proc is not None:
            # try and avoid over-saturating large systems
            self.num_proc = min(self.num_proc, 8)


class FileType(Enum):
    GEN2 = 0
    OG = 1


@dataclass(frozen=True)
class FileInfo:
    file_type: FileType
    camera: int

    def __post_init__(self):
        if not (self.camera == 1 or self.camera == 2):
            raise ValueError(f"Invalid camera number {self.camera}")

    @classmethod
    def read(cls, filename):
        with fits.open(filename) as hdus:
            hdu = hdus[0]
            if "U_OGFNAM" in hdu.header:
                filetype = FileType.GEN2
            else:
                filetype = FileType.OG
            camera = hdu.header["U_CAMERA"]
        return cls(filetype, camera)


## Define classes for each config block
@serde
@dataclass
class MasterDark(FileInput, OutputDirectory, MultiProcessing):
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


@serde
@dataclass
class MasterFlat(FileInput, OutputDirectory, MultiProcessing):
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


@dataclass
class DistortionOptions:
    transform: Path

    def __post_init__(self):
        self.transform = Path(self.transform)


@serde
@dataclass
class CalibrateOptions(FileInput, OutputDirectory):
    cam1_dark: Optional[Path] = None
    cam2_dark: Optional[Path] = None
    cam1_flat: Optional[Path] = None
    cam2_flat: Optional[Path] = None
    # distortion: Optional[DistortionCorrect] = field(default=None, skip_if_default=True)
    deinterleave: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.cam1_dark is not None:
            self.cam1_dark = Path(self.cam1_dark)
        if self.cam2_dark is not None:
            self.cam2_dark = Path(self.cam2_dark)
        if self.cam1_flat is not None:
            self.cam1_flat = Path(self.cam1_flat)
        if self.cam2_flat is not None:
            self.cam2_flat = Path(self.cam2_flat)
