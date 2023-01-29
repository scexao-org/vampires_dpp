from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import List, Optional

import tqdm.auto as tqdm
from astropy.io import fits
from serde import field, serde


class CollapseMethods(Enum):
    median = 1
    mean = 2
    varmean = 3
    biweight = 4


## Some base classes for repeated functionality


@serde
@dataclass(kw_only=True)
class OutputDirectory:
    """
    Output path, will be created if it does not exist. If None, will use root directory.
    """

    output_directory: Optional[PathLike] = field(default=None, skip_if_false=True)

    def __post_init__(self):
        if self.output_directory is not None:
            self.output_directory = Path(self.output_directory)

    def process(self):
        if self.output_directory is None:
            return
        if not self.output_directory.is_dir():
            self.output_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class FileInput:

    filenames: PathLike | List[PathLike]

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
        self.cam1_paths = []
        self.cam2_paths = []
        for path in self.paths:
            file_info = FileInfo.read(path)
            match file_info.camera:
                case 1:
                    self.cam1_paths.append(path)
                case 2:
                    self.cam2_paths.append(path)


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
class MasterDark(FileInput, OutputDirectory):
    collapse: str = "median"
    force: bool = False

    def process(self):
        super(self, FileInput).process()
        super(self, OutputDirectory).process()

        # make darks for each camera
        self.cam1_darks = []
        for path in tqdm(self.cam1_paths, desc="Making cam 1 master dark"):
            outname = self.output_directory / f"{path.stem}_collapsed"
            make_dark_file(path, output=outname, force=self.force)

        self.cam2_darks = []
        for path in tqdm(self.cam2_paths, desc="Making cam 2 master dark"):
            pass

    def make_dark_frame(filename):
        self.master_darks[key] = outdir / f"{self.config['name']}_master_dark_{key}.fits"
        collapse_frames_files(
            dark_frames, method="mean", output=self.master_darks[key], skip=skip_darks
        )


@serde
@dataclass
class MasterFlat(FileInput, OutputDirectory):
    cam1: PathLike | List[PathLike]
    cam2: Optional[PathLike | List[PathLike]]
    cam1_dark: Optional[PathLike]
    cam2_dark: Optional[PathLike]
    collapse: CollapseMethods = "median"

    def process(self):
        pass


## Define the main config
@serde
@dataclass
class VAMPIRESConfig(FileInput, OutputDirectory):
    root_directory: PathLike
    configs: List
    name: str
    target: Optional[str] = None
