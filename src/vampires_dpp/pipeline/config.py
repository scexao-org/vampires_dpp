from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional

from astropy.io import fits
from serde import field, serialize
from tqdm.auto import tqdm

import vampires_dpp as vpp
from vampires_dpp.calibration import make_dark_file, make_flat_file
from vampires_dpp.image_processing import collapse_frames_files
from vampires_dpp.util import FileInfo, FileType


## Some base classes for repeated functionality
@serialize
@dataclass(kw_only=True)
class OutputDirectory:
    output_directory: Optional[Path] = field(default=None, skip_if_default=True)
    force: bool = field(default=False, skip_if_default=True)

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


@serialize
@dataclass(frozen=True)
class SatspotOptions:
    radius: float = field(default=15.9)
    angle: float = field(default=-4)
    amp: float = field(default=50)


@serialize
@dataclass
class FrameSelectOptions(OutputDirectory):
    cutoff: float
    metric: str = field(default="normvar")
    window_size: int = field(default=30, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.metric not in ("peak", "l2norm", "normvar"):
            raise ValueError(f"Frame selection metric not recognized: {self.metric}")
        if self.cutoff < 0 or self.cutoff > 1:
            raise ValueError(
                f"Must use a value between 0 and 1 for frame selection quantile (got {self.cutoff})"
            )


@serialize
@dataclass
class CoregisterOptions(OutputDirectory):
    method: str = field(default="com")
    unshapr: bool = field(default=False, skip_if_default=True)
    window_size: int = field(default=30, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("com", "peak", "dft", "airydisk", "moffat", "gaussian"):
            raise ValueError(f"Registration method not recognized: {self.method}")


@serialize
@dataclass
class CollapseOptions(OutputDirectory):
    method: str = field(default="median", skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("median", "mean", "varmean", "biweight"):
            raise ValueError(f"Collapse method not recognized: {self.method}")


@serialize
@dataclass
class IPOptions:
    method: str = "photometry"
    aper_rad: float = 6

    def __post_init__(self):
        if self.method not in ("photometry", "satspots", "mueller"):
            raise ValueError(f"Polarization calibration method not recognized: {self.method}")


@serialize
@dataclass
class PolarimetryOptions(OutputDirectory):
    ip: Optional[IPOptions] = field(default=None, skip_if_default=True)


@serialize
class CamCtrOption:
    cam1: Optional[List[float]] = field(default=None, skip_if_default=True)
    cam2: Optional[List[float]] = field(default=None, skip_if_default=True)


@serialize
@dataclass
class MasterDarkOptions(FileInput, OutputDirectory):
    collapse: str = field(default="median", skip_if_default=True)
    cam1: Optional[Path] = field(default=None, skip_if_default=True)
    cam2: Optional[Path] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.collapse not in ("median", "mean", "varmean", "biweight"):
            raise ValueError(f"Collapse method not recognized: {self.collapse}")
        if self.cam1 is not None:
            self.cam1 = Path(self.cam1)
        if self.cam2 is not None:
            self.cam2 = Path(self.cam2)


@serialize
@dataclass
class MasterFlatOptions(FileInput, OutputDirectory):
    collapse: str = field(default="median", skip_if_default=True)
    cam1: Optional[Path] = field(default=None, skip_if_default=True)
    cam2: Optional[Path] = field(default=None, skip_if_default=True)
    cam1_dark: Optional[Path] = field(default=None, skip_if_default=True)
    cam2_dark: Optional[Path] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.collapse not in ("median", "mean", "varmean", "biweight"):
            raise ValueError(f"Collapse method not recognized: {self.collapse}")
        if self.cam1 is not None:
            self.cam1 = Path(self.cam1)
        if self.cam2 is not None:
            self.cam2 = Path(self.cam2)
        if self.cam1_dark is not None:
            self.cam1_dark = Path(self.cam1_dark)
        if self.cam2_dark is not None:
            self.cam2_dark = Path(self.cam2_dark)


## Define classes for entire pipelines now
@serialize
@dataclass
class PipelineOptions(FileInput, OutputDirectory):
    name: str
    target: Optional[str] = field(default=None, skip_if_default=True)
    frame_centers: Optional[CamCtrOption] = field(default=None, skip_if_default=True)
    coronagraph: Optional[CoronagraphOptions] = field(default=None, skip_if_default=True)
    satspots: Optional[SatspotOptions] = field(default=None, skip_if_default=True)
    master_dark: Optional[MasterDarkOptions] = field(default=None, skip_if_default=True)
    master_flat: Optional[MasterFlatOptions] = field(default=None, skip_if_default=True)
    calibrate: Optional[CalibrateOptions] = field(default=None, skip_if_default=True)
    frame_select: Optional[FrameSelectOptions] = field(default=None, skip_if_default=True)
    coregister: Optional[CoregisterOptions] = field(default=None, skip_if_default=True)
    collapse: Optional[CollapseOptions] = field(default=None, skip_if_default=True)
    polarimetry: Optional[PolarimetryOptions] = field(default=None, skip_if_default=True)
    version: str = vpp.__version__

    def __post_init__(self):
        super().__post_init__()

        if self.coronagraph is not None and isinstance(self.coronagraph, dict):
            self.coronagraph = CoronagraphOptions(**self.coronagraph)
        if self.satspots is not None and isinstance(self.satspots, dict):
            self.satspots = SatspotOptions(**self.satspots)
        if self.frame_centers is not None and isinstance(self.frame_centers, dict):
            self.frame_centers = CamCtrOption(**self.frame_centers)
        if self.master_dark is not None and isinstance(self.master_dark, dict):
            self.master_dark = MasterDarkOptions(**self.master_dark)
        if self.master_flat is not None and isinstance(self.master_flat, dict):
            self.master_flat = MasterFlatOptions(**self.master_flat)
        if self.calibrate is not None and isinstance(self.calibrate, dict):
            self.calibrate = CalibrateOptions(**self.calibrate)
        if self.frame_select is not None and isinstance(self.frame_select, dict):
            self.frame_select = FrameSelectOptions(**self.frame_select)
        if self.coregister is not None and isinstance(self.coregister, dict):
            self.coregister = CoregisterOptions(**self.coregister)
        if self.collapse is not None and isinstance(self.collapse, dict):
            self.collapse = CollapseOptions(**self.collapse)
        if self.polarimetry is not None and isinstance(self.polarimetry, dict):
            self.polarimetry = PolarimetryOptions(**self.polarimetry)


@serialize
@dataclass
class MasterFlat(FileInput, OutputDirectory):
    cam1: PathLike | List[PathLike]
    cam2: Optional[PathLike | List[PathLike]]
    cam1_dark: Optional[PathLike]
    cam2_dark: Optional[PathLike]
    collapse: str = "median"
