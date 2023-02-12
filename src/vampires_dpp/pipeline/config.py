from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

from serde import field, serialize
from tqdm.auto import tqdm

import vampires_dpp as vpp


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


@serialize
@dataclass
class CalibrateOptions(OutputDirectory):
    master_darks: Optional[CamFileInput] = field(default=CamFileInput())
    master_flats: Optional[CamFileInput] = field(default=CamFileInput())
    distortion: Optional[DistortionOptions] = field(default=None, skip_if_default=True)
    fix_bad_pixels: bool = field(default=False, skip_if_default=True)
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
class RegisterOptions(OutputDirectory):
    method: str = field(default="com")
    window_size: int = field(default=30, skip_if_default=True)
    median_smooth: Optional[int] = field(default=None, skip_if_default=True)
    dft_factor: int = field(default=1, skip_if_default=True)

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
    force: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        if self.method not in ("photometry", "satspots", "mueller"):
            raise ValueError(f"Polarization calibration method not recognized: {self.method}")


@serialize
@dataclass
class PolarimetryOptions(OutputDirectory):
    ip: Optional[IPOptions] = field(default=None, skip_if_default=True)
    N_per_hwp: int = field(default=1, skip_if_default=True)
    order: str = field(default="QQUU", skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.ip is not None and isinstance(self.ip, dict):
            self.ip = IPOptions(**self.ip)
        self.order = self.order.strip().upper()
        if self.order not in ("QQUU", "QUQU"):
            raise ValueError(f"HWP order not recognized: {self.order}")


@serialize
@dataclass
class CamCtrOption:
    cam1: Optional[List[float]] = field(default=None, skip_if_default=True)
    cam2: Optional[List[float]] = field(default=None, skip_if_default=True)


@serialize
@dataclass
class ProductsOptions(OutputDirectory):
    header_table: bool = field(default=True, skip_if_default=True)
    adi_cubes: bool = field(default=True, skip_if_default=True)
    pdi_cubes: bool = field(default=True, skip_if_default=True)


## Define classes for entire pipelines now
@serialize
@dataclass
class PipelineOptions:
    name: str
    target: Optional[str] = field(default=None, skip_if_default=True)
    frame_centers: Optional[CamCtrOption] = field(default=None, skip_if_default=True)
    coronagraph: Optional[CoronagraphOptions] = field(default=None, skip_if_default=True)
    satspots: Optional[SatspotOptions] = field(default=None, skip_if_default=True)
    calibrate: Optional[CalibrateOptions] = field(default=None, skip_if_default=True)
    frame_select: Optional[FrameSelectOptions] = field(default=None, skip_if_default=True)
    register: Optional[RegisterOptions] = field(default=None, skip_if_default=True)
    collapse: Optional[CollapseOptions] = field(default=None, skip_if_default=True)
    polarimetry: Optional[PolarimetryOptions] = field(default=None, skip_if_default=True)
    products: Optional[ProductsOptions] = field(default=None, skip_if_default=True)
    version: str = vpp.__version__

    def __post_init__(self):
        if self.coronagraph is not None and isinstance(self.coronagraph, dict):
            self.coronagraph = CoronagraphOptions(**self.coronagraph)
        if self.satspots is not None and isinstance(self.satspots, dict):
            self.satspots = SatspotOptions(**self.satspots)
        if self.frame_centers is not None and isinstance(self.frame_centers, dict):
            self.frame_centers = CamCtrOption(**self.frame_centers)
        if self.calibrate is not None and isinstance(self.calibrate, dict):
            self.calibrate = CalibrateOptions(**self.calibrate)
        if self.frame_select is not None and isinstance(self.frame_select, dict):
            self.frame_select = FrameSelectOptions(**self.frame_select)
        if self.register is not None and isinstance(self.register, dict):
            self.register = RegisterOptions(**self.register)
        if self.collapse is not None and isinstance(self.collapse, dict):
            self.collapse = CollapseOptions(**self.collapse)
        if self.polarimetry is not None and isinstance(self.polarimetry, dict):
            self.polarimetry = PolarimetryOptions(**self.polarimetry)
        if self.products is not None and isinstance(self.products, dict):
            self.products = ProductsOptions(**self.products)
