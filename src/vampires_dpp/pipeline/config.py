from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

from serde import field, serialize
from tqdm.auto import tqdm

import vampires_dpp as vpp
from vampires_dpp.constants import SATSPOT_ANGLE


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
    """Geometric distortion correction options

    .. admonition:: Advanced Usage
        :class: warning

        Distortion correction requires specialized calibration files. Please get in contact with the SCExAO team for more details

    Parameters
    ----------
    transform_filename: Path
        The path to a CSV with the distortion corrections for each camera.
    """

    transform_filename: Path

    def __post_init__(self):
        self.transform_filename = Path(self.transform_filename)


@serialize
@dataclass
class CalibrateOptions(OutputDirectory):
    """Options for general image calibration

    The calibration strategy is generally

    #. Subtract dark frame, if provided
    #. Normalize by flat field, if provided
    #. Bad pixel correction, if set
    #. Flip camera 1 data along y-axis
    #. Apply distortion correction, if provided
    #. Deinterleave FLC states, if set

    .. admonition:: Deinterleaving
        :class: warning

        Deinterleaving should not be required for data downloaded from STARS. Only set this option if you are dealing with PDI data downloaded directly from the VAMPIRES computer.

    .. admonition:: Outputs

        If `deinterleave` is set, two files will be saved in the output directory for each input file, with `_FLC1` and `_FLC2` appended. The calibrated files will be saved with the "_calib" suffix.

    Parameters
    ----------
    master_darks: Optional[Dict[str, Optional[Path]]]
        If provided, must be a dict with keys for "cam1" and "cam2" master darks. You can omit one of the cameras. By default None.
    master_flats: Optional[Dict[str, Optional[Path]]]
        If provided, must be a dict with keys for "cam1" and "cam2" master flats. You can omit one of the cameras. By default None.
    distortion: Optional[DistortionOptions]
        (Advanced) Options for geometric distortion correction. By default None.
    fix_bad_pixels: bool
        If true, will run LACosmic algorithm for one iteration on each frame and correct bad pixels. By default false.
    deinterleave: bool
        **(Advanced)** If true, will deinterleave every other file into the two FLC states. By default false.
    output_directory : Optional[Path]
        The calibrated files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

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
    """Coronagraph options

    The IWAs for the masks are listed on the `VAMPIRES website <https://www.naoj.org/Projects/SCEXAO/scexaoWEB/030openuse.web/040vampires.web/100vampcoronagraph.web/indexm.html>`_.

    Parameters
    ----------
    iwa : float
        Coronagraph inner working angle (IWA) in mas.
    """

    iwa: float


@serialize
@dataclass(frozen=True)
class SatspotOptions:
    f"""Satellite spot options

    Parameters
    ----------
    radius : float
        Satellite spot radius in lambda/D, by default 15.9. If doing PDI with CHARIS this should be 11.2.
    angle : float
        Satellite spot position angle (in degrees), by default {SATSPOT_ANGLE:.01f}.
    amp : float
        Satellite spot modulation amplitude (in nm), by default 50.
    """
    radius: float = field(default=15.9)
    angle: float = field(default=SATSPOT_ANGLE)
    amp: float = field(default=50)


@serialize
@dataclass
class FrameSelectOptions(OutputDirectory):
    """Frame selection options

    Frame selection metrics can be measured on the central PSF, or can be done on calibration speckles (satellite spots). Satellite spots will be used if the `satspots` option is set in the pipeline. The quality metrics are

    * normvar - The variance normalized by the mean.
    * l2norm - The L2-norm, roughly equivalent to the RMS value
    * peak - maximum value

    Parameters
    ----------
    cutoff : float
        The cutoff quantile for frame selection where 0 means no frame selection and 1 would discard all frames.
    metric : str
        The frame selection metric, one of `"peak"`, `"l2norm"`, and `"normvar"`, by default `"normvar"`.
    window_size : int
        The window size (in pixels) to cut out around PSFs before measuring the frame selection metric, by default 30.
    output_directory : Optional[Path]
        The trimmed files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

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
    """Image registration options

    Image registration can be done on the central PSF, or can be done on calibration speckles (satellite spots). Satellite spots will be used if the `satspots` option is set in the pipeline. The registration methods are

    * com - centroid
    * peak - pixel at highest value
    * dft - Cross-correlation between frames using discrete Fourier transform (DFT) upsampling for subpixel accuracy
    * gaussian - Model fit using a Gaussian PSF
    * airydisk - Model fit using a Moffat PSF
    * moffat - Model fit using an Airy disk PSF

    .. admonition:: Outputs

        For each input file, a CSV with PSF centroids (or centroids for each satellite spot) will be saved in the output directory with the "_offsets" suffix and a registered cube will be saved with the "_aligned" suffix.

    Parameters
    ----------
    method : str
        The image registration method, one of `"com"`, `"peak"`, `"dft"`, `"airydisk"`, `"moffat"`, or `"gaussian"`. By default `"com"`.
    window_size : int
        The window size (in pixels) to cut out around PSFs before measuring the centroid, by default 30.
    smooth : bool
        If true, will Gaussian smooth the input frames before measuring offsets, by default false.
    dft_factor : int
        If using the DFT method, the upsampling factor (inverse of centroid precision), by default 1.
    output_directory : Optional[Path]
        The PSF offsets and aligned files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

    method: str = field(default="com")
    window_size: int = field(default=30, skip_if_default=True)
    smooth: Optional[int] = field(default=None, skip_if_default=True)
    dft_factor: int = field(default=1, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("com", "peak", "dft", "airydisk", "moffat", "gaussian"):
            raise ValueError(f"Registration method not recognized: {self.method}")


@serialize
@dataclass
class CollapseOptions(OutputDirectory):
    """
    Cube collapse options

    * median - Pixel-by-pixel median
    * mean - Pixel-by-pixel mean
    * varmean - Pixel-by-pixel mean weighted by frame variance
    * biweight - Pixel-by-pixel biweight location

    .. admonition:: Outputs

        For each input file, a collapsed frame will be saved in the output directory with the "_collapsed" suffix.


    Parameters
    ----------
    method : str
        The collapse method, one of `"median"`, `"mean"`, `"varmean"`, or `"biweight"`. By default `"median"`.
    output_directory : Optional[Path]
        The collapsed files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

    method: str = field(default="median", skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("median", "mean", "varmean", "biweight"):
            raise ValueError(f"Collapse method not recognized: {self.method}")


@serialize
@dataclass
class IPOptions:
    """Instrumental polarization (IP) correction options.

    There are three main IP correction techniques

    * Ad-hoc correction using PSF photometry
        In each diff image, the partial polarization of the central PSF is measured and removed, presuming there should be no polarized stellar signal. In coronagraphic data, this uses the light penetrating the partially transmissive focal plane masks (~0.06%).
    * Ad-hoc correction using PSF photometry of calibration speckles (satellite spots)
        Same as above, but using the average correction for the four satellite spot PSFs instead of the central PSF.
    * Mueller-matrix model correction (not currently implemented)
        Uses a calibrated Mueller-matrix model which accurately reflects the impacts of all polarizing optics in VAMPIRES. WIP.

    .. admonition:: Outputs

        For each diff image a copy will be saved with the IP correction applied and the "_ip" file suffix attached.


    Parameters
    ----------
    method : str
        IP correction method, one of `"photometry"`, `"satspots"`, or `"mueller"`. By default, `"photometry"`
    aper_rad : float
        For photometric-based methods, the aperture radius in pixels. By default, 6.
    force : bool
        If true, will force this processing step to occur.
    """

    method: str = "photometry"
    aper_rad: float = 6
    force: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        if self.method not in ("photometry", "satspots", "mueller"):
            raise ValueError(f"Polarization calibration method not recognized: {self.method}")


@serialize
@dataclass
class PolarimetryOptions(OutputDirectory):
    """Polarimetric differential imaging (PDI) options

    .. admonition:: Warning: experimental
        :class: warning

        The polarimetric reduction in this pipeline is an active work-in-progress. Do not consider any outputs publication-ready without further vetting and consultation with the SCExAO team.

    PDI is processed after all of the individual file processing since it requires sorting the files into complete sets for the triple-differential calibration.

    .. admonition:: Outputs

        Diff images will be saved in the output directory. If IP options are set, the IP corrected frames will also be saved.

    Parameters
    ----------
    ip : Optional[IPOptions]
        Instrumental polarization (IP) correction options, by default None.
    N_per_hwp : int
        Number of cubes expected per HWP position, by default 1.
    order : str
        HWP iteration order, one of `"QQUU"` or `"QUQU"`. By default `"QQUU"`.
    output_directory : Optional[Path]
        The diff images will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

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
class ProductOptions(OutputDirectory):
    """The output products from the processing pipeline.

    .. admonition:: Header Table

        A table with the header information of all the input files will be saved to a CSV in the output directory.

    .. admonition:: ADI Outputs

        The ADI outputs will include a cube for each camera and the corresponding derotation angles. For ADI analysis, you can either interleave the two cubes into one cube with double the frames, add the two camera frames before post-processing, or add the two ADI residuals from each camera after post-processing.

    .. admonition:: PDI Outputs

        If `polarimetry` is set, PDI outputs will be constructed from the triple-differential method. This includes a cube with various Stokes quantities from each HWP cycle, and a derotated and collapsed cube of Stokes quantities. The Stokes quantities are listed in the "STOKES" header, and are

        #. Stokes I
        #. Stokes Q
        #. Stokes U
        #. Radial Stokes Qphi
        #. Radial Stokse Uphi
        #. Linear polarized intensity
        #. Angle of linear polarization

    Parameters
    ----------
    header_table : bool
        If true, saves a CSV with header information, by default true.
    adi_cubes : bool
        If true, saves ADI outputs
    pdi_cubes : bool
        If true, saves PDI triple-diff cubes
    output_directory : Optional[Path]
        The products will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force all products to be recreated step to occur.

    """

    header_table: bool = field(default=True, skip_if_default=True)
    adi_cubes: bool = field(default=True, skip_if_default=True)
    pdi_cubes: bool = field(default=True, skip_if_default=True)


## Define classes for entire pipelines now
@serialize
@dataclass
class PipelineOptions:
    """Data Processing Pipeline options

    The processing configuration is all done through this class, which can easily be converted to and from TOML. The options will set the processing steps in the pipeline. An important paradigm in the processing pipeline is skipping unnecessary operations. That means if a file already exists, the pipeline will only reprocess it if the `force` flag is set, which will reprocess all files for that step (and subsequent steps), or if the input file or files are newer. You can try this out by deleting one calibrated file from a processed output and re-running the pipeline.

    Parameters
    ----------
    name : str
        filename-friendly name used for outputs from this pipeline. For example "20230101_ABAur"
    target : Optional[str]
        `SIMBAD <https://simbad.cds.unistra.fr/simbad/>`_-friendly object name used for looking up precise coordinates. If not provided, will use coordinate from headers, by default None.
    frame_centers : Optional[Dict[str, Optional[List]]]
        Estimates of the star position in pixels (x, y) for each camera provided as a dict with "cam1" and "cam2" keys. If not provided, will use the geometric frame center, by default None. *Note: if you are estimating centers from raw data, keep in mind cam1 files are flipped on the y-axis, so any estimate needs to be flipped, too*.
    coronagraph : Optional[CoronagraphOptions]
        If provided, sets coronagraph-specific options and processing
    satspots : Optional[SatspotOptions]
        If provided, sets satellite-spot specific options and enable satellite spot processing for frame selection and image registration
    calibrate : Optional[CalibrateOptions]
        If set, provides options for basic image calibration
    frame_select : Optional[FrameSelectOptions]
        If set, provides options for frame selection
    register : Optional[RegisterOptions]
        If set, provides options for image registration
    collapse : Optional[CollapseOptions]
        If set, provides options for collapsing image cubes
    polarimetry : Optional[PolarimetryOptions]
        If set, provides options for polarimetric differential imaging (PDI)
    products : Optional[ProductOptions]
        If set, provides options for saving metadata, ADI, and PDI outputs.
    version : str
        The version of vampires_dpp that this configuration file is valid with. Typically not set by user.
    """

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
    products: Optional[ProductOptions] = field(default=None, skip_if_default=True)
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
            self.products = ProductOptions(**self.products)
