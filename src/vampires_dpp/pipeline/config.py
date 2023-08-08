from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Optional

import astropy.units as u
import tomli
from astropy.coordinates import Angle, SkyCoord
from serde import field, serialize
from serde.toml import to_toml

import vampires_dpp as dpp
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

    def to_toml(self) -> str:
        return to_toml(self)


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

    def to_toml(self) -> str:
        return to_toml(self)


@serialize
@dataclass
class CoordinateOptions:
    """Astronomical coordinate options

    .. admonition:: Tip: GAIA
        :class: Tip

        This can be auto-generated wtih GAIA coordinate information through the command line ``dpp new`` interface.

    Parameters
    ----------
    object: str
        SIMBAD-friendly object name
    ra: str
        Right ascension in sexagesimal hour angles
    dec: str
        Declination in sexagesimal degrees
    parallax: float
        parallax of system in mas
    pm_ra: float
        Proper motion of RA axis in mas/yr, by default 0.
    pm_dec: float
        Proper motion of DEC axis in mas/yr, by default 0.
    frame: str
        Coordinate reference frame, by default "icrs".
    obstime: str
        Observation time as a string, by default "J2016" (to coincide with GAIA coordinates)
    """

    object: str
    ra: str
    dec: str
    parallax: float
    pm_ra: float = field(default=0)
    pm_dec: float = field(default=0)
    frame: str = field(default="icrs", skip_if_default=True)
    obstime: str = field(default="J2016", skip_if_default=True)

    def __post_init__(self):
        if isinstance(self.ra, str):
            self.ra_ang = Angle(self.ra, "hour")
        else:
            self.ra_ang = Angle(self.ra, "deg")
        self.ra = self.ra_ang.to_string(pad=True, sep=":")
        self.dec_ang = Angle(self.dec, "deg")
        self.dec = self.dec_ang.to_string(pad=True, sep=":")

    def to_toml(self) -> str:
        obj = {"coordinate": self}
        return to_toml(obj)

    def get_coord(self) -> SkyCoord:
        return SkyCoord(
            ra=self.ra_ang,
            dec=self.dec_ang,
            pm_ra_cosdec=self.pm_ra * u.mas / u.year,
            pm_dec=self.pm_dec * u.mas / u.year,
            distance=1e3 / self.parallax * u.pc,
            frame=self.frame,
            obstime=self.obstime,
        )


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

    def to_toml(self) -> str:
        obj = {"calibrate": {"distortion": self}}
        return to_toml(obj)


@serialize
@dataclass
class CalibrateOptions(OutputDirectory):
    """Options for general image calibration

    The calibration strategy is generally

    #. Subtract background frame, if provided
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
    master_backgrounds: Optional[dict[str, Optional[Path]]]
        If provided, must be a dict with keys for "cam1" and "cam2" master backgrounds. You can omit one of the cameras. By default None.
    master_flats: Optional[dict[str, Optional[Path]]]
        If provided, must be a dict with keys for "cam1" and "cam2" master flats. You can omit one of the cameras. By default None.
    distortion: Optional[DistortionOptions]
        (Advanced) Options for geometric distortion correction. By default None.
    fix_bad_pixels: bool
        If true, will run LACosmic algorithm for one iteration on each frame and correct bad pixels. By default false.
    output_directory : Optional[Path]
        The calibrated files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> master_backs = {"cam1": "master_back_cam1.fits", "cam2": "master_back_cam2.fits"}
    >>> dist = {"transform_filename": "20230102_fcs16000_params.csv"}
    >>> conf = CalibrateOptions(
            master_backgrounds=master_backs,
            distortion=dist,
            output_directory="calibrated"
        )
    >>> print(conf.to_toml())

    .. code-block:: toml

        [calibrate]
        output_directory = "calibrated"

        [calibrate.master_backgrounds]
        cam1 = "master_back_cam1.fits"
        cam2 = "master_back_cam2.fits"

        [calibrate.master_flats]

        [calibrate.distortion]
        transform_filename = "20230102_fcs16000_params.csv"
    """

    master_backgrounds: Optional[CamFileInput] = field(default=CamFileInput())
    master_flats: Optional[CamFileInput] = field(default=CamFileInput())
    distortion: Optional[DistortionOptions] = field(default=None, skip_if_default=True)
    fix_bad_pixels: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.master_backgrounds is not None and isinstance(self.master_backgrounds, dict):
            self.master_backgrounds = CamFileInput(**self.master_backgrounds)
        if self.master_flats is not None and isinstance(self.master_flats, dict):
            self.master_flats = CamFileInput(**self.master_flats)
        if self.distortion is not None and isinstance(self.distortion, dict):
            self.distortion = DistortionOptions(**self.distortion)

    def to_toml(self) -> str:
        obj = {"calibrate": self}
        return to_toml(obj)


@serialize
@dataclass
class FrameSelectOptions(OutputDirectory):
    """Frame selection options

    Frame selection metrics can be measured on the central PSF, or can be done on calibration speckles (satellite spots). Satellite spots will be used if the `satspots` option is set in the pipeline. The quality metrics are

    * normvar - The variance normalized by the mean.
    * l2norm - The L2-norm, roughly equivalent to the RMS value
    * peak - maximum value

    .. admonition:: Outputs

        For each input file, a CSV with frame selection metrics for each slice will be saved in the output directory with the "_metrics" suffix. If `save_intermediate` is true, a cube with bad frames discarded will be saved with the "_selected" suffix.

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
    save_intermediate : bool
        If true, will save the frame-selected FITS files in the output directory. WARNING this can lead to extremely large data volumes.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> conf = FrameSelectOptions(cutoff=0.7, output_directory="selected")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [frame_select]
        output_directory = "selected"
        cutoff = 0.7
        metric = "normvar"
    """

    cutoff: float
    metric: str = field(default="normvar")
    window_size: int = field(default=30, skip_if_default=True)
    save_intermediate: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.metric not in ("peak", "l2norm", "normvar"):
            raise ValueError(f"Frame selection metric not recognized: {self.metric}")
        if self.cutoff < 0 or self.cutoff > 1:
            raise ValueError(
                f"Must use a value between 0 and 1 for frame selection quantile (got {self.cutoff})"
            )

    def to_toml(self) -> str:
        obj = {"frame_select": self}
        return to_toml(obj)


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

        For each input file, a CSV with PSF centroids (or centroids for each satellite spot) will be saved in the output directory with the "_offsets" suffix. If `save_intermediate` is true, a registered cube will be saved with the "_aligned" suffix.

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
    dft_ref : str
        If using the DFT method, the reference method, one of "com", "peak".
    output_directory : Optional[Path]
        The PSF offsets and aligned files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    save_intermediate : bool
        If true, will save the registered FITS files in the output directory. WARNING this can lead to extremely large data volumes.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> conf = RegisterOptions(method="com", output_directory="registered")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [register]
        output_directory = "registered"
        method = "com"
    """

    method: str = field(default="com")
    window_size: int = field(default=30, skip_if_default=True)
    smooth: Optional[int] = field(default=None, skip_if_default=True)
    dft_factor: int = field(default=1, skip_if_default=True)
    dft_ref: str = field(default="com")
    save_intermediate: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("com", "peak", "dft", "airydisk", "moffat", "gaussian"):
            raise ValueError(f"Registration method not recognized: {self.method}")
        if self.dft_ref not in ("com", "peak"):
            raise ValueError(f"Registration method not recognized: {self.dft_ref}")

    def to_toml(self) -> str:
        obj = {"register": self}
        return to_toml(obj)


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


    Examples
    --------
    >>> conf = CollapseOptions(output_directory="collapsed")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [collapse]
        output_directory = "collapsed"
    """

    method: str = field(default="median", skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method not in ("median", "mean", "varmean", "biweight"):
            raise ValueError(f"Collapse method not recognized: {self.method}")

    def to_toml(self) -> str:
        obj = {"collapse": self}
        return to_toml(obj)


@serialize
@dataclass
class DiffOptions(OutputDirectory):
    """DIfference imaging options.

    Difference images are made by taking camera 1 - camera 2.

    .. admonition:: Outputs

        Each pair of camera 1 / camera 2 inputs will be subtracted and added. The difference image is stored in the first slice of the output file and the sum image is stored in the second slice.

    Parameters
    ----------
    output_directory : Optional[Path]
        The collapsed files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.
    """

    def to_toml(self) -> str:
        obj = {"diff": self}
        return to_toml(obj)


@serialize
@dataclass
class PhotometryOptions:
    """ """

    aper_rad: float = 10
    ann_rad: Optional[list[float, float]] = field(default=None, skip_if_default=True)

    def to_toml(self) -> str:
        obj = {"analysis": {"photometry": self}}
        return to_toml(obj)


@serialize
@dataclass
class AnalysisOptions(OutputDirectory):
    """PSF modeling and analysis options.

    .. admonition:: Outputs


    Parameters
    ----------
    model
    strehl
    recenter
    subtract_radprof
    output_directory : Optional[Path]
        The diff images will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> conf = AnalysisOptions()
    >>> print(conf.to_toml())

    .. code-block:: toml

        [analysis]
        output_directory = "analysis"
        recenter = true
        subtract_radprof = true
        strehl = false
    """

    model: str = "gaussian"
    strehl: bool = False
    recenter: bool = True
    subtract_radprof: bool = True
    window_size: int = field(default=30, skip_if_default=True)
    photometry: Optional[PhotometryOptions] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.model.strip().lower() not in ("gaussian", "moffat", "airydisk"):
            raise ValueError(f"PSF model not recognized: {self.model}")
        if self.strehl:
            raise NotImplementedError()
        if self.photometry is not None and isinstance(self.photometry, dict):
            self.photometry = PhotometryOptions(**self.photometry)

    def to_toml(self) -> str:
        obj = {"analysis": self}
        return to_toml(obj)


@serialize
@dataclass
class IPOptions:
    """Instrumental polarization (IP) correction options.

    There are three main IP correction techniques

    * Ad-hoc correction using PSF photometry
        In each diff image, the partial polarization of the central PSF is measured and removed, presuming there should be no polarized stellar signal. In coronagraphic data, this uses the light penetrating the partially transmissive focal plane masks (~0.06%).
    * Ad-hoc correction using PSF photometry of calibration speckles (satellite spots)
        Same as above, but using the average correction for the four satellite spot PSFs instead of the central PSF.

    .. admonition:: Outputs

        The IP corrected versions of any Stokes cubes or collapsed frames will be saved with the additional suffix `_ipcorr`.

    Parameters
    ----------
    method : str
        IP correction method, one of `"photometry"`, `"satspots"`. By default, `"photometry"`
    aper_rad : float
        For photometric-based methods, the aperture radius in pixels. By default, 6.
    force : bool
        If true, will force this processing step to occur.
    """

    method: str = "photometry"
    aper_rad: float = 10
    force: bool = field(default=False, skip_if_default=True)

    def __post_init__(self):
        if self.method.lower() not in ("photometry", "satspots"):
            raise ValueError(f"Polarization IP correction method not recognized: {self.method}")

    def to_toml(self) -> str:
        obj = {"polarimetry": {"ip": self}}
        return to_toml(obj)


@serialize
@dataclass
class PolarimetryOptions(OutputDirectory):
    """Polarimetric differential imaging (PDI) options

    .. admonition:: Warning: experimental
        :class: warning

        The polarimetric reduction in this pipeline is an active work-in-progress. Do not consider any outputs publication-ready without further vetting and consultation with the SCExAO team.

    PDI is processed after all of the individual file processing since it requires sorting the files into complete sets for the triple-differential calibration.

    .. admonition:: Outputs

        If using the `difference` method, a cube of Stokes frames for each HWP set and a derotated and collapsed Stokes frame will be saved in the products directory with the suffixes `_stokes_cube` and `_stokes_cube_collapsed`, respectively.

        If using the `mueller` method, a Stokes frame will be saved in the products directory with the suffix `_stokes_frame`.

    Parameters
    ----------
    method: Optional[str]
        Determines the polarization calibration method, either the double/triple-difference method (`difference`) or using the inverse least-squares solution from Mueller calculus (`leastsq`). In both cases, the Mueller matrix calibration is performed, but for the difference method data are organized into distinct HWP sets. This can result in data being discarded, however it is much easier to remove effects from e.g., satellite spots because you can median collapse the data from each HWP set, whereas for the inverse least-squares the data is effectively collapsed with a mean.
    ip : Optional[IPOptions]
        Instrumental polarization (IP) correction options, by default None.
    order : str
        HWP iteration order, one of `"QQUU"` or `"QUQU"`. By default `"QQUU"`.
    output_directory : Optional[Path]
        The diff images will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> conf = PolarimetryOptions(ip=IPOptions(), output_directory="pdi")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [polarimetry]
        output_directory = "pdi"

        [polarimetry.ip]
        method = "photometry"
        aper_rad = 6
    """

    method: str = "difference"
    ip: Optional[IPOptions] = field(default=None, skip_if_default=True)
    N_per_hwp: int = field(default=1, skip_if_default=True)
    order: str = field(default="QQUU", skip_if_default=True)

    def __post_init__(self):
        super().__post_init__()
        if self.method.strip().lower() not in ("difference", "leastsq"):
            raise ValueError(f"Polarization calibration method not recognized: {self.method}")

        if self.ip is not None and isinstance(self.ip, dict):
            self.ip = IPOptions(**self.ip)
        self.order = self.order.strip().upper()
        if self.order not in ("QQUU", "QUQU"):
            raise ValueError(f"HWP order not recognized: {self.order}")

    def to_toml(self) -> str:
        obj = {"polarimetry": self}
        return to_toml(obj)


@serialize
@dataclass
class CamCtrOption:
    cam1: Optional[list[float]] = field(default=None, skip_if_default=True)
    cam2: Optional[list[float]] = field(default=None, skip_if_default=True)

    def __post_init__(self):
        if self.cam1 is not None:
            self.cam1 = list(self.cam1)
            if len(self.cam1) == 0:
                self.cam1 = None
        if self.cam2 is not None:
            self.cam2 = list(self.cam2)
            if len(self.cam2) == 0:
                self.cam2 = None


@serialize
@dataclass
class ProductOptions(OutputDirectory):
    """The output products from the processing pipeline.

    .. admonition:: Outputs

        **Header Table:**

        A table with the header information of all the input files will be saved to a CSV in the output directory.

        **ADI Outputs:**

        The ADI outputs will include a cube for each camera and the corresponding derotation angles. For ADI analysis, you can either interleave the two cubes into one cube with double the frames, add the two camera frames before post-processing, or add the two ADI residuals from each camera after post-processing.

        **PDI Outputs:**

        If `polarimetry` is set, PDI outputs will be constructed from the triple-differential or inverse least-squares methods. The Stokes images in each cube or frame are listed in the "STOKES" header, and are

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

    Examples
    --------
    >>> conf = ProductOptions(output_directory="products")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [products]
        output_directory = "products"
    """

    header_table: bool = field(default=True, skip_if_default=True)
    adi_cubes: bool = field(default=True, skip_if_default=True)
    pdi_cubes: bool = field(default=True, skip_if_default=True)

    def to_toml(self) -> str:
        obj = {"products": self}
        return to_toml(obj)


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
    coordinate : Optional[CoordinateOptions]
    frame_centers : Optional[dict[str, Optional[list]]]
        Estimates of the star position in pixels (x, y) for each camera provided as a dict with "cam1" and "cam2" keys. If not provided, will use the geometric frame center, by default None.
    use_satspots : bool
        If True, uses satellite spot-specific options and processing, by default False.
    calibrate : Optional[CalibrateOptions]
        If set, provides options for basic image calibration
    frame_select : Optional[FrameSelectOptions]
        If set, provides options for frame selection
    register : Optional[RegisterOptions]
        If set, provides options for image registration
    collapse : Optional[CollapseOptions]
        If set, provides options for collapsing image cubes
    analysis : Optional[AnalysisOptions]
        If set, provides options for PSF/flux analysis in collapsed data
    diff : Optional[DiffOptions]
        If set, provides options for creating difference images
    polarimetry : Optional[PolarimetryOptions]
        If set, provides options for polarimetric differential imaging (PDI)
    products : Optional[ProductOptions]
        If set, provides options for saving metadata, ADI, and PDI outputs.
    version : str
        The version of vampires_dpp that this configuration file is valid with. Typically not set by user.

    Notes
    -----
    **Frame Centers**

    Frame centers need to be given as a dictionary of x, y pairs, like

    .. code-block:: python

        frame_centers = {
            "cam1": (127.5, 127.5),
            "cam2": (127.5, 127.5)
        }
    It is important to note that these frame centers are in the *raw* frames. If you open up the frames in DS9 and set the cross on the image center, you can copy the x, y coordinates directly into the configuration. We recommend doing this, especially for coronagraphic data since the satellite spot cutout indices depend on the frame centers and any off-center data may not register appropriately.

    Examples
    --------
    >>> conf = PipelineOptions(
            name="test_config",
            use_satspots=True,
            calibrate=dict(output_directory="calibrated"),
            collapse=dict(output_directory="collapsed"),
            polarimetry=dict(output_directory="pdi"),
        )
    >>> print(conf.to_toml())

    .. code-block:: toml

        name = "test_config"
        version = "0.4.0"

        [coronagraph]
        iwa = 55

        [calibrate]
        output_directory = "calibrated"

        [calibrate.master_backgrounds]

        [calibrate.master_flats]

        [collapse]
        output_directory = "collapsed"

        [polarimetry]
        output_directory = "pdi"

    """

    name: str
    coordinate: Optional[CoordinateOptions] = field(default=None, skip_if_default=True)
    frame_centers: Optional[CamCtrOption] = field(default=None, skip_if_default=True)
    use_satspots: bool = field(default=False, skip_if_default=True)
    calibrate: Optional[CalibrateOptions] = field(default=None, skip_if_default=True)
    frame_select: Optional[FrameSelectOptions] = field(default=None, skip_if_default=True)
    register: Optional[RegisterOptions] = field(default=None, skip_if_default=True)
    collapse: Optional[CollapseOptions] = field(default=None, skip_if_default=True)
    analysis: Optional[AnalysisOptions] = field(default=None, skip_if_default=True)
    diff: Optional[DiffOptions] = field(default=None, skip_if_default=True)
    polarimetry: Optional[PolarimetryOptions] = field(default=None, skip_if_default=True)
    products: Optional[ProductOptions] = field(default=None, skip_if_default=True)
    version: str = dpp.__version__

    def __post_init__(self):
        if self.coordinate is not None and isinstance(self.coordinate, dict):
            self.coordinate = CoordinateOptions(**self.coordinate)
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
        if self.analysis is not None and isinstance(self.analysis, dict):
            self.analysis = AnalysisOptions(**self.analysis)
        if self.diff is not None and isinstance(self.diff, dict):
            self.diff = DiffOptions(**self.diff)
        if self.polarimetry is not None and isinstance(self.polarimetry, dict):
            self.polarimetry = PolarimetryOptions(**self.polarimetry)
        if self.products is not None and isinstance(self.products, dict):
            self.products = ProductOptions(**self.products)

    def to_toml(self) -> str:
        return to_toml(self)

    @classmethod
    def from_file(cls, filename: PathLike):
        """
        Load configuration from TOML file

        Parameters
        ----------
        filename : PathLike
            Path to TOML file with configuration settings.

        Raises
        ------
        ValueError
            If the configuration `version` is not compatible with the current `vampires_dpp` version.

        Examples
        --------
        >>> Pipeline.from_file("config.toml")
        """
        with open(filename, "rb") as fh:
            config = tomli.load(fh)
        return cls(**config)

    @classmethod
    def from_str(cls, toml_str: str):
        """
        Load configuration from TOML string.

        Parameters
        ----------
        toml_str : str
            String of TOML configuration settings.

        Raises
        ------
        ValueError
            If the configuration `version` is not compatible with the current `vampires_dpp` version.
        """
        config = tomli.loads(toml_str)
        return cls(**config)

    def to_file(self, filename: PathLike):
        """
        Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        # use serde.to_toml to serialize self
        path = Path(filename)
        # save output TOML
        toml_str = self.to_toml()
        with path.open("w") as fh:
            fh.write(toml_str)
