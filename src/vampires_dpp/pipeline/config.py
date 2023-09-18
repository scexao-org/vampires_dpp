from os import PathLike
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Optional

import astropy.units as u
import tomli
import tomli_w
from annotated_types import Interval
from astropy.coordinates import Angle, SkyCoord
from pydantic import BaseModel, Field

import vampires_dpp as dpp


## Some base classes for repeated functionality
class CamFileInput(BaseModel):
    cam1: Optional[Path] = None
    cam2: Optional[Path] = None


class CoordinateConfig(BaseModel):
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
    pm_ra: float = 0
    pm_dec: float = 0
    frame: str = "icrs"
    obstime: str = "J2016"

    def model_post_init(self):
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


class CalibrateConfig(BaseModel):
    """Config for general image calibration

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
    distortion: Optional[DistortionConfig]
        (Advanced) Config for geometric distortion correction. By default None.
    fix_bad_pixels: bool
        If true, will run LACosmic algorithm for one iteration on each frame and correct bad pixels. By default false.
    deinterleave: bool
        **(Advanced)** If true, will deinterleave every other file into the two FLC states. By default false.
    output_directory : Optional[Path]
        The calibrated files will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> master_backs = {"cam1": "master_back_cam1.fits", "cam2": "master_back_cam2.fits"}
    >>> dist = {"transform_filename": "20230102_fcs16000_params.csv"}
    >>> conf = CalibrateConfig(
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

    master_backgrounds: Optional[CamFileInput] = CamFileInput()
    master_flats: Optional[CamFileInput] = CamFileInput()
    distortion_file: Optional[Path] = None
    fix_bad_pixels: bool = False
    deinterleave: bool = False
    save_intermediate: bool = False
    output_directory: ClassVar[Path] = Path("calib")

    def get_output_path(self, filename: Path):
        # replace any .fits.fz with .fits
        name = filename.name.split(".fits")[0]
        # take input filename and append '_calib'
        return self.output_directory / f"{name}_calib.fits"


class FrameSelectConfig(BaseModel):
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
    >>> conf = FrameSelectConfig(cutoff=0.7, output_directory="selected")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [frame_select]
        output_directory = "selected"
        cutoff = 0.7
        metric = "normvar"
    """

    cutoff: Annotated[float, Interval(ge=0, le=1)] = 0
    metric: Literal["peak", "l2norm", "normvar"] = "normvar"
    output_directory: ClassVar[Path] = Path("metrics")

    def get_output_path(self, filename: Path):
        # replace any .fits.fz with .fits
        name = filename.name.split(".fits")[0]
        # take input filename and append '_metrics'
        return self.output_directory / f"{name}_metrics.npz"


class RegisterConfig(BaseModel):
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
    >>> conf = RegisterConfig(method="com", output_directory="registered")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [register]
        output_directory = "registered"
        method = "com"
    """

    method: Literal["centroid", "peak", "dft", "psf"] = "centroid"
    window_size: int = 30
    smooth: bool = False
    smooth_kernel: int = 3
    dft_factor: int = 5
    dft_ref: Literal["centroid", "peak"] | Path = "centroid"
    output_directory: ClassVar[Path] = Path("offsets")

    def get_output_path(self, filename: Path):
        # replace any .fits.fz with .fits
        name = filename.name.split(".fits")[0]
        # take input filename and append '_metrics'
        return self.output_directory / f"{name}_offsets.npz"


class CollapseConfig(BaseModel):
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
    >>> conf = CollapseConfig(output_directory="collapsed")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [collapse]
        output_directory = "collapsed"
    """

    method: Literal["median", "mean", "varmean", "biweight"] = "median"


class AnalysisConfig(BaseModel):
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
    >>> conf = AnalysisConfig()
    >>> print(conf.to_toml())

    .. code-block:: toml

        [analysis]
        output_directory = "analysis"
        recenter = true
        subtract_radprof = true
        strehl = false
    """

    fit_model: bool = True
    model: Literal["gaussian", "moffat", "airydisk"] = "gaussian"
    strehl: bool = True
    subtract_radprof: bool = False
    window_size: int = 40
    photometry: bool = True
    aper_rad: float | Literal["auto"] = 10
    ann_rad: Optional[tuple[float, float]] = field(default=None, skip_if_default=True)

    # def __post_init__(self):
    #     if not isinstance(self.aper_rad, str) and self.aper_rad > self.window_size / 2:
    #         raise ValueError(f"Photometric radius ({self.aper_rad}) must be smaller than the window half-length ({self.window_size // 2})")
    #     if self.ann_rad is not None and self.ann_rad[1] > self.window_size / 2:
    #         raise ValueError(f"Photometric annulus radius ({self.ann_rad[1]}) must be smaller than the window half-length ({self.window_size // 2})")


class PolarimetryConfig(BaseModel):
    """Polarimetric differential imaging (PDI) options

    .. admonition:: Warning: experimental
        :class: warning

        The polarimetric reduction in this pipeline is an active work-in-progress. Do not consider any outputs publication-ready without further vetting and consultation with the SCExAO team.

    PDI is processed after all of the individual file processing since it requires sorting the files into complete sets for the triple-differential calibration.

    .. admonition:: Outputs

        If using the `difference` method, a cube of Stokes frames for each HWP set and a derotated and collapsed Stokes frame will be saved in the products directory with the suffixes `_stokes_cube` and `_stokes_cube_collapsed`, respectively.

        If using the `leastsq` method, a Stokes frame will be saved in the products directory with the suffix `_stokes_frame`.

    Parameters
    ----------
    method: Optional[str]
        Determines the polarization calibration method, either the double/triple-difference method (`difference`) or using the inverse least-squares solution from Mueller calculus (`leastsq`). In both cases, the Mueller matrix calibration is performed, but for the difference method data are organized into distinct HWP sets. This can result in data being discarded, however it is much easier to remove effects from e.g., satellite spots because you can median collapse the data from each HWP set, whereas for the inverse least-squares the data is effectively collapsed with a mean.
    mm_correct : bool
        Apply Mueller-matrix correction (only applicable to data reduced using the `"difference"` method). By default, True.
    ip : Optional[IPConfig]
        Instrumental polarization (IP) correction options, by default None.
    order : str
        HWP iteration order, one of `"QQUU"` or `"QUQU"`. By default `"QQUU"`.
    adi_sync : bool
        If true, will assume the HWP is in pupil-tracking mode. By default, True.
    output_directory : Optional[Path]
        The diff images will be saved to the output directory. If not provided, will use the current working directory. By default None.
    force : bool
        If true, will force this processing step to occur.

    Examples
    --------
    >>> conf = PolarimetryConfig(ip=IPConfig(), output_directory="pdi")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [polarimetry]
        output_directory = "pdi"

        [polarimetry.ip]
        method = "photometry"
        aper_rad = 6
    """

    method: Literal["difference", "leastsq"] = "difference"
    order: Literal["QQUU", "QUQU"] = "QQUU"
    mm_correct: bool = True
    mm_method: Literal["calibrated", "ideal"] = "calibrated"
    hwp_adi_sync: bool = True
    ip_correct: bool = True
    ip_method: Literal["aperture", "annulus", "satspots"] = "aperture"
    ip_radius: float = 15
    output_directory: ClassVar[Path] = Path("pdi")


class PipelineConfig(BaseModel):
    """Data Processing Pipeline options

    The processing configuration is all done through this class, which can easily be converted to and from TOML. The options will set the processing steps in the pipeline. An important paradigm in the processing pipeline is skipping unnecessary operations. That means if a file already exists, the pipeline will only reprocess it if the `force` flag is set, which will reprocess all files for that step (and subsequent steps), or if the input file or files are newer. You can try this out by deleting one calibrated file from a processed output and re-running the pipeline.

    Parameters
    ----------
    name : str
        filename-friendly name used for outputs from this pipeline. For example "20230101_ABAur"
    coordinate : Optional[CoordinateConfig]
    frame_centers : Optional[dict[str, Optional[list]]]
        Estimates of the star position in pixels (x, y) for each camera provided as a dict with "cam1" and "cam2" keys. If not provided, will use the geometric frame center, by default None.
    coronagraph : Optional[CoronagraphConfig]
        If provided, sets coronagraph-specific options and processing
    satspots : Optional[SatspotConfig]
        If provided, sets satellite-spot specific options and enable satellite spot processing for frame selection and image registration
    calibrate : Optional[CalibrateConfig]
        If set, provides options for basic image calibration
    frame_select : Optional[FrameSelectConfig]
        If set, provides options for frame selection
    register : Optional[RegisterConfig]
        If set, provides options for image registration
    collapse : Optional[CollapseConfig]
        If set, provides options for collapsing image cubes
    analysis : Optional[AnalysisConfig]
        If set, provides options for PSF/flux analysis in collapsed data
    diff : Optional[DiffConfig]
        If set, provides options for creating difference images
    polarimetry : Optional[PolarimetryConfig]
        If set, provides options for polarimetric differential imaging (PDI)
    products : Optional[ProductConfig]
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
    >>> conf = PipelineConfig(
            name="test_config",
            coronagraph=dict(iwa=55),
            satspots=dict(radius=11.2),
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

        [satspots]
        radius = 11.2
        angle = 84.6
        amp = 50

        [calibrate]
        output_directory = "calibrated"

        [calibrate.master_backgrounds]

        [calibrate.master_flats]

        [collapse]
        output_directory = "collapsed"

        [polarimetry]
        output_directory = "pdi"

    """

    dpp_version: str = dpp.__version__
    name: str = ""
    fields: Optional[CamFileInput] = CamFileInput()
    calibrate: CalibrateConfig = CalibrateConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    coronagraphic: bool = False
    make_diff_images: bool = False
    save_adi_cubes: bool = True
    coordinate: Optional[CoordinateConfig] = None
    frame_select: Optional[FrameSelectConfig] = None
    register: Optional[RegisterConfig] = None
    collapse: Optional[CollapseConfig] = None
    polarimetry: Optional[PolarimetryConfig] = None

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
        return cls.model_validate(config)

    def to_toml(self):
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True)
        return tomli_w.dumps(model_dict)

    def save(self, filename: Path):
        """
        Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True)
        # save output TOML
        with Path(filename).open("wb") as fh:
            tomli_w.dump(model_dict, fh)
