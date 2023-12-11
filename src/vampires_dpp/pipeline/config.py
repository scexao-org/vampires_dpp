from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal

import astropy.units as u
import tomli
import tomli_w
from annotated_types import Interval
from astropy.coordinates import Angle, SkyCoord
from pydantic import BaseModel

import vampires_dpp as dpp
from vampires_dpp.util import check_version

__all__ = (
    "ObjectConfig",
    "SpecphotConfig",
    "CalibrateConfig",
    "CollapseConfig",
    "AnalysisConfig",
    "PolarimetryConfig",
    "PipelineConfig",
)


class ObjectConfig(BaseModel):
    """Astronomical coordinate options.

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

    @property
    def ra_ang(self):
        ra_ang = Angle(self.ra, "hour")
        return ra_ang

    @property
    def dec_ang(self):
        dec_ang = Angle(self.dec, "deg")
        return dec_ang

    def get_coord(self) -> SkyCoord:
        """
        Return SkyCoord from the current parameters
        """
        return SkyCoord(
            ra=self.ra_ang,
            dec=self.dec_ang,
            pm_ra_cosdec=self.pm_ra * u.mas / u.year,
            pm_dec=self.pm_dec * u.mas / u.year,
            distance=1e3 * u.pc / self.parallax,
            frame=self.frame,
            obstime=self.obstime,
        )


class SpecphotConfig(BaseModel):
    """Spectrophotometric Configuration

    Spectrophotometric calibration requires determining the precise conversion from detector data numbers ($adu/s$) to astronomical flux ($Jy$).
    We enable this through synthetic photometry of calibrated spectra. The synthetic photometry is accomplished with `synphot <https://synphot.readthedocs.io/en/latest/>`_.
    We offer two input types for the stellar spectrum-

    1. Calibrated spectrum data
        * Requires an absolutely calibrated spectrum
        * Data must be prepared such that calling `SourceSpectrum.from_file <https://synphot.readthedocs.io/en/latest/api/synphot.spectrum.SourceSpectrum.html#synphot.spectrum.SourceSpectrum.from_file>`_ loads the spectrum. Refer to their documentation for format information.
        * Set `source` as the path
    2. Stellar Model Library
        * Uses `pickles uvk <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/pickles-atlas>`_ model library
        * Spectral type and reference magnitude required for normalizing model

    The synthetic photometry is used to determine the expected flux in Jy, which is used to determine the conversion factor. This conversion factor maps from data
    flux in adu/s to Jy, with units of Jy s/adu. To determine the factor we take the flux metric from each collapsed frame (and average between any satellite spots) before
    dividing by the frame exposure time (``header["EXPTIME"]``). We store the conversion factor and other derivatives, such as the Vega zero-point magnitude in the FITS
    header of the output file. Lastly, we convert the input data pixel-by-pixel from adu to Jy using the conversion factor and integration time. We lastly convert to surface
    brightness by dividing each pixel by its solid angle (``header["PXSCALE"]^2``).

    .. admonition:: Note: combining data

        Because each camera's data is calibrated independently, when you combine cam1 and cam2  data (such as PDI, ADI post-processing) you should *average* the two
        cameras' data to maintain accurate spectrophotometric calibration.

    Parameters
    ----------
    source:
        Spectrum source type. If a path, must be a file that can be loaded by `synphot.SourceSpectrum.from_file`.
    sptype:
        Only used if `source` is "pickles". Stellar spectral type. Note: must be one of the spectral types available in the pickles model atlas. Refer to the STScI documentation for more information on available spectral types.
    mag:
        Only used if `source` is "pickles". Stellar reference magnitude
    mag_band:
        Only used if `source` is "pickles". Stellar reference magnitude band
    flux_metric:
        Which frame analysis statistic to use for determining flux. "photometry" uses an aperture sum, while "sum" uses the sum in the analysis cutout window.
    """

    source: Literal["pickles"] | Path = "pickles"
    sptype: str | None = None
    mag: float | None = None
    mag_band: Literal["U", "B", "V", "r", "i", "J", "H", "K"] | None = "V"
    flux_metric: Literal["photometry", "sum"] = "photometry"


class CalibrateConfig(BaseModel):
    """Config for general image calibration.

    The calibration strategy is generally

    #. Load data and fix header values
    #. Calculate precise coordinates if ``ObjectConfig`` is used in pipeline
    #. Background subtraction
    #. (Optional) flat-field normalization
    #. (Optional) bad pixel correction
    #. Flip camera 1 data along y-axis

    We use a file-matching approach for calibrations to try and flexibly use the calibration data you have, even if it's not the ideal calibration file
    or is from a different night. The file matching will always require the calibrations to have the same pixel crop (both size and location) and detector
    read mode. For background files, we'll try and find files with the same exposure time and detector gain, but will accept others. Flat files will try and
    match detector gain, filter, and exposure time, in that order. For all files, if there are multiple matches we will select the single file closest in time.

    Parameters
    ----------
    calib_directory:
        Path to calibration file directory, if not provided no calibration will be done, regardless of other settings.
    back_subtract:
        If true will look for background files in ``calib_directory`` and subtract them if found. If not found, will subtract detector bias value.
    flat_correct:
        If true will look for flat files in ``calib_directory`` and perform flat normalization if found.
    fix_bad_pixels:
        If true, will run adaptive sigma-clipping algorithm for one iteration on each frame and correct bad pixels. By default false.
    save_intermediate:
        If true, will save intermediate calibrated data to ``calibrated/`` folder.

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

    .. admonition:: Outputs

        If the ``save_intermediate`` pipeline option is true, calibrated data will be saved in the ``calibrated`` folder (``paths.calib_dir``).

    """

    calib_directory: Path | None = None
    back_subtract: bool = True
    flat_correct: bool = True
    distortion_file: Path | None = None
    fix_bad_pixels: bool = False
    save_intermediate: bool = False


class CollapseConfig(BaseModel):
    """Cube collapse options

    * median - Pixel-by-pixel median
    * mean - Pixel-by-pixel mean
    * varmean - Pixel-by-pixel mean weighted by frame variance
    * biweight - Pixel-by-pixel biweight location

    .. admonition:: Outputs

        For each input file, a collapsed frame will be saved in the output directory with the "_collapsed" suffix.


    Parameters
    ----------
    method : str
        The collapse method, one of `"median"`, `"mean"`, `"varmean"`, or `"biweight"`. By default
        `"median"`.


    Examples
    --------
    >>> conf = CollapseConfig(output_directory="collapsed")
    >>> print(conf.to_toml())

    .. code-block:: toml

        [collapse]
        output_directory = "collapsed"
    """

    method: Literal["median", "mean", "varmean", "biweight"] = "median"
    frame_select: Literal["max", "l2norm", "normvar"] | None = "normvar"
    centroid: Literal["com", "peak", "gauss", "dft"] | None = "com"
    select_cutoff: Annotated[float, Interval(ge=0, le=1)] = 0
    recenter: Literal["com", "peak", "gauss", "dft"] | None = "com"
    reproject: bool = True


class AnalysisConfig(BaseModel):
    """PSF modeling and analysis options.

    .. admonition:: Outputs


    Parameters
    ----------
    model
    strehl
    recenter
    subtract_radprof

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

    strehl: Literal[False] = False
    subtract_radprof: bool = False
    aper_rad: float | Literal["auto"] = 8
    ann_rad: Sequence[float] | None = None
    window_size: int = 30
    dft_factor: int = 100


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
    hwp_adi_sync : bool
        If true, will assume the HWP is in pupil-tracking mode. By default, True.

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

    method: Literal["triplediff", "doublediff", "leastsq"] = "triplediff"
    mm_correct: bool = True
    hwp_adi_sync: bool = True
    use_ideal_mm: bool = False
    ip_correct: bool = True
    ip_method: Literal["aperture", "annulus"] = "aperture"
    ip_radius: float = 15
    ip_radius2: float | None = None


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

    name: str = ""
    diff_images: Literal["doublediff", "singlediff", "both"] | None = None
    save_adi_cubes: bool = True
    coronagraphic: bool = False
    dpp_version: str = dpp.__version__
    object: ObjectConfig | None = None
    calibrate: CalibrateConfig = CalibrateConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    specphot: SpecphotConfig | None = None
    collapse: CollapseConfig = CollapseConfig()
    polarimetry: PolarimetryConfig | None = None

    @classmethod
    def from_file(cls, filename: PathLike):
        """Load configuration from TOML file

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
        with Path(filename).open("rb") as fh:
            config = tomli.load(fh)
        if not check_version(config["dpp_version"], dpp.__version__):
            msg = f"Input pipeline version ({config['dpp_version']}) is not compatible \
                    with installed version of `vampires_dpp` ({dpp.__version__}). Try running \
                    `dpp upgrade {config}`."
            raise ValueError(msg)
        return cls.model_validate(config)

    def to_toml(self):
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True, mode="json", round_trip=True)
        return tomli_w.dumps(model_dict)

    def save(self, filename: Path):
        """Save configuration settings to TOML file

        Parameters
        ----------
        filename : PathLike
            Output filename
        """
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True, mode="json", round_trip=True)
        # save output TOML
        with Path(filename).open("wb") as fh:
            tomli_w.dump(model_dict, fh)
