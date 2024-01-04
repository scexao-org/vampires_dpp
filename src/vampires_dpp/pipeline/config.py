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
from vampires_dpp.constants import SATSPOT_REF_ANGLE, SATSPOT_REF_NACT
from vampires_dpp.util import check_version

__all__ = (
    "TargetConfig",
    "SpecphotConfig",
    "CalibrateConfig",
    "CollapseConfig",
    "AnalysisConfig",
    "PolarimetryConfig",
    "PipelineConfig",
)


class TargetConfig(BaseModel):
    """Astronomical coordinate options.

    .. admonition:: Tip: GAIA
       :class: Tip

        This can be auto-generated wtih GAIA coordinate information through the command line ``dpp new`` interface.

    Parameters
    ----------
    name: str
        SIMBAD-friendly target name
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

    name: str
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
    #. Calculate precise coordinates if ``TargetConfig`` is used in pipeline
    #. Background subtraction
    #. (Optional) flat-field normalization
    #. (Optional) bad pixel correction
    #. Flip camera 1 data along y-axis

    We use a file-matching approach for calibrations to try and flexibly use the calibration data you have, even if it's not the ideal calibration file
    or is from a different night. The file matching will always require the calibrations to have the same pixel crop (both size and location) and detector
    read mode. For background files, we'll try and find files with the same exposure time and detector gain, but will accept others. Flat files will try and
    match detector gain, filter, and exposure time, in that order. For all files, if there are multiple matches we will select the single file closest in time.



        **File Outputs**

        - If ``save_intermediate`` is true, will save calibrated data to ``calibrated/``


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


        **File Outputs**

        - Each input file is collapsed and saved into the ``collapsed/`` folder


    Parameters
    ----------
    method:
        The collapse method, one of `"median"`, `"mean"`, `"varmean"`, or `"biweight"`. By default
        `"median"`.
    frame_select:
        If provided, will use the given metric to select frames for inclusions/exclusion from each data cube.
    centroid:
        If provided, will register each frame using the given method. If not provided for MBI data a very rough
        estimate of the center will be used to get the cutout crops for each field.
    select_cutoff:
        If ``frame_select`` is provided, this is the cutoff _quantile_ (from 0 to 1), where 0.2 means 20% of the frames
        from each cube will be discarded according the the selection metric.
    recenter:
        If a string is provided, will measure the centroids in the higher S/N collapsed frame using the given method.
    reproject:
        If true, will reproject cam2 data onto cam1's WCS coordinates
    """

    method: Literal["median", "mean", "varmean", "biweight"] = "median"
    frame_select: Literal["max", "l2norm", "normvar"] | None = "normvar"
    centroid: Literal["com", "peak", "gauss", "dft"] | None = "com"
    select_cutoff: Annotated[float, Interval(ge=0, le=1)] = 0
    recenter: Literal["com", "peak", "gauss", "dft"] | None = "com"
    reproject: bool = True
    satspot_reference: dict[str, float] = {
        "separation": SATSPOT_REF_NACT,
        "angle": SATSPOT_REF_ANGLE,
    }


class AnalysisConfig(BaseModel):
    """PSF modeling and analysis options.


        **File Outputs**

        - For each file an `NPZ <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html>_` file is created in ``metrics/``
            - Keys are metrics/centroids/statistics
            - Values are arrays with dimensions ``(nfields, npsfs, nframes)``


    Parameters
    ----------
    aper_rad:
        Aperture radius in pixels for circular aperture photometry. If "auto", will use the FWHM from the file header.
    ann_rad:
        If provided, will do local background-subtracted photometry with an annulus with the given inner and outer radius, in pixels.
    dft_factor:
        The DFT upsampling factor for measuring phase cross-correlation centroid.
    subtract_radprof:
        If true, will subtract a radial profile for measuring centroids and Strehl ratio, but no other metrics
    window_size:
        The cutout side length when getting cutouts for each PSF. Cutouts are centered around the file centroid estimate.
    """

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

    **File Outputs**

    - All PDI outputs are in ``pdi/``
    - Top-level products include
        - Collapsed stokes cubes (and wavelength-collapsed cubes for MBI data)
        - Header table for Stokes frames, if using a difference method.
    - If using Mueller-matrices (``method="leastsq"`` or ``mm_correct=True``) FITS file with matrices for each input file in ``pdi/mm/``
    - If using a difference method, will form individual Stokes frames and save in ``pdi/stokes/``

    Parameters
    ----------
    method:
        Determines the polarization calibration method, either the double/triple-difference method (`difference`) or using the inverse least-squares solution from Mueller calculus (`leastsq`). In both cases, the Mueller matrix calibration is performed, but for the difference method data are organized into distinct HWP sets. This can result in data being discarded, however it is much easier to remove effects from e.g., satellite spots because you can median collapse the data from each HWP set, whereas for the inverse least-squares the data is effectively collapsed with a mean.
    mm_correct:
        Apply Mueller-matrix correction (only applicable to data reduced using the `"difference"` method). By default, True.
    hwp_adi_sync:
        If true, will assume the HWP is in pupil-tracking mode. By default, True.
    use_ideal_mm:
        If true and doing Mueller-matrix correction (``mm_correct=True``) will use only idealized versions for the components in the
        Mueller-matrix model.
    ip_correct:
        If provided, will do post-hoc IP correction from the photometric sum in the given region.
    ip_method:
        If ``ip_correct=True`` this determines the region type for IP measurement.
    ip_radius:
        The first radius for IP correction, if set. For "aperture" this is the radius, for "annulus" this is the inner radius.
    ip_radius2:
        The second radius for IP correction. This is only used if ``ip_method="annulus"``- this is the outer radius.
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

        **File Outputs**

        - Auxilliary files in ``aux/``
            - Copy of config., centroid file, astrometry file, mean PSFs, filter curve(s), synth. PSF(s).
        - Data products (ADI cubes, output file header table) in ``products/``
        - Difference images in ``diff/``
            - ``diff/single/`` and ``diff/double/``

    Parameters
    ----------
    name:
        filename-friendly name used for outputs from this pipeline. For example "20230101_ABAur"
    coronagraph:
        If true will use coronagraphic routines for processing.
    calibrate:
        Options for basic image calibration
    analysis:
        Options for PSF/flux analysis in collapsed data
    collapse:
        Options for collapsing image cubes
    target:
        If set, provides options for target object, primarily coordinates. If not set, will use header values.
    specphot:
        If set, provides options for spectrophotometric calibration. If not set, will leave data in units of ``adu``.
    polarimetry:
        If set, provides options for polarimetric differential imaging (PDI)
    dpp_version:
        The version of vampires_dpp that this configuration file is valid with. Typically not set by user.
    save_adi_cubes:
        If true, will save ADI cubes and derotation angles in product directory.
    diff_images:
        Diagnostic difference imaging options. Double-differencing requires an FLC.

    """

    name: str = ""
    diff_images: Literal["singlediff", "doublediff", "both"] | None = None
    save_adi_cubes: bool = True
    coronagraphic: bool = False
    dpp_version: str = dpp.__version__
    target: TargetConfig | None = None
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
        filename: PathLike
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

    def to_toml(self) -> str:
        """Create serializable TOML string"""
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True, mode="json", round_trip=True)
        return tomli_w.dumps(model_dict)

    def save(self, filename: Path):
        """Save configuration settings to TOML file

        Parameters
        ----------
        filename: PathLike
            Output filename
        """
        # get serializable output using pydantic
        model_dict = self.model_dump(exclude_none=True, mode="json", round_trip=True)
        # save output TOML
        with Path(filename).open("wb") as fh:
            tomli_w.dump(model_dict, fh)
