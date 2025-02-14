from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal

import astropy.units as u
import tomli
import tomli_w
from annotated_types import Interval
from astropy.coordinates import Angle, SkyCoord
from pydantic import BaseModel

import vampires_dpp as dpp
from vampires_dpp.util import check_version

__all__ = (
    "TargetConfig",
    "SpecphotConfig",
    "CalibrateConfig",
    "CombineConfig",
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
        Spectrum source type. If a path, must be a file that can be loaded by `synphot.SourceSpectrum.from_file`. If "pickles", uses the pickles atlast. If "zeropoints", uses coefficients from Lucas+2024.
    sptype:
        Only used if `source` is "pickles". Stellar spectral type. Note: must be one of the spectral types available in the pickles model atlas. Refer to the STScI documentation for more information on available spectral types.
    mag:
        Only used if `source` is "pickles". Stellar reference magnitude
    mag_band:
        Only used if `source` is "pickles". Stellar reference magnitude band
    unit:
        Output unit. (Note: e-/s is the default without spectrophotometry, and source calibration will be skipped)
    flux_metric:
        Which frame analysis statistic to use for determining flux. "photometry" uses an aperture sum, while "sum" uses the sum in the analysis cutout window.
    """

    unit: Literal["e-/s", "contrast", "Jy", "Jy/arcsec^2"] = "e-/s"
    source: Literal["pickles", "zeropoints"] | Path | None = "zeropoints"
    sptype: str | None = None
    mag: float | None = None
    mag_band: Literal["U", "B", "V", "r", "i", "J", "H", "K"] | None = None
    flux_metric: Literal["photometry", "sum"] = "photometry"

    def model_post_init(self, __context: Any) -> None:
        if "Jy" in self.unit:
            if self.source is None:
                msg = "Must provide a spectrum, specify stellar model, or use zero points if you want to calibrate to Jy"
                raise ValueError(msg)
            if (
                self.source == "pickles"
                and self.sptype is None
                or self.mag is None
                or self.mag_band is None
            ):
                msg = "Must specify target magnitude (and filter) as well as spectral type to use 'pickles' stellar model"
                raise ValueError(msg)

        return super().model_post_init(__context)


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
    reproject:
        If true, will use custom astrometry solution to warp frame
    save_intermediate:
        If true, will save intermediate calibrated data to ``calibrated/`` folder.
    """

    calib_directory: Path | None = None
    back_subtract: bool = True
    flat_correct: bool = False
    fix_bad_pixels: bool = False
    save_intermediate: bool = False


class AnalysisConfig(BaseModel):
    """PSF modeling and analysis options.


        **File Outputs**

        - For each file an `NPZ <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html>_` file is created in ``metrics/``
            - Keys are metrics/centroids/statistics
            - Values are arrays with dimensions ``(nfields, npsfs, nframes)``


    Parameters
    ----------
    fit_psf_model:
        If true, fits a PSF model to each window
    psf_model:
        Only Moffat available right now
    photometry:
        If true, will measure photometric sums in apertures at the centroid (or the DFT centroid if available)
    phot_aper_rad:
        Aperture radius in pixels for circular aperture photometry. If "auto", will use the FWHM from the file header.
    phot_ann_rad:
        If provided, will do local background-subtracted photometry with an annulus with the given inner and outer radius, in pixels.
    strehl:
        If true, will measure the Strehl ratio by comparing the PSF peak to the synthetic PSF peak, normalized by the flux in an aperture 16 pixels wide.
    window_size:
        The cutout side length when getting cutouts for each PSF. Cutouts are centered around the file centroid estimate. A size of 21 is a decent size to avoid including too much of the PSF halo around any coronagraph masks.
    """

    fit_psf_model: bool = False
    psf_model: Literal["moffat",] = "moffat"
    photometry: bool = True
    phot_aper_rad: float = 8
    phot_ann_rad: Sequence[float] | Literal[False] = False
    strehl: bool = True
    window_size: int = 21


class CombineConfig(BaseModel):
    """Frame combination options

    VAMPIRES data comes in many shapes and sizes. We like to think the core data product is every individual frame, and therefore the FITS cubes we started with does not have to define the boundaries of our data analysis. There are two methods for reshuffling data, currently:

    1. "cube" -- this method will effectively do nothing; the data will be combined by their original FITS cubes
    2. "pdi" -- this method will combine all frames from a single HWP angle, and is required for polarimetry

    When data is combined it will become a single FITS file

    Parameters
    ----------
    method:
    save_intermediate:
        If true, will save the combined data cubes into the ``<output>/combined`` folder (WARNING can lead to insane data volume)
    """

    method: Literal["cube", "pdi"] = "cube"
    save_intermediate: bool = False


class FrameSelectConfig(BaseModel):
    """Frame selection options

    Parameters
    ----------
    frame_select:
        If true, will use the given metric to select frames for inclusions/exclusion from each data cube.
    metric:
        Frame selection metric
    cutoff:
        If ``frame_select`` is provided, this is the cutoff _quantile_ (from 0 to 1), where 0.2 means 20% of the frames
        from each cube will be discarded according the the selection metric.
    save_intermediate:
        If true, will save the frame-selected files to the ``<output>frame_select`` folder (WARNING can lead to insane data volume)
    """

    frame_select: bool = False
    metric: Literal["max", "l2norm", "normvar", "strehl"] = "strehl"
    cutoff: Annotated[float, Interval(ge=0, le=1)] = 0
    save_intermediate: bool = False


class AlignmentConfig(BaseModel):
    """Frame alignment options

    Parameters
    ----------
    align:
        If true, data will be aligned by the give method
    method:
        Alignment method, (if "dft" is not provided, it will not be measured at all)
    crop_width:
        Post-alignment crop width, should be roughly equal to FOV. Cropped data can set this lower for reduced memory footprint.
    reproject:
        If true, will reproject cam2 astrometry onto cam1 for better image differences
    save_intermediate:
        If true, will save the registered files to the ``<output>/registered`` folder (WARNING can lead to insance data volume)
    """

    align: bool = True
    method: Literal["dft", "com", "peak", "model"] = "dft"
    crop_width: int = 536
    reproject: bool = False
    save_intermediate: bool = False


class CoaddConfig(BaseModel):
    """Frame combination options


    **File Outputs**

        - Each input file is collapsed and saved into the ``collapsed/`` folder if coadd is true, otherwise will save in the ``registered/`` folder.

    Parameters
    ----------
    coadd:
        If true, will coadd each cube of data (where the cube is determined from the combination method). If false, the data will be saved as cubes.
    method:
    recenter:
        If true, will measure the centroid of the PSF in the collapsed frame and realign the data
    recenter_method:
        Only used if recenter is true; method for PSF registration.
    """

    coadd: bool = True
    method: Literal["median", "mean", "varmean", "biweight"] = "median"
    recenter: bool = True
    recenter_method: Literal["dft", "com", "peak", "model"] = "dft"


class DiffImageConfig(BaseModel):
    """Difference image options

    Synchronized/polarimetric data can be automatically difference imaged after registration/coadding. Single diff will take cam1-cam2 and cam1+cam2. Double diff will perform single diff first, and then subtract FLC state B from FLC state A.
    """

    make_diff: bool = False
    save_double: bool = False


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
    derotate:
        Derotate images to north up east left when forming Stokes images. Required for Mueller-matrix correction
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
    cyl_stokes:
        If 'azimuthal' will calculate (Qphi, Uphi); if 'radial' will calculate (Qr, Ur) in final Stokes products.
    """

    method: Literal["triplediff", "doublediff"] = "triplediff"
    derotate: bool = True
    mm_correct: bool = True
    hwp_adi_sync: bool = True
    use_ideal_mm: bool = False
    ip_correct: bool = True
    ip_method: Literal["aperture", "annulus"] = "aperture"
    ip_radius: float = 15
    ip_radius2: float | None = None
    cyl_stokes: Literal["azimuthal", "radial"] = "azimuthal"

    def model_post_init(self, __context: Any) -> None:
        if self.mm_correct and not self.derotate:
            msg = "Cannot do MM correction without derotation!"
            raise ValueError(msg)
        return super().model_post_init(__context)


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
    dpp_version:
        The version of vampires_dpp that this configuration file is valid with. Typically not set by user.
    coronagraphic:
        If true will use coronagraphic routines for processing.
    planetary:
        If true will use planetary routines for processing.
    save_adi_cubes:
        If true, will save ADI cubes and derotation angles in product directory.
    target:
        If set, provides options for target object, primarily coordinates. If not set, will use header values.
    combine:
        Options for frame combinations
    calibrate:
        Options for basic image calibration
    analysis:
        Options for PSF/flux analysis in collapsed data
    frame_select:
        Options for frame selection
    align:
        Options for frame alignment
    coadd:
        Options for coadding image cubes
    specphot:
        If set, provides options for spectrophotometric calibration. If not set, will leave data in units of ``adu``.
    diff_images:
        Diagnostic difference imaging options. Double-differencing requires an FLC.
    polarimetry:
        If set, enables and provides settings for polarimetric differential imaging (PDI).
    """

    name: str = ""
    dpp_version: str = dpp.__version__
    coronagraphic: bool = False
    planetary: bool = False
    save_adi_cubes: bool = False
    target: TargetConfig | None = None
    combine: CombineConfig = CombineConfig()
    calibrate: CalibrateConfig = CalibrateConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    frame_select: FrameSelectConfig = FrameSelectConfig()
    align: AlignmentConfig = AlignmentConfig()
    coadd: CoaddConfig = CoaddConfig()
    specphot: SpecphotConfig = SpecphotConfig()
    diff_images: DiffImageConfig = DiffImageConfig()
    polarimetry: PolarimetryConfig | None = None

    def model_post_init(self, __context: Any) -> None:
        if (
            self.frame_select.frame_select
            and self.frame_select.metric == "strehl"
            and not self.analysis.strehl
        ):
            msg = "You must set `strehl=true` in the analysis section if you want to use the Strehl ratio as a selection metric"
            raise ValueError(msg)
        if self.align.align and self.align.method == "model" and not self.analysis.fit_psf_model:
            msg = "You must set `fit_psf_model` to true if you want to align using the PSF model centroid"
            raise ValueError(msg)
        if (
            self.specphot.flux_metric == "photometry"
            and not self.analysis.photometry
            and self.specphot.unit != "e-/s"
        ):
            msg = "Can't use photometry for specphot.flux_metric if analysis.photometry is False"
            raise ValueError(msg)
        return super().model_post_init(__context)

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
