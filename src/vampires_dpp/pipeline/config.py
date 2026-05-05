from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal

import astropy.units as u
import tomli
import tomli_w
from annotated_types import Interval
from astropy.coordinates import Angle, SkyCoord
from pydantic import BaseModel, Field, field_validator, model_validator

import vampires_dpp as dpp
from vampires_dpp.util import check_version

__all__ = (
    "AnalysisConfig",
    "CalibrateConfig",
    "CombineConfig",
    "NRMConfig",
    "PipelineConfig",
    "PolarimetryConfig",
    "SpecphotConfig",
    "TargetConfig",
)


class TargetConfig(BaseModel):
    """Astronomical coordinate options.

    .. admonition:: Tip: GAIA
       :class: Tip

        This can be auto-generated wtih GAIA coordinate information through the command line ``dpp new`` interface.
    """

    name: str = Field(description="SIMBAD-friendly target name")
    ra: str = Field(description="Right ascension in sexagesimal hour angles")
    dec: str = Field(description="Declination in sexagesimal degrees")
    parallax: float = Field(description="Parallax of system in mas")
    pm_ra: float = Field(default=0, description="Proper motion of RA axis in mas/yr")
    pm_dec: float = Field(default=0, description="Proper motion of DEC axis in mas/yr")
    frame: str = Field(default="icrs", description="Coordinate reference frame")
    obstime: str = Field(
        default="J2016",
        description="Observation time as a string (default J2016 to coincide with GAIA coordinates)",
    )

    @property
    def ra_ang(self):
        return Angle(self.ra, "hour")

    @property
    def dec_ang(self):
        return Angle(self.dec, "deg")

    def get_coord(self) -> SkyCoord:
        """Return SkyCoord from the current parameters."""
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
    """

    unit: Literal["e-/s", "contrast", "Jy", "Jy/arcsec^2"] = Field(
        default="e-/s",
        description="Output unit. (e-/s is the default without spectrophotometry, and source calibration will be skipped)",
    )
    source: Literal["pickles", "zeropoints"] | Path | None = Field(
        default="zeropoints",
        description=(
            "Spectrum source type. If a path, must be a file loadable by "
            "`synphot.SourceSpectrum.from_file`. If 'pickles', uses the pickles atlas. "
            "If 'zeropoints', uses coefficients from Lucas+2024."
        ),
    )
    sptype: str | None = Field(
        default=None,
        description="Only used if `source` is 'pickles'. Stellar spectral type (must be one of the spectral types in the pickles model atlas).",
    )
    mag: float | None = Field(
        default=None, description="Only used if `source` is 'pickles'. Stellar reference magnitude."
    )
    mag_band: Literal["U", "B", "V", "r", "i", "J", "H", "K"] | None = Field(
        default=None,
        description="Only used if `source` is 'pickles'. Stellar reference magnitude band.",
    )
    flux_metric: Literal["photometry", "sum"] = Field(
        default="photometry",
        description=(
            "Which frame analysis statistic to use for determining flux. "
            "'photometry' uses an aperture sum, while 'sum' uses the sum in the analysis cutout window."
        ),
    )

    @model_validator(mode="after")
    def _check_specphot(self) -> "SpecphotConfig":
        if "Jy" in self.unit:
            if self.source is None:
                msg = "Must provide a spectrum, specify stellar model, or use zero points if you want to calibrate to Jy"
                raise ValueError(msg)
            if self.source != "zeropoints" and (
                self.sptype is None or self.mag is None or self.mag_band is None
            ):
                msg = "Must specify target magnitude (and filter) as well as spectral type to use 'pickles' stellar model"
                raise ValueError(msg)
        return self


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
    """

    calib_directory: Path | None = Field(
        default=None,
        description="Path to calibration file directory; if not provided, no calibration will be done regardless of other settings.",
    )
    back_subtract: bool = Field(
        default=True,
        description=(
            "If true will look for background files in `calib_directory` and subtract them if found. "
            "If not found, will subtract detector bias value."
        ),
    )
    flat_correct: bool = Field(
        default=False,
        description="If true will look for flat files in `calib_directory` and perform flat normalization if found.",
    )
    fix_bad_pixels: bool = Field(
        default=False,
        description="If true, run an adaptive sigma-clipping algorithm for one iteration on each frame and correct bad pixels.",
    )
    save_intermediate: bool = Field(
        default=False,
        description="If true, save intermediate calibrated data to the `calibrated/` folder.",
    )


class AnalysisConfig(BaseModel):
    """PSF modeling and analysis options.

    **File Outputs**

    - For each file an `NPZ <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html>_` file is created in ``metrics/``
        - Keys are metrics/centroids/statistics
        - Values are arrays with dimensions ``(nfields, npsfs, nframes)``
    """

    fit_psf_model: bool = Field(
        default=False, description="If true, fits a PSF model to each window."
    )
    psf_model: Literal["moffat",] = Field(
        default="moffat", description="PSF model type (only Moffat available right now)."
    )
    photometry: bool = Field(
        default=True,
        description="If true, measure photometric sums in apertures at the centroid (or DFT centroid if available).",
    )
    phot_aper_rad: float = Field(
        default=8,
        description="Aperture radius in pixels for circular aperture photometry. If 'auto', uses the FWHM from the file header.",
    )
    phot_ann_rad: Sequence[float] | Literal[False] = Field(
        default=False,
        description="If provided, do local background-subtracted photometry with an annulus given as (inner, outer) radius in pixels.",
    )
    strehl: bool = Field(
        default=True,
        description="If true, measure the Strehl ratio by comparing the PSF peak to the synthetic PSF peak (normalized by the flux in a 16-pixel aperture).",
    )
    window_size: int = Field(
        default=21,
        description="Cutout side length when getting cutouts for each PSF; centered on the file centroid estimate. ~21 avoids including too much halo around any coronagraph masks.",
    )


class CombineConfig(BaseModel):
    """Frame combination options

    VAMPIRES data comes in many shapes and sizes. We like to think the core data product is every individual frame, and therefore the FITS cubes we started with does not have to define the boundaries of our data analysis. There are two methods for reshuffling data, currently:

    1. "cube" -- this method will effectively do nothing; the data will be combined by their original FITS cubes
    2. "pdi" -- this method will combine all frames from a single HWP angle, and is required for polarimetry

    When data is combined it will become a single FITS file.
    """

    method: Literal["cube", "pdi"] = Field(
        default="cube",
        description="Frame combination method. 'cube' keeps the original FITS cube boundaries; 'pdi' combines all frames from a single HWP angle.",
    )
    save_intermediate: bool = Field(
        default=False,
        description="If true, save the combined data cubes into the `combined/` folder (WARNING: can lead to large data volume).",
    )


class FrameSelectConfig(BaseModel):
    """Frame selection options."""

    frame_select: bool = Field(
        default=False,
        description="If true, use the given metric to select frames for inclusion/exclusion from each data cube.",
    )
    metric: Literal["max", "l2norm", "normvar", "strehl"] = Field(
        default="strehl", description="Frame selection metric."
    )
    cutoff: Annotated[float, Interval(ge=0, le=1)] = Field(
        default=0,
        description="If `frame_select` is true, this is the cutoff quantile (0 to 1); 0.2 means 20% of frames in each cube are discarded.",
    )
    save_intermediate: bool = Field(
        default=False,
        description="If true, save the frame-selected files to the `frame_select/` folder (WARNING: can lead to large data volume).",
    )


class AlignmentConfig(BaseModel):
    """Frame alignment options."""

    align: bool = Field(
        default=True, description="If true, data will be aligned by the given method."
    )
    pad: bool = Field(
        default=True,
        description="If true, data will be padded so the full FOV is retained after rotation.",
    )
    method: Literal["dft", "com", "peak", "model"] = Field(
        default="dft",
        description="Alignment method (if 'dft' is not provided, it will not be measured at all).",
    )
    crop_width: int = Field(
        default=536,
        description="Post-alignment crop width; should be roughly equal to FOV. Lower values reduce memory footprint.",
    )
    reproject: bool = Field(
        default=False,
        description="If true, reproject cam2 astrometry onto cam1 for better image differences.",
    )
    save_intermediate: bool = Field(
        default=False,
        description="If true, save the registered files to the `registered/` folder (WARNING: can lead to large data volume).",
    )


class CoaddConfig(BaseModel):
    """Frame combination options.

    **File Outputs**

    - Each input file is collapsed and saved into the ``collapsed/`` folder if coadd is true, otherwise will save in the ``registered/`` folder.
    """

    coadd: bool = Field(
        default=True,
        description="If true, coadd each cube of data (cube boundaries determined from the combination method). If false, the data is saved as cubes.",
    )
    method: Literal["median", "mean", "varmean", "biweight"] = Field(
        default="median", description="Coadd reduction method."
    )
    recenter: bool = Field(
        default=True,
        description="If true, measure the centroid of the PSF in the collapsed frame and realign the data.",
    )
    recenter_method: Literal["dft", "com", "peak", "model"] = Field(
        default="dft", description="Only used if `recenter` is true; method for PSF registration."
    )


class DiffImageConfig(BaseModel):
    """Difference image options.

    Synchronized/polarimetric data can be automatically difference-imaged after registration/coadding. Single diff will take ``cam1-cam2`` and ``cam1+cam2``. Double diff will perform single diff first, then subtract FLC state B from FLC state A.
    """

    make_diff: bool = Field(default=False, description="If true, produce difference images.")
    save_double: bool = Field(
        default=False, description="If true, also save double-difference images (requires FLC)."
    )


class PolarimetryConfig(BaseModel):
    """Polarimetric differential imaging (PDI) options.

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
    """

    method: Literal["triplediff", "doublediff"] = Field(
        default="triplediff",
        description=(
            "Polarization calibration method. Difference methods organize data into HWP sets (some data may be discarded "
            "but median collapse can remove e.g. satellite spot effects). 'leastsq' (Mueller calculus) uses all data via mean."
        ),
    )
    derotate: bool = Field(
        default=True,
        description="Derotate images to north up east left when forming Stokes images. Required for Mueller-matrix correction.",
    )
    mm_correct: bool = Field(
        default=True,
        description="Apply Mueller-matrix correction (only applicable to data reduced using a `difference` method).",
    )
    hwp_adi_sync: bool = Field(
        default=True, description="If true, assume the HWP is in pupil-progress.tracking mode."
    )
    use_ideal_mm: bool = Field(
        default=False,
        description="If true and doing Mueller-matrix correction, use only idealized versions for the components in the Mueller-matrix model.",
    )
    ip_correct: bool = Field(
        default=True,
        description="If true, do post-hoc instrumental polarization (IP) correction from the photometric sum in the given region.",
    )
    ip_method: Literal["aperture", "annulus"] = Field(
        default="aperture",
        description="If `ip_correct=True`, this determines the region type for IP measurement.",
    )
    ip_radius: float = Field(
        default=15,
        description="First radius for IP correction. For 'aperture' this is the radius; for 'annulus' this is the inner radius.",
    )
    ip_radius2: float | None = Field(
        default=None,
        description="Second radius for IP correction (only used if `ip_method='annulus'`); the outer radius.",
    )
    cyl_stokes: Literal["azimuthal", "radial"] = Field(
        default="azimuthal",
        description="If 'azimuthal' will calculate (Qphi, Uphi); if 'radial' will calculate (Qr, Ur) in final Stokes products.",
    )
    mask_satspots: bool = Field(
        default=False, description="If true, mask satellite spots when forming Stokes images."
    )

    @model_validator(mode="after")
    def _check_polarimetry(self) -> "PolarimetryConfig":
        if self.mm_correct and not self.derotate:
            msg = "Cannot do MM correction without derotation!"
            raise ValueError(msg)
        return self


class NRMConfig(BaseModel):
    """NRM processing options.

    **File Outputs**

    - For each file an `H5 <https://support.hdfgroup.org/documentation/hdf5/latest/index.html>_` file is created in ``nrm/`` containing the extracted Fourier observables.
    """

    nbootstrap: int = Field(
        default=1000, description="Number of bootstrap samples for PDI calibration."
    )


class PipelineConfig(BaseModel):
    """Data Processing Pipeline options.

    The processing configuration is all done through this class, which can easily be converted to and from TOML. The options will set the processing steps in the pipeline. An important paradigm in the processing pipeline is skipping unnecessary operations. That means if a file already exists, the pipeline will only reprocess it if the `force` flag is set, which will reprocess all files for that step (and subsequent steps), or if the input file or files are newer. You can try this out by deleting one calibrated file from a processed output and re-running the pipeline.

    **File Outputs**

    - Auxilliary files in ``aux/``
        - Copy of config., centroid file, astrometry file, mean PSFs, filter curve(s), synth. PSF(s).
    - Data products (ADI cubes, output file header table) in ``products/``
    - Difference images in ``diff/``
        - ``diff/single/`` and ``diff/double/``
    """

    name: str = Field(
        default="",
        description="Filename-friendly name used for outputs from this pipeline (e.g. '20230101_ABAur').",
    )
    dpp_version: str = Field(
        description="Version of vampires_dpp this configuration file was authored against. Required; the loader will reject configs whose version is not SemVer-compatible with the installed package."
    )
    coronagraphic: bool = Field(
        default=False, description="If true, use coronagraphic routines for processing."
    )
    planetary: bool = Field(
        default=False, description="If true, use planetary routines for processing."
    )
    save_adi_cubes: bool = Field(
        default=False,
        description="If true, save ADI cubes and derotation angles in the product directory.",
    )
    target: TargetConfig | None = Field(
        default=None,
        description="If set, provides target object options (primarily coordinates). If not set, header values are used.",
    )
    combine: CombineConfig = Field(
        default_factory=CombineConfig, description="Options for frame combinations."
    )
    calibrate: CalibrateConfig = Field(
        default_factory=CalibrateConfig, description="Options for basic image calibration."
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        description="Options for PSF/flux analysis in collapsed data.",
    )
    frame_select: FrameSelectConfig = Field(
        default_factory=FrameSelectConfig, description="Options for frame selection."
    )
    align: AlignmentConfig = Field(
        default_factory=AlignmentConfig, description="Options for frame alignment."
    )
    coadd: CoaddConfig = Field(
        default_factory=CoaddConfig, description="Options for coadding image cubes."
    )
    specphot: SpecphotConfig = Field(
        default_factory=SpecphotConfig,
        description="Options for spectrophotometric calibration. If unit is 'e-/s', calibration is skipped.",
    )
    diff_images: DiffImageConfig = Field(
        default_factory=DiffImageConfig,
        description="Diagnostic difference imaging options. Double-differencing requires an FLC.",
    )
    nrm: NRMConfig | None = Field(
        default=None, description="If set, enables NRM (non-redundant masking) processing."
    )
    polarimetry: PolarimetryConfig | None = Field(
        default=None,
        description="If set, enables and provides settings for polarimetric differential imaging (PDI).",
    )

    @field_validator("dpp_version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not check_version(v, dpp.__version__):
            msg = (
                f"Input pipeline version ({v}) is not compatible with installed version of "
                f"`vampires_dpp` ({dpp.__version__}). Try running `dpp upgrade <config>`."
            )
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "PipelineConfig":
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
            self.specphot.unit != "e-/s"
            and self.specphot.source != "zeropoints"
            and self.specphot.flux_metric == "photometry"
            and not self.analysis.photometry
        ):
            msg = "Can't use photometry for specphot.flux_metric if analysis.photometry is False"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, filename: PathLike) -> "PipelineConfig":
        """Load configuration from TOML file.

        The version compatibility check runs as part of model validation;
        a stale or missing `dpp_version` field will raise during validation.
        """
        with Path(filename).open("rb") as fh:
            config = tomli.load(fh)
        return cls.model_validate(config)

    def to_toml(self) -> str:
        """Create a serializable TOML string."""
        model_dict = self.model_dump(exclude_none=True, mode="json", round_trip=True)
        return tomli_w.dumps(model_dict)

    def save(self, filename: PathLike) -> None:
        """Save configuration settings to a TOML file."""
        Path(filename).write_text(self.to_toml())
