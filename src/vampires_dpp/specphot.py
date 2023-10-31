from pathlib import Path
from typing import Final

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import QTable
from synphot import Empirical1D, Observation, SourceSpectrum, SpectralElement
from synphot.models import Empirical1D, get_waveset
from synphot.units import VEGAMAG

from .pipeline.config import SpecphotConfig

# vampires_dpp/data
BASE_DIR: Final[Path] = Path(__file__).parent / "data"
PICKLES_DIR: Final[Path] = BASE_DIR / "pickles_uvk"
TYPES: Final[tuple] = ("V", "IV", "III", "II", "I")

SCEXAO_AREA: Final = 40.64 * u.m**2
VEGASPEC: Final[SourceSpectrum] = SourceSpectrum.from_vega()

VAMPIRES_STD_FILTERS: Final[set] = {
    "Open",
    "625-50",
    "675-50",
    "725-50",
    "750-50",
    "775-50",
}
VAMPIRES_MBI_FILTERS: Final[set] = {
    "F610",
    "F670",
    "F720",
    "F760",
}
VAMPIRES_NB_FILTERS: Final[set] = {"Halpha", "Ha-Cont", "SII", "SII-Cont"}
VAMPIRES_FILTERS: Final[set] = set.union(
    VAMPIRES_STD_FILTERS, VAMPIRES_MBI_FILTERS, VAMPIRES_NB_FILTERS
)

FILTERS: Final[dict[str, SpectralElement]] = {
    "U": SpectralElement.from_filter("johnson_u"),
    "B": SpectralElement.from_filter("johnson_b"),
    "V": SpectralElement.from_filter("johnson_v"),
    "R": SpectralElement.from_filter("johnson_r"),
    "I": SpectralElement.from_filter("johnson_i"),
    "J": SpectralElement.from_filter("johnson_j"),
    "H": SpectralElement.from_filter("bessel_h"),
    "K": SpectralElement.from_filter("johnson_k"),
}


def load_vampires_filter(name: str, csv_path=BASE_DIR / "vampires_filters.csv"):
    return SpectralElement.from_file(
        str(csv_path.absolute()), wave_unit="nm", include_names=["wave", name]
    )


def update_header_with_filt_info(header: fits.Header, filt: SpectralElement) -> fits.Header:
    waves = filt.waveset
    through = filt.model.lookup_table
    above_50, _ = np.nonzero(through >= 0.5 * np.nanmax(through))
    waveset = waves[above_50]
    header["WAVEMIN"] = waveset[0].to(u.nm), "[nm] Cut-on wavelength (50%)"
    header["WAVEMAX"] = waveset[-1].to(u.nm), "[nm] Cut-off wavelength (50%)"
    header["WAVEAVE"] = filt.avgwave(waveset).to(u.nm), "[nm] Average bandpass wavelength"
    header["FWHM"] = (
        header["WAVEMAX"] - header["WAVEMIN"],
        "[nm] Width of 50% transmission through bandpass",
    )
    header["DLAMLAM"] = header["FWHM"] / header["WAVEAVE"], "Filter inverse spectral resolution"
    return header


def save_filter_fits(filt: SpectralElement, outpath, force=False):
    if not force and outpath.exists():
        return outpath
    filt.to_fits(outpath, overwrite=True)
    return outpath


def prepare_pickles_dict(base_dir: Path = PICKLES_DIR) -> dict[str, Path]:
    tbl = fits.getdata(base_dir / "pickles_uk.fits")
    fnames = (base_dir / f"{fname}.fits" for fname in tbl["FILENAME"])
    return dict(zip(tbl["SPTYPE"], fnames))


PICKLES_MAP: Final[dict[str, Path]] = prepare_pickles_dict(PICKLES_DIR)


def load_pickles(spec_type: str) -> SourceSpectrum:
    filename = PICKLES_MAP[spec_type]
    tbl = fits.getdata(filename)
    return SourceSpectrum(
        Empirical1D,
        points=tbl["WAVELENGTH"] * u.angstrom,
        lookup_table=tbl["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
    )


def color_correction(
    model: SourceSpectrum, filt1: SpectralElement, filt2: SpectralElement
) -> float:
    """
    Return the magnitude of the color correction from the first filter to the second filter.

        mc = m2 - m1
    """
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    color = obs2.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs1.effstim(VEGAMAG, vegaspec=VEGASPEC)
    return color.value


def load_source(config: SpecphotConfig):
    match config.source:
        case "pickles":
            return load_pickles(config.sptype)
        case _:
            return SourceSpectrum.from_file(config.source)


def get_observation(config: SpecphotConfig, obs_filt: SpectralElement) -> Observation:
    # load source spectrum
    source = load_source(config)
    # load filters and calculate color correction
    if config.mag and config.mag_band:
        source_filt = FILTERS[config.mag_band]
        color_cor = color_correction(source, source_filt, obs_filt)
        # calculate corrected mag and normalize spectrum
        mag_cor = config.mag + color_cor
        source = source.normalize(mag_cor * VEGAMAG, source_filt, vegaspec=VEGASPEC, force="taper")

    # create observation
    return Observation(source, obs_filt)


def update_header_from_obs(
    header: fits.Header, config: SpecphotConfig, obs_filt: SpectralElement, flux: float
) -> fits.Header:
    # add filter info
    header = update_header_with_filt_info(header, obs_filt)
    # config inputs
    header["MAG"] = config.mag, "[mag] Source magnitude"
    header["MAGFLT"] = config.mag_band, "Source filter"
    # get source mag/flux
    obs = get_observation(config, obs_filt)
    obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
    obs_jy = obs.effstim(u.Jy)
    header["OBSMAG"] = obs_mag.value, "[mag] Source magnitude after color-correction"
    header["OBSFLX"] = obs_jy.value, "[Jy] Source flux after color-correction"
    # get calibrated flux (adu / s)
    inst_flux = flux / header["EXPTIME"]
    inst_mag = -2.5 * np.log10(inst_flux)
    header["INSFLX"] = inst_flux, "[adu/s] Instrumental flux"
    header["INSMAG"] = inst_mag, "[mag] Instrumental magnitude"
    # calculate surface density conversion factory
    pxarea = (header["PXSCALE"] / 1e3) ** 2
    c_fd = obs_jy / inst_flux
    header["C_FD"] = c_fd, "[Jy/(adu/s)] Absolute flux conversion factor"
    header["PXAREA"] = pxarea, "[arcsec^2/pix] Solid angle of each pixel"
    # calculate Vega zero point
    zp = obs_mag.value - inst_mag
    zp_jy = c_fd * 10 ** (0.4 * zp)
    header["ZPMAG"] = zp, "[mag] Zero point in the Vega magnitude system"
    header["ZPJY"] = zp_jy, "[Jy] Zero point in the Vega magnitude system"
    # calculate total throughput (atmosphere + instrument + QE)
    waverange = header["WAVEMIN"] * u.nm, header["WAVEMAX"] * u.nm
    throughput = inst_flux / obs.countrate(area=SCEXAO_AREA, waverange=waverange).value
    header["THROUGH"] = throughput, "[adu/ct] Estimated total throughput (Atm+Inst+QE)"
    return header


def get_flux_from_metrics(metrics, config: SpecphotConfig) -> float:
    match config.flux_metric:
        case "photometry":
            fluxes = metrics["photr"]
        case "sum":
            fluxes = metrics["sum"]
    return np.nanmedian(fluxes)


def convert_to_surface_brightness(data, header):
    # Jy / arc^2 / (adu/s)
    conv_factor = header["C_FD"] / header["PXAREA"]
    # convert data to Jy / arc^2
    return data / header["EXPTIME"] * conv_factor


def specphot_calibration(header, metrics, outdir: Path, config: SpecphotConfig):
    if "MBI" in header["OBS-MOD"]:
        filt = header["FIELD"]
    else:
        filt = header["FILTER01"]
    obs_filt = load_vampires_filter(filt)
    save_filter_fits(obs_filt, outdir / f"{filt}_filter.fits")
    flux = get_flux_from_metrics(metrics, config)
    header = update_header_from_obs(header, config, obs_filt, flux)
    return header
