from pathlib import Path
from typing import Final, Literal, TypeAlias

import astropy.units as u
import numpy as np
from astropy.io import fits
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.units import VEGAMAG

from vampires_dpp.pipeline.config import SpecphotConfig

from .filters import FILTERS, save_filter_fits, update_header_with_filt_info
from .pickles import load_pickles_model

__all__ = ("specphot_calibration", "convert_to_surface_brightness", "SpecphotUnits")

SpecphotUnits: TypeAlias = Literal["adu/s", "e-/s", "Jy", "Jy/arcsec^2"]

# vampires_dpp/data
SCEXAO_AREA: Final = 40.64 * u.m**2
VEGASPEC: Final[SourceSpectrum] = SourceSpectrum.from_vega()


def color_correction(
    model: SourceSpectrum, filt1: SpectralElement, filt2: SpectralElement
) -> float:
    """Return the magnitude of the color correction from the first filter to the second filter.

    delta = m1 - m2
    m2 = m1 - delta
    """
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    color = obs1.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs2.effstim(VEGAMAG, vegaspec=VEGASPEC)
    return color.value


def load_source(config: SpecphotConfig):
    match config.source:
        case "pickles":
            return load_pickles_model(config.sptype)
        case _:
            return SourceSpectrum.from_file(config.source)


def get_observation(config: SpecphotConfig, obs_filt: SpectralElement) -> Observation:
    # load source spectrum
    source = load_source(config)
    # load filters and calculate color correction
    if config.mag and config.mag_band:
        source_filt = FILTERS[config.mag_band]
        source = source.normalize(
            config.mag * VEGAMAG, source_filt, vegaspec=VEGASPEC, force="taper"
        )

    # create observation
    return Observation(source, obs_filt)


def update_header_from_obs(
    header: fits.Header, config: SpecphotConfig, obs_filt: SpectralElement, flux: float
) -> fits.Header:
    # config inputs
    header["REFMAG"] = config.mag, "[mag] Source magnitude"
    header["REFFILT"] = config.mag_band, "Source filter"
    # get source mag/flux
    obs = get_observation(config, obs_filt)
    obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
    obs_jy = obs.effstim(u.Jy)
    header["MAG"] = obs_mag.value, "[mag] Source magnitude after color correction"
    header["FLUX"] = obs_jy.value, "[Jy] Source flux after color correction"
    # get calibrated flux (e- / s)
    inst_flux = np.maximum(flux, 0)
    inst_mag = -2.5 * np.log10(inst_flux)
    header["INSTFLUX"] = inst_flux, "[e-/s] Instrumental flux"
    header["INSTMAG"] = inst_mag, "[mag] Instrumental magnitude"
    # calculate surface density conversion factory
    c_fd = obs_jy / inst_flux
    header["CALIBFAC"] = c_fd.value, "[Jy/(e-/s)] Absolute flux conversion factor"
    # calculate Vega zero point
    zp = obs_mag.value - inst_mag
    zp_jy = c_fd * 10 ** (0.4 * zp)
    header["ZEROPT"] = zp, "[mag] Zero point in the Vega magnitude system"
    header["ZEROPTJY"] = zp_jy.value, "[Jy] Vega zero point in Jy"
    # calculate total throughput (atmosphere + instrument + QE)
    throughput = inst_flux / obs.countrate(area=SCEXAO_AREA).value
    header["THROUGH"] = throughput, "[e-/ct] Est. total throughput (Atm+Inst+QE)"
    return header


def get_flux_from_metrics(metrics, config: SpecphotConfig) -> float:
    match config.flux_metric:
        case "photometry":
            fluxes = metrics["photf"]
            weights = 1 / metrics["phote"] ** 2
        case "sum":
            fluxes = metrics["sum"]
            weights = 1 / metrics["var"]
    return np.nansum(fluxes * weights) / np.nansum(weights)


def convert_to_surface_brightness(data, header):
    # Jy / arc^2 / (e-/s)
    conv_factor = header["CALIBFAC"] / header["PXAREA"]
    # convert data to Jy / arc^2
    return data * conv_factor


def specphot_calibration(header, outdir: Path, config: SpecphotConfig):
    header, obs_filt = update_header_with_filt_info(header)
    save_filter_fits(obs_filt, outdir / f"VAMPIRES_{header['FILTNAME']}_filter.fits")
    match config.flux_metric:
        case "photometry":
            flux = header["PHOTF"]
        case "sum":
            flux = header["SUM"]
    header = update_header_from_obs(header, config, obs_filt, flux)
    return header


def specphot_cal_hdul(
    hdul: fits.HDUList, metrics, flux_metric: Literal["photometry", "sum"] = "photometry"
):
    ...
