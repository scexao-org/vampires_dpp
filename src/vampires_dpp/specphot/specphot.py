from pathlib import Path
from typing import Final

import astropy.units as u
import numpy as np
from astropy.io import fits
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.units import VEGAMAG

from ..pipeline.config import SpecphotConfig
from .filters import FILTERS, load_vampires_filter, save_filter_fits, update_header_with_filt_info
from .pickles import load_pickles_model

__all__ = ["specphot_calibration", "convert_to_surface_brightness"]

# vampires_dpp/data
SCEXAO_AREA: Final = 40.64 * u.m**2
VEGASPEC: Final[SourceSpectrum] = SourceSpectrum.from_vega()


def color_correction(
    model: SourceSpectrum, filt1: SpectralElement, filt2: SpectralElement
) -> float:
    """Return the magnitude of the color correction from the first filter to the second filter.

    mc = m2 - m1

    m2 = m1 + mc
    """
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    color = obs2.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs1.effstim(VEGAMAG, vegaspec=VEGASPEC)
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
    # add filter info
    header = update_header_with_filt_info(header, obs_filt)
    # config inputs
    header["REFMAG"] = config.mag, "[mag] Source magnitude"
    header["REFFILT"] = config.mag_band, "Source filter"
    # get source mag/flux
    obs = get_observation(config, obs_filt)
    obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
    obs_jy = obs.effstim(u.Jy)
    header["MAG"] = obs_mag.value, "[mag] Source magnitude after color correction"
    header["FLUX"] = obs_jy.value, "[Jy] Source flux after color correction"
    # get calibrated flux (adu / s)
    inst_flux = np.maximum(flux / header["EXPTIME"], 0)
    inst_mag = -2.5 * np.log10(inst_flux)
    header["INSTFLUX"] = inst_flux, "[adu/s] Instrumental flux"
    header["INSTMAG"] = inst_mag, "[mag] Instrumental magnitude"
    # calculate surface density conversion factory
    pxarea = (header["PXSCALE"] / 1e3) ** 2
    c_fd = obs_jy / inst_flux
    header["CALIBFAC"] = c_fd.value, "[Jy/(adu/s)] Absolute flux conversion factor"
    header["PXAREA"] = pxarea, "[arcsec^2/pix] Solid angle of each pixel"
    # calculate Vega zero point
    zp = obs_mag.value - inst_mag
    zp_jy = c_fd * 10 ** (0.4 * zp)
    header["ZEROPT"] = zp, "[mag] Zero point in the Vega magnitude system"
    header["ZEROPTJY"] = zp_jy.value, "[Jy] Vega zero point in Jy"
    # calculate total throughput (atmosphere + instrument + QE)
    inst_flux_e = inst_flux * header["GAIN"] / max(header.get("DETGAIN", 1), 1)
    throughput = inst_flux_e / obs.countrate(area=SCEXAO_AREA).value
    header["THROUGH"] = throughput, "[e-/ct] Est. total throughput (Atm+Inst+QE)"
    return header


def get_flux_from_metrics(metrics, config: SpecphotConfig) -> float:
    match config.flux_metric:
        case "photometry":
            fluxes = metrics["photf"]
        case "sum":
            fluxes = metrics["sum"]
    return np.nanmedian(fluxes)


def convert_to_surface_brightness(data, header):
    # Jy / arc^2 / (adu/s)
    conv_factor = header["CALIBFAC"] / header["PXAREA"]
    # convert data to Jy / arc^2
    return data / header["EXPTIME"] * conv_factor


def specphot_calibration(header, outdir: Path, config: SpecphotConfig):
    if "MBI" in header["OBS-MOD"]:
        filt = header["FIELD"]
    elif "SDI" in header["OBS-MOD"]:
        filt = header["FILTER02"]
    else:
        filt = header["FILTER01"]
    obs_filt = load_vampires_filter(filt)
    save_filter_fits(obs_filt, outdir / f"VAMPIRES_{filt}_filter.fits")
    match config.flux_metric:
        case "photometry":
            flux = header["PHOTF"]
        case "sum":
            flux = header["SUM"]
    header = update_header_from_obs(header, config, obs_filt, flux)
    return header
