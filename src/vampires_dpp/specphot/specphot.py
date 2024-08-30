from typing import Final, Literal, TypeAlias

import astropy.units as u
import numpy as np
from astropy.io import fits
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.units import VEGAMAG

from vampires_dpp.pipeline.config import SpecphotConfig

from .filters import FILTERS, update_header_with_filt_info
from .pickles import load_pickles_model

SpecphotUnits: TypeAlias = Literal["e-/s", "Jy", "Jy/arcsec^2"]
FluxMetric: TypeAlias = Literal["photometry", "sum"]

# vampires_dpp/data
SCEXAO_AREA: Final = 40.64 * u.m**2
VEGASPEC: Final[SourceSpectrum] = SourceSpectrum.from_vega()


def specphot_cal_hdul(hdul: fits.HDUList, metrics, config: SpecphotConfig):
    # determine any conversion factors
    if config.unit in ("Jy", "Jy/arcsec^2"):
        assert metrics, "Must provide metrics to calculate photometry"

    hdul, inst_flux = measure_inst_flux(hdul, metrics, config.flux_metric)

    match config.unit:
        case "e-/s":
            conv_factor = 1
        case "Jy":
            conv_factor = determine_jy_factor(hdul, inst_flux, config)
        case "Jy/arcsec^2":
            conv_factor = determine_jy_factor(hdul, inst_flux, config) / hdul[0].header["PXAREA"]

    hdul[0].data *= conv_factor
    hdul["ERR"].data *= conv_factor

    info = fits.Header()
    info["BUNIT"] = config.unit

    for hdu in hdul:
        hdu.header.update(info)

    return hdul


def _format(number, sigfigs=4):
    return float(f"%.{sigfigs-1}g" % number)


def measure_inst_flux(hdul, metrics, flux_metric: FluxMetric):
    info = fits.Header()

    match flux_metric:
        case "photometry":
            flux = metrics["photf"]
        case "sum":
            flux = metrics["sum"]
    # flux has units (nlambda, npsfs, ntime)
    # collapse all but wavelength axis
    inst_flux = np.nanmedian(flux, axis=(1, 2))
    for flux, hdu in zip(inst_flux, hdul[2:], strict=True):
        field = hdu.header["FIELD"]
        info[f"hierarch DPP SPECPHOT INSTFLUX {field}"] = _format(flux), "[e-/s] Instrumental flux"
        info[f"hierarch DPP SPECPHOT INSTMAG {field}"] = (
            np.round(-2.5 * np.log10(flux), 3),
            "[mag] Instrumental magnitude",
        )

    for hdu in hdul:
        hdu.header.update(info)
    return hdul, inst_flux


def determine_jy_factor(hdul, fluxes, config: SpecphotConfig):
    info = fits.Header()
    # config inputs
    info["hierarch DPP SPECPHOT REFMAG"] = (
        np.round(config.mag, 3),
        "[mag] Reference source magnitude",
    )
    info["hierarch DPP SPECPHOT REFFILT"] = config.mag_band, "Reference source filter"

    # determine whether MBI or not
    conv_factors = []
    for flux, hdu in zip(fluxes, hdul[2:], strict=True):
        header = hdu.header
        field = header["FIELD"]
        header, obs_filt = update_header_with_filt_info(header)
        # get source mag/flux
        obs = get_observation(config, obs_filt)
        obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
        obs_jy = obs.effstim(u.Jy)
        info[f"hierarch DPP SPECPHOT MAG {field}"] = (
            np.round(obs_mag.value, 3),
            "[mag] Source magnitude after color correction",
        )
        info[f"hierarch DPP SPECPHOT REFFLUX {field}"] = (
            _format(obs_jy.value),
            "[Jy] Source flux after color correction",
        )
        # calculate surface density conversion factory
        inst_mag = -2.5 * np.log10(flux)
        c_fd = obs_jy.value / flux
        info[f"hierarch DPP SPECPHOT CALIBFAC {field}"] = (
            _format(c_fd),
            "[Jy/(e-/s)] Absolute flux conversion factor",
        )
        # calculate Vega zero point
        zp = obs_mag.value - inst_mag
        zp_jy = c_fd * 10 ** (0.4 * zp)
        info[f"hierarch DPP SPECPHOT ZEROPT {field}"] = (
            np.round(zp, 3),
            "[mag] Zero point in the Vega magnitude system",
        )
        info[f"hierarch DPP SPECPHOT ZEROPTJY {field}"] = (zp_jy, "[Jy] Vega zero point in Jy")
        # calculate total throughput (atmosphere + instrument + QE)
        throughput = flux / obs.countrate(area=SCEXAO_AREA).value
        info[f"hierarch DPP SPECPHOT THROUGH {field}"] = (
            _format(throughput),
            "[e-/ct] Est. total throughput (Atm+Inst+QE)",
        )
        conv_factors.append(c_fd)
    conv_factors = np.array(conv_factors)[None, :, None, None]

    for hdu in hdul:
        hdu.header.update(info)

    return conv_factors


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


def get_flux_from_metrics(metrics, config: SpecphotConfig) -> float:
    match config.flux_metric:
        case "photometry":
            fluxes = metrics["photf"]
            weights = 1 / metrics["phote"] ** 2
        case "sum":
            fluxes = metrics["sum"]
            weights = 1 / metrics["var"]
    return np.nansum(fluxes * weights) / np.nansum(weights)
