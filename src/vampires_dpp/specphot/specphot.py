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

    hdul = measure_inst_flux(hdul, metrics, config.flux_metric)

    match config.unit:
        case "e-/s":
            conv_factor = 1
        case "Jy":
            conv_factor = determine_jy_factor(hdul, config)
        case "Jy/arcsec^2":
            conv_factor = determine_jy_factor(hdul, config) / hdul[0].header["PXAREA"]

    hdul[0].data *= conv_factor
    hdul["ERR"].data *= conv_factor

    info = fits.Header()
    info["BUNIT"] = config.unit

    for hdu in hdul:
        hdu.header |= info

    return hdul


def measure_inst_flux(hdul, metrics, flux_metric: FluxMetric):
    info = fits.Header()

    match flux_metric:
        case "photometry":
            flux = metrics["photf"]
        case "sum":
            flux = metrics["sum"]

    # flux has units (ntime, nlambda, npsfs)
    # collapse all but wavelength axis
    inst_flux = np.squeeze(np.nanmedian(flux, axis=(0, 2)))
    inst_mag = -2.5 * np.log10(inst_flux)
    if flux.shape[1] > 1:
        for i in range(2, len(hdul)):
            field = hdul[i].header["FIELD"]
            info[f"hierarch DPP SPECPHOT INSTFLUX {field}"] = (
                inst_flux[i],
                "[e-/s] Instrumental flux",
            )
            info[f"hierarch DPP SPECPHOT INSTMAG {field}"] = (
                inst_mag[i],
                "[mag] Instrumental magnitude",
            )

    else:
        # get calibrated flux (e- / s)
        info["hierarch DPP SPECPHOT INSTFLUX"] = inst_flux, "[e-/s] Instrumental flux"
        info["hierarch DPP SPECPHOT INSTMAG"] = inst_mag, "[mag] Instrumental magnitude"

    for hdu in hdul:
        hdu.header |= info

    return hdul


def determine_jy_factor(hdul, config: SpecphotConfig):
    info = fits.Header()
    # config inputs
    info["hierarch DPP SPECPHOT REFMAG"] = config.mag, "[mag] Reference source magnitude"
    info["hierarch DPP SPECPHOT REFFILT"] = config.mag_band, "Reference source filter"

    # determine whether MBI or not
    fluxes = hdul[0].header["hierarch DPP SPECPHOT INSTFLUX*"]
    if len(fluxes) > 1:
        for i in range(2, len(hdul)):
            header = hdul[i].header
            field = header["FIELD"]
            header, obs_filt = update_header_with_filt_info(header)
            # get source mag/flux
            obs = get_observation(config, obs_filt)
            obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
            obs_jy = obs.effstim(u.Jy)
            info[f"hierarch DPP SPECTPHOT MAG {field}"] = (
                obs_mag.value,
                "[mag] Source magnitude after color correction",
            )
            info[f"hierarch DPP SPECPHOT REFFLUX {field}"] = (
                obs_jy.value,
                "[Jy] Source flux after color correction",
            )
            # calculate surface density conversion factory
            inst_flux = header[f"hierarch DPP SPECPHOT INSTFLUX {field}"]
            inst_mag = header[f"hierarch DPP SPECPHOT INSTMAG {field}"]
            c_fd = obs_jy / inst_flux
            info[f"hierarch DPP SPECPHOT CALIBFAC {field}"] = (
                c_fd.value,
                "[Jy/(e-/s)] Absolute flux conversion factor",
            )
            # calculate Vega zero point
            zp = obs_mag.value - inst_mag
            zp_jy = c_fd * 10 ** (0.4 * zp)
            info[f"hierarch DPP SPECPHOT ZEROPT {field}"] = (
                zp,
                "[mag] Zero point in the Vega magnitude system",
            )
            info[f"hierarch DPP SPECPHOT ZEROPTJY {field}"] = (
                zp_jy.value,
                "[Jy] Vega zero point in Jy",
            )
            # calculate total throughput (atmosphere + instrument + QE)
            throughput = inst_flux / obs.countrate(area=SCEXAO_AREA).value
            info[f"hierarch DPP SPECPHOT THROUGH {field}"] = (
                throughput,
                "[e-/ct] Est. total throughput (Atm+Inst+QE)",
            )

    else:
        header = hdul[0].header
        header, obs_filt = update_header_with_filt_info(header)

        # get source mag/flux
        obs = get_observation(config, obs_filt)
        obs_mag = obs.effstim(VEGAMAG, vegaspec=VEGASPEC)
        obs_jy = obs.effstim(u.Jy)
        info["hierarch DPP SPECTPHOT MAG"] = (
            obs_mag.value,
            "[mag] Source magnitude after color correction",
        )
        info["hierarch DPP SPECPHOT REFFLUX"] = (
            obs_jy.value,
            "[Jy] Source flux after color correction",
        )
        # calculate surface density conversion factory
        inst_flux = header["hierarch DPP SPECPHOT INSTFLUX"]
        inst_mag = header["hierarch DPP SPECPHOT INSTMAG"]
        c_fd = obs_jy / inst_flux
        info["hierarch DPP SPECPHOT CALIBFAC"] = (
            c_fd.value,
            "[Jy/(e-/s)] Absolute flux conversion factor",
        )
        # calculate Vega zero point
        zp = obs_mag.value - inst_mag
        zp_jy = c_fd * 10 ** (0.4 * zp)
        info["hierarch DPP SPECPHOT ZEROPT"] = zp, "[mag] Zero point in the Vega magnitude system"
        info["hierarch DPP SPECPHOT ZEROPTJY"] = zp_jy.value, "[Jy] Vega zero point in Jy"
        # calculate total throughput (atmosphere + instrument + QE)
        throughput = inst_flux / obs.countrate(area=SCEXAO_AREA).value
        info["hierarch DPP SPECPHOT THROUGH"] = (
            throughput,
            "[e-/ct] Est. total throughput (Atm+Inst+QE)",
        )

    for hdu in hdul:
        hdu.header |= info

    return c_fd


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


def convert_to_surface_brightness(data, header):
    # Jy / arc^2 / (e-/s)
    conv_factor = header["CALIBFAC"] / header["PXAREA"]
    # convert data to Jy / arc^2
    return data * conv_factor
