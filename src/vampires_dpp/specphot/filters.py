import urllib
from pathlib import Path
from typing import Final

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from synphot import SpectralElement

VAMPIRES_STD_FILTERS: Final[set] = {"Open", "625-50", "675-50", "725-50", "750-50", "775-50"}
VAMPIRES_MBI_FILTERS: Final[set] = {"F610", "F670", "F720", "F760"}
VAMPIRES_NB_FILTERS: Final[set] = {"Halpha", "Ha-Cont", "SII", "SII-Cont"}
VAMPIRES_FILTERS: Final[set] = set.union(
    VAMPIRES_STD_FILTERS, VAMPIRES_MBI_FILTERS, VAMPIRES_NB_FILTERS
)
VAMPIRES_ND_FILTERS: Final[set] = {"ND10", "ND25"}

FILTERS: Final[dict[str, SpectralElement]] = {
    "U": SpectralElement.from_filter("johnson_u"),
    "B": SpectralElement.from_filter("johnson_b"),
    "V": SpectralElement.from_filter("johnson_v"),
    "r": SpectralElement.from_filter("johnson_r"),
    "i": SpectralElement.from_filter("johnson_i"),
    "J": SpectralElement.from_filter("johnson_j"),
    "H": SpectralElement.from_filter("bessel_h"),
    "K": SpectralElement.from_filter("johnson_k"),
}

VAMP_FILT_KEY: Final[str] = "1FHGh3tLlDUwATP6smFGz0nk2e0NF14rywTUjFTUT1OY"
VAMP_FILT_NAME: Final[str] = urllib.parse.quote("VAMPIRES Filter Curves")
VAMPIRES_FILTER_URL: Final[
    str
] = f"https://docs.google.com/spreadsheets/d/{VAMP_FILT_KEY}/gviz/tq?tqx=out:csv&sheet={VAMP_FILT_NAME}"


def determine_primary_filter(header):
    if "MBI" in header["OBS-MOD"]:
        if "FIELD" in header:
            filt_name = header["FIELD"]
        else:
            msg = "Unable to load MBI filter without 'FIELD' keyword, provide filter manually."
            raise ValueError(msg)
    elif "SDI" in header["OBS-MOD"] or header["FILTER02"].lower() != "open":
        filt_name = header["FILTER02"]
    else:
        filt_name = header["FILTER01"]
    return filt_name


def load_vampires_filter(name: str) -> SpectralElement:
    if name not in VAMPIRES_FILTERS:
        msg = f"VAMPIRES filter '{name}' not recognized"
        raise ValueError(msg)
    csv_path = download_file(VAMPIRES_FILTER_URL, cache=True)
    return SpectralElement.from_file(csv_path, wave_unit="nm", include_names=["wave", name])


def update_header_with_filt_info(header: fits.Header) -> tuple[fits.Header, SpectralElement]:
    filt_name = determine_primary_filter(header)
    filt = load_vampires_filter(filt_name)
    waves = filt.waveset
    through = filt.model.lookup_table
    above_50 = np.nonzero(through >= 0.5 * np.nanmax(through))
    waveset = waves[above_50]
    header["FILTNAME"] = filt_name, "DPP filter name"
    header["WAVEMIN"] = waveset[0].to(u.nm).value, "[nm] Cut-on wavelength (50%)"
    header["WAVEMAX"] = waveset[-1].to(u.nm).value, "[nm] Cut-off wavelength (50%)"
    header["WAVEAVE"] = (filt.avgwave(waveset).to(u.nm).value, "[nm] Average bandpass wavelength")
    header["WAVEFWHM"] = (
        header["WAVEMAX"] - header["WAVEMIN"],
        "[nm] Bandpass full width at half-max",
    )
    header["DLAMLAM"] = (header["WAVEFWHM"] / header["WAVEAVE"], "Filter relative bandwidth")
    header["DIAMETER"] = 7.92, "[m] Telescope aperture diameter"
    header["RESELEM"] = (
        np.rad2deg(header["WAVEAVE"] * 1e-9 / header["DIAMETER"]) * 3.6e6,
        "[mas] Resolution element (WAVEAVE/DIAMETER)",
    )
    return header, filt


def save_filter_fits(filt: SpectralElement, outpath: Path, force=False) -> Path:
    if not force and outpath.exists():
        return outpath
    filt.to_fits(outpath, overwrite=True)
    # with fits.open(outpath, "update") as hdul:
    #     update_header_with_filt_info(hdul[0].header)
    return outpath


def determine_filterset_from_header(header):
    mod = header["OBS-MOD"]
    if mod.endswith("MBIR"):
        filts = ("F670", "F720", "F760")
    elif mod.endswith("MBI"):
        filts = ("F610", "F670", "F720", "F760")
    elif mod.endswith("SDI"):
        filts = (header["FILTER02"],)
    else:
        filts = (header["FILTER01"],)
    return filts
