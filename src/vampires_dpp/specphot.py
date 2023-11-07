from pathlib import Path
from typing import Final

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import QTable
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
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
    "r": SpectralElement.from_filter("johnson_r"),
    "i": SpectralElement.from_filter("johnson_i"),
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
    above_50 = np.nonzero(through >= 0.5 * np.nanmax(through))
    waveset = waves[above_50]
    header["WAVEMIN"] = waveset[0].to(u.nm).value, "[nm] Cut-on wavelength (50%)"
    header["WAVEMAX"] = waveset[-1].to(u.nm).value, "[nm] Cut-off wavelength (50%)"
    header["WAVEAVE"] = filt.avgwave(waveset).to(u.nm).value, "[nm] Average bandpass wavelength"
    header["WAVEFWHM"] = (
        header["WAVEMAX"] - header["WAVEMIN"],
        "[nm] Bandpass full width at half-max",
    )
    header["DLAMLAM"] = header["WAVEFWHM"] / header["WAVEAVE"], "Filter inverse spectral resolution"
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
    if spec_type not in PICKLES_MAP:
        raise ValueError(
            f"""No pickles model found for sptype: {spec_type}
        Please see the STScI pickles atlas documentation for info on available spectral types.
        List of downloaded spectral types:\n{', '.join(PICKLES_MAP.keys())}"""
        )
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

        mc = m1 - m2
    """
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    color = obs1.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs2.effstim(VEGAMAG, vegaspec=VEGASPEC)
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
    inst_flux = flux / header["EXPTIME"]
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


def update_header_zeropoints(
    header: fits.Header, config: SpecphotConfig, flux, csv_path=BASE_DIR / "vampires_zeropoints.csv"
) -> fits.Header:
    zp_table = pd.read_csv(csv_path)
    if "MBI" in header["OBS-MOD"]:
        filt = header["FIELD"].strip()
    elif "SDI" in header["OBS-MOD"]:
        filt = header["FILTER02"].strip()
    else:
        filt = header["FILTER01"].strip()
    row = zp_table.loc[zp_table["filter"] == filt].iloc[0]
    # if beamsplitter in, need to double calibfac
    cfd = row["C_FD (Jy.s/e-)"]
    if header["U_BS"].strip().lower() != "open":
        cfd *= 2
    # and convert to adu with gain
    cfd *= header["EFFGAIN"]
    header["CALIBFAC"] = cfd, "[Jy/(adu/s)] Absolute flux conversion factor"
    header["PXAREA"] = (header["PXSCALE"] / 1e3) ** 2, "[arcsec^2/pix] Solid angle of each pixel"
    # get calibrated flux (adu / s)
    inst_flux = flux / header["EXPTIME"]
    inst_flux_e = inst_flux * header["EFFGAIN"]
    inst_mag = -2.5 * np.log10(inst_flux)
    header["INSTFLUX"] = inst_flux, "[adu/s] Instrumental flux"
    header["INSTMAG"] = inst_mag, "[mag] Instrumental magnitude"

    inst_mag_e = -2.5 * np.log10(inst_flux_e)
    zp = row["zp_mag"]
    if header["U_BS"].strip().lower() != "open":
        zp -= 2.5 * np.log10(2)
    mag = inst_mag_e + zp
    header["MAG"] = mag, "[mag] Source mag after zero point correction"
    zp_adus = zp + 2.5 * np.log10(header["EFFGAIN"])
    header["ZEROPT"] = zp_adus, "[mag] Zero point in the Vega magnitude system"
    header["ZEROPTJY"] = 10 ** (-0.4 * zp_adus) * cfd, "[Jy] Vega zero point in Jy"
    header["FLUX"] = inst_flux * cfd, "[Jy] Source flux after zero point correction"
    return header


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
    save_filter_fits(obs_filt, outdir / f"{filt}_filter.fits")
    match config.flux_metric:
        case "photometry":
            flux = header["PHOTF"]
        case "sum":
            flux = header["SUM"]
    match config.source:
        case "zeropoint":
            header = update_header_zeropoints(header, config, flux)
        case _:
            header = update_header_from_obs(header, config, obs_filt, flux)
    return header


def get_simbad_table(obj: str):
    sim = Simbad()
    sim.remove_votable_fields("coordinates")
    sim.add_votable_fields("sptype")
    return sim.query_object(obj)


def get_ucac_flux(obj: str, radius=1):
    cat = "I/322A/out"
    viz = Vizier(catalog=cat, columns=["Vmag", "rmag", "imag"])
    # get precise RA and DEC
    ucac4_list = viz.query_object(
        obj,
        radius=radius * u.arcsec,
    )
    return ucac4_list[0]
