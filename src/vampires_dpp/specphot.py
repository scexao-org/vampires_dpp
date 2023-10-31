from pathlib import Path
from typing import Any, Final

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import QTable
from synphot import Empirical1D, Observation, SourceSpectrum, SpectralElement
from synphot.models import Empirical1D, get_waveset
from synphot.units import VEGAMAG

from .pipeline.config import ObjectConfig


def get_mag(obj: ObjectConfig, obsfilt: str) -> float:
    """
    Get expected flux (after filter color-correction) in Vega mag
    """


def get_flux(obj: ObjectConfig, obsfilt: str) -> float:
    """
    Get expected flux (after filter color-correction) in Jy
    """


# vampires_dpp/data
BASE_DIR: Final[Path] = Path(__file__).parent / "data"


def load_vampires_filter(name: str, csv_path=BASE_DIR / "vampires_filters.csv"):
    return SpectralElement.from_file(
        str(csv_path.absolute()), wave_unit="nm", include_names=["wave", name]
    )


def get_filter_info(filt: SpectralElement) -> dict[str, Any]:
    waves = filt.waveset
    through = filt.model.lookup_table
    above_50 = np.nonzero(through >= 0.5 * np.nanmax(through))[0]
    waveset = waves[above_50]
    filt_info = dict(
        lam_min=waveset[0].to(u.nm),
        lam_max=waveset[-1].to(u.nm),
        lam_ave=filt.avgwave(waveset).to(u.nm),
    )
    filt_info["width"] = filt_info["lam_max"] - filt_info["lam_min"]
    filt_info["dlam/lam"] = filt_info["width"] / filt_info["lam_ave"]
    return filt_info


def save_filter_fits(filt: SpectralElement, outpath):
    pass  # TODO


VAMPIRES_STD_FILTERS = {
    "Open",
    "625-50",
    "675-50",
    "725-50",
    "750-50",
    "775-50",
}
VAMPIRES_MBI_FILTERS = {
    "F610",
    "F670",
    "F720",
    "F760",
}
VAMPIRES_NB_FILTERS = {"Halpha", "Ha-Cont", "SII", "SII-Cont"}
VAMPIRES_FILTERS = set.union(VAMPIRES_STD_FILTERS, VAMPIRES_MBI_FILTERS, VAMPIRES_NB_FILTERS)

FILTERS = {
    "U": SpectralElement.from_filter("johnson_u"),
    "B": SpectralElement.from_filter("johnson_b"),
    "V": SpectralElement.from_filter("johnson_v"),
    "R": SpectralElement.from_filter("johnson_r"),
    "I": SpectralElement.from_filter("johnson_i"),
    # "G": SpectralElement.from_filter("johnson_v"),
    # "G_BP": SpectralElement.from_filter("johnson_v"),
    # "G_RP": SpectralElement.from_filter("johnson_v"),
    "J": SpectralElement.from_filter("johnson_j"),
    "H": SpectralElement.from_filter("bessel_h"),
    "K": SpectralElement.from_filter("johnson_k"),
}

VAMP_FILTERS = {f: load_vampires_filter(f) for f in VAMPIRES_FILTERS}


BASE_DIR = Path("/Users/mileslucas/dev/python/scexao_etc/data/pickles_uvk")  # TODO
TYPES = ("V", "IV", "III", "II", "I")


def prepare_pickles_dict(base_dir=BASE_DIR):
    tbl = fits.getdata(BASE_DIR / "pickles_uk.fits")
    fnames = (base_dir / f"{fname}.fits" for fname in tbl["FILENAME"])
    return dict(zip(tbl["SPTYPE"], fnames))


PICKLES_MAP = prepare_pickles_dict(BASE_DIR)
VEGASPEC = SourceSpectrum.from_vega()


def load_pickles(spec_type):
    filename = PICKLES_MAP[spec_type]
    tbl = fits.getdata(filename)
    sp = SourceSpectrum(
        Empirical1D,
        points=tbl["WAVELENGTH"] * u.angstrom,
        lookup_table=tbl["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
    )
    return sp


def color_correction(model, filt1: SpectralElement, filt2: SpectralElement):
    obs1 = Observation(model, filt1)
    obs2 = Observation(model, filt2)
    return obs2.effstim(VEGAMAG, vegaspec=VEGASPEC) - obs1.effstim(VEGAMAG, vegaspec=VEGASPEC)
