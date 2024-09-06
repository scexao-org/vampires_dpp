import re
from typing import Final

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from synphot import Empirical1D, SourceSpectrum

__all__ = ["load_pickles_model"]

PICKLES_URL: Final[
    str
] = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/pickles/dat_uvk/"
PICKLES_MAP_URL: Final[str] = PICKLES_URL + "pickles_uk.fits"


def get_pickles_url(sptype: str) -> str:
    tbl = fits.getdata(download_file(PICKLES_MAP_URL, cache=True))
    if sptype in tbl["SPTYPE"]:
        mask = np.where(tbl["SPTYPE"] == sptype)[0][0]
        return PICKLES_URL + f"{tbl[mask]['FILENAME']}.fits"
    else:
        msg = f"No pickles model found for sptype: {sptype}\n        Please see the STScI pickles atlas documentation for info on available spectral types.\n        List of available spectral types:\n{', '.join(tbl['SPTYPE'])}"
        raise ValueError(msg)


def load_pickles_model(sptype: str) -> SourceSpectrum:
    url = get_pickles_url(sptype)
    tbl = fits.getdata(download_file(url, cache=True))
    return SourceSpectrum(
        Empirical1D,
        points=tbl["WAVELENGTH"] * u.angstrom,
        lookup_table=tbl["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
    )


# Regex pattern to match stellar spectral types
SPTYPE_CHARS: Final = ("O", "B", "A", "F", "G", "K", "M", "L", "T", "Y", "W")

SPECTRAL_TYPE_RE: Final = re.compile(
    r"""
    ([wr])?
    ([OBAFGKMLTYW])          # Spectral class (e.g., O, B, A, F, G, K, M, L, T, Y, W)
    (\d{1,2})                 # Optional subclass (e.g., 0, 1, 2, ..., 9)
    ([IV]{1,3}|0|Ia|Ib|II|III|IV|V|VI|VII)?  # Optional luminosity class (e.g., I, II, III, IV, V, or 0)
""",
    re.VERBOSE,
)


def check_spectral_type_in_pickles(sptype_to_check: str) -> bool:
    match = SPECTRAL_TYPE_RE.match(sptype_to_check)
    if match is None:
        print(f"Failed to parse spectral type {sptype_to_check}")
        return False
    sptuple_to_check = match.groups()
    tbl = fits.getdata(download_file(PICKLES_MAP_URL, cache=True))
    close_matches = []
    for sptype in tbl["SPTYPE"]:
        match = SPECTRAL_TYPE_RE.match(sptype)
        assert match is not None, f"Couldn't parse pickles SPTYPE {sptype}"
        sptuple = match.groups()
        if sptuple_to_check == sptuple:
            return True
        elif (
            sptuple_to_check[3] == sptuple[3]
            and abs(SPTYPE_CHARS.index(sptuple_to_check[1]) - SPTYPE_CHARS.index(sptuple[1])) <= 1
        ):
            close_matches.append(sptuple)

    print(
        f"! Could not find a matching model for the '{sptype_to_check}' spectral type in the pickles_uvk library"
    )
    if len(close_matches) > 0:
        print("We found the following close substitutes:")
        print(" - " + ", ".join("".join(str(c) for c in m if c is not None) for m in close_matches))
    return False
