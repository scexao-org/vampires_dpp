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

sptype_chars: Final = ["O", "B", "A", "F", "G", "K", "M", "L", "T", "Y", "W"]

spectral_type_pattern = re.compile(
    r"""
    ([wr])?
    ([OBAFGKMLTYW])          # Spectral class (e.g., O, B, A, F, G, K, M, L, T, Y, W)
    (\d{1,2})                 # Optional subclass (e.g., 0, 1, 2, ..., 9)
    ([IV]{1,3}|0|Ia|Ib|II|III|IV|V|VI|VII)?  # Optional luminosity class (e.g., I, II, III, IV, V, or 0)
""",
    re.VERBOSE,
)


def parse_spectral_type(spectral_type) -> dict | None:
    """
    Parse a stellar spectral type into its components.

    :param spectral_type: The stellar spectral type to parse (e.g., 'G2V')
    :return: A dictionary with the parsed components, or None if not matched
    """
    match = spectral_type_pattern.match(spectral_type)
    if match:
        return match.groups()
    else:
        return None


def check_spectral_type_in_pickles(sptype_to_check: str):
    sptuple_to_check = parse_spectral_type(sptype_to_check)
    if sptuple_to_check is None:
        return False
    tbl = fits.getdata(download_file(PICKLES_MAP_URL, cache=True))
    close_matches = []
    for sptype in tbl["SPTYPE"]:
        sptuple = parse_spectral_type(sptype)
        if sptuple_to_check == sptuple:
            return True
        elif (
            sptuple_to_check[3] == sptuple[3]
            and abs(sptype_chars.index(sptuple_to_check[1]) - sptype_chars.index(sptuple[1])) <= 1
        ):
            close_matches.append(sptuple)

    print(
        f"Could not find a matching model for the '{sptype_to_check}' spectral type in the pickles_uvk library"
    )
    if len(close_matches) > 0:
        print("We found the following close substitutes:")
        print(" - " + ", ".join("".join(str(c) for c in m if c is not None) for m in close_matches))
    return False
