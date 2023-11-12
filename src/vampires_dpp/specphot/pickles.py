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
        raise ValueError(
            f"""No pickles model found for sptype: {sptype}
        Please see the STScI pickles atlas documentation for info on available spectral types.
        List of available spectral types:\n{', '.join(tbl['SPTYPE'])}"""
        )


def load_pickles_model(sptype: str) -> SourceSpectrum:
    url = get_pickles_url(sptype)
    tbl = fits.getdata(download_file(url, cache=True))
    return SourceSpectrum(
        Empirical1D,
        points=tbl["WAVELENGTH"] * u.angstrom,
        lookup_table=tbl["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
    )
