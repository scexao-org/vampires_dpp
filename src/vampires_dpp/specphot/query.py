from collections.abc import Iterable

import astropy.units as u
from astropy.table import QTable
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


def get_simbad_table(obj: str) -> QTable:
    sim = Simbad()
    sim.add_votable_fields("sptype", "flux(V)", "flux(R)", "flux(I)", "flux(r)", "flux(i)")
    result = sim.query_object(obj)
    return result


def get_simbad_flux(
    table: QTable, preference: Iterable[str] = ("r", "R", "i", "I", "V")
) -> tuple[float, str] | None:
    for band in preference:
        key = f"FLUX_{band}"
        if not table[key].mask[0]:
            if band in ("R", "I"):
                band = band.lower()
            return table[key][0], band
    return None


def get_ucac_table(obj: str, radius=1) -> QTable:
    cat = "I/322A/out"
    viz = Vizier(catalog=cat, columns=["Vmag", "rmag", "imag"])
    # get precise RA and DEC
    ucac4_list = viz.query_object(obj, radius=radius * u.arcsec)
    return ucac4_list[0]


def get_ucac_flux(
    table: QTable, preference: Iterable[str] = ("r", "i", "V")
) -> tuple[float, str] | None:
    for band in preference:
        key = f"{band}mag"
        if not table[key].mask[0]:
            return table[key][0], band

    return None
