from collections.abc import Iterable

from astropy.table import QTable
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


def get_simbad_table(obj: str) -> QTable:
    sim = Simbad()
    sim.remove_votable_fields("coordinates")
    sim.add_votable_fields("sptype")
    return sim.query_object(obj)


def get_ucac_table(obj: str, radius=1) -> QTable:
    cat = "I/322A/out"
    viz = Vizier(catalog=cat, columns=["Vmag", "rmag", "imag"])
    # get precise RA and DEC
    ucac4_list = viz.query_object(
        obj,
        radius=radius * u.arcsec,
    )
    return ucac4_list[0]


def get_ucac_flux(table: QTable, preference: Iterable[str] = ("r", "i", "V")) -> tuple(float, str):
    for band in preference:
        key = f"{band}mag"
        if not table[key].mask[0]:
            return table[key][0], band
