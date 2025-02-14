import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.vizier import Vizier
from numpy.typing import NDArray

from .constants import SUBARU_LOC


def apply_wcs(image: NDArray, header: fits.Header, angle: float = 0):
    ny, nx = image.shape[-2:]
    # delete any CD keys from header
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        for subkey in ("", "B", "C", "D"):
            full_key = key + subkey
            if full_key in header:
                del header[full_key]

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [Angle(header["RA"], u.hourangle).degree, Angle(header["DEC"], u.degree).degree]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cdelt = [-header["PXSCALE"] / 3.6e6, header["PXSCALE"] / 3.6e6]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ang = np.deg2rad(angle)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    # work around fact PC doesn't get written if unit
    w.wcs.pc = [[cosang, -sinang], [sinang, cosang]]
    header.update(w.to_header())
    return header


def get_coord_header(header, time=None):
    coord = SkyCoord(
        ra=header["RA"],
        dec=header["DEC"],
        unit=(u.hourangle, u.deg),
        frame=header["RADESYS"].lower(),
        equinox=f"J{header['EQUINOX']}",
        obstime=time,
    )
    return coord


GAIA_CATALOGS = {"dr1": "I/337/gaia", "dr2": "I/345/gaia2", "dr3": "I/355/gaiadr3"}


def get_gaia_astrometry(target: str, catalog="dr3", radius=1):
    """
    Get coordinate from GAIA catalogue with proper motions and parallax

    Radius is in arcminute.
    """
    # get precise RA and DEC
    gaia_catalog_list = Vizier.query_object(
        target, radius=radius * u.arcminute, catalog=GAIA_CATALOGS[catalog.lower()]
    )
    if len(gaia_catalog_list) == 0:
        return None
    gaia_info = gaia_catalog_list[0][0]  # first row of first table
    plx = np.abs(gaia_info["Plx"]) * u.mas
    coord = SkyCoord(
        ra=gaia_info["RA_ICRS"] * u.deg,
        dec=gaia_info["DE_ICRS"] * u.deg,
        pm_ra_cosdec=gaia_info["pmRA"] * u.mas / u.year,
        pm_dec=gaia_info["pmDE"] * u.mas / u.year,
        distance=plx.to(u.parsec, equivalencies=u.parallax()),
        frame="icrs",
        obstime="J2016",
    )
    return coord


def get_precise_coord(coord: SkyCoord, time: str, scale="utc"):
    """Use astropy to get proper-motion corrected coordinate"""
    _time = Time(time, scale=scale, location=SUBARU_LOC)
    return coord.apply_space_motion(_time)
