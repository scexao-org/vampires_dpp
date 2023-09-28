import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astroquery.vizier import Vizier

from vampires_dpp.constants import PIXEL_SCALE, SUBARU_LOC


def apply_wcs(header, pxscale=PIXEL_SCALE, parang=0):
    nx = header["NAXIS1"]
    ny = header["NAXIS2"]

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [
        Angle(header["RA"], u.hourangle).degree,
        Angle(header["DEC"], u.degree).degree,
    ]
    w.wcs.cunit = ["deg", "deg"]
    cdelt = [-pxscale / 3.6e6, pxscale / 3.6e6]
    w.wcs.cdelt = cdelt
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ang = np.deg2rad(-parang)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    w.wcs.cd = [[cdelt[0] * cosang, -cdelt[1] * sinang], [cdelt[0] * sinang, cdelt[1] * cosang]]
    header.update(w.to_header())
    return header


def derotate_wcs(header, angle):
    ang = -np.deg2rad(angle)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    # store temporarily because inplace operations ruin calculation
    components = (
        cosang * header["PC1_1"] - sinang * header["PC2_1"],
        cosang * header["PC1_2"] - sinang * header["PC2_2"],
        sinang * header["PC1_1"] + cosang * header["PC2_1"],
        sinang * header["PC1_2"] + cosang * header["PC2_2"],
    )
    header["PC1_1"] = components[0]
    header["PC1_2"] = components[1]
    header["PC2_1"] = components[2]
    header["PC2_2"] = components[3]
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


def get_gaia_astrometry(target, catalog="dr3", radius=3):
    # get precise RA and DEC
    gaia_catalog_list = Vizier.query_object(
        target, radius=radius * u.arcsec, catalog=GAIA_CATALOGS[catalog.lower()]
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


def get_precise_coord(coord, time, scale="utc"):
    _time = Time(time, scale=scale, location=SUBARU_LOC)
    return coord.apply_space_motion(_time)
