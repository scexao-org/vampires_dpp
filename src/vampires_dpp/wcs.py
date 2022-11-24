from astropy import wcs
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
import numpy as np


def apply_wcs(header, pxscale=6.24, pupil_offset=140.4):
    nx = header["NAXIS1"]
    ny = header["NAXIS2"]

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [
        Angle(header["RA"], u.hourangle).degree,
        Angle(header["DEC"], u.degree).degree,
    ]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cdelt = [-pxscale / 3.6e6, pxscale / 3.6e6]
    w.wcs.ctype = ["RA", "DEC"]
    ang = np.deg2rad(header["D_IMRPAD"] + pupil_offset)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    w.wcs.pc = [[cosang, -sinang], [sinang, cosang]]
    header.update(w.to_header())
    return header


def derotate_wcs(header, angle):
    ang = -np.deg2rad(angle)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    header["PC1_1"] += cosang
    header["PC1_2"] += -sinang
    header["PC2_1"] += sinang
    header["PC2_2"] += cosang
    return header


def apply_wcs_file(filename, output=None, skip=False):
    pass
