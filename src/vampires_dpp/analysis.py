import numpy as np
import sep
from astropy.io import fits

from vampires_dpp.image_processing import radial_profile_image, shift_frame
from vampires_dpp.indexing import (
    cutout_inds,
    frame_center,
    frame_radii,
    lamd_to_pixel,
    window_slices,
)
from vampires_dpp.psf_models import fit_model
from vampires_dpp.util import get_paths


def safe_aperture_sum(frame, r, center=None, ann_rad=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circle(
        np.ascontiguousarray(frame).astype("f4"),
        (center[1],),
        (center[0],),
        r,
        mask=mask,
        bkgann=ann_rad,
    )

    return flux[0]


def analyze_frame(
    frame, aper_rad, header=None, ann_rad=None, model="gaussian", recenter=True, **kwargs
):
    ## fit PSF to center
    inds = cutout_inds(frame, **kwargs)
    model_fit = fit_model(frame, inds, model)

    old_ctr = frame_center(frame)
    ctr = np.array((model_fit["y"], model_fit["x"]))
    if any(np.abs(old_ctr - ctr) > 10):
        ctr = old_ctr

    phot = safe_aperture_sum(frame, r=aper_rad, center=ctr, ann_rad=ann_rad)

    ## use PSF centers to recenter
    if recenter:
        offsets = frame_center(frame) - ctr
        frame = shift_frame(frame, offsets)

    # update header
    if header is not None:
        header["MODEL"] = model, "PSF model name"
        header["MOD_AMP"] = model_fit["amplitude"], "[adu] PSF model amplitude"
        header["MOD_X"] = model_fit["x"], "[px] PSF model x"
        header["MOD_Y"] = model_fit["y"], "[px] PSF model y"
        header["PHOTFLUX"] = phot, "[adu] Aperture photometry flux"
        header["PHOTRAD"] = aper_rad, "[px] Aperture photometry radius"
        header["MEDFLUX"] = np.nanmedian(frame), "[adu] Median frame flux"
        header["SUMFLUX"] = np.nansum(frame), "[adu] Total frame flux"
        header["PEAKFLUX"] = np.nanmax(frame), "[adu] Peak frame flux"

    return frame, header


def analyze_satspots_frame(
    frame,
    aper_rad,
    subtract_radprof=True,
    header=None,
    ann_rad=None,
    model="gaussian",
    recenter=True,
    **kwargs,
):
    ## subtract radial profile
    data = frame
    if subtract_radprof:
        profile = radial_profile_image(frame)
        data = frame - profile
    ## fit PSF to each satellite spot
    slices = window_slices(frame, **kwargs)
    N = len(slices)
    ave_x = ave_y = ave_amp = ave_flux = 0
    for sl in slices:
        model_fit = fit_model(data, sl, model)
        ave_x += model_fit["x"] / N
        ave_y += model_fit["y"] / N
        ave_amp += model_fit["amplitude"] / N
        phot = safe_aperture_sum(
            data, r=aper_rad, center=(model_fit["y"], model_fit["x"]), ann_rad=ann_rad
        )
        ave_flux += phot / N

    old_ctr = frame_center(frame)
    ctr = np.array((ave_y, ave_x))
    if any(np.abs(old_ctr - ctr) > 10):
        ctr = old_ctr

    ## use PSF centers to recenter
    if recenter:
        offsets = frame_center(frame) - ctr
        frame = shift_frame(frame, offsets)

    # update header
    if header is not None:
        header["MODEL"] = model, "PSF model name"
        header["MOD_AMP"] = ave_amp, "[adu] PSF model amplitude"
        header["MOD_X"] = ave_x, "[px] PSF model x"
        header["MOD_Y"] = ave_y, "[px] PSF model y"
        header["PHOTFLUX"] = ave_flux, "[adu] Aperture photometry flux"
        header["PHOTRAD"] = aper_rad, "[px] Aperture photometry radius"

    return frame, header


def analyze_file(filename, aper_rad, coronagraphic=False, force=False, **kwargs):
    path, outpath = get_paths(filename, suffix="analyzed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    frame, header = fits.getdata(path, header=True)

    if coronagraphic:
        kwargs["radius"] = lamd_to_pixel(kwargs["radius"], header["U_FILTER"])
        frame, header = analyze_satspots_frame(frame, aper_rad, header=header, **kwargs)
    else:
        frame, header = analyze_frame(frame, aper_rad, header=header, **kwargs)

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath
