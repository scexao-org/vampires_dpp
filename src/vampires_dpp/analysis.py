import numpy as np
import sep
from astropy.io import fits

from vampires_dpp.image_processing import shift_frame
from vampires_dpp.indexing import cutout_slice, frame_center
from vampires_dpp.psf_models import fit_model
from vampires_dpp.util import get_paths


def safe_aperture_sum(frame, r, center=None, ann_rad=None):
    if center is None:
        center = frame_center(frame)
    mask = ~np.isfinite(frame)
    flux, fluxerr, flag = sep.sum_circle(
        np.ascontiguousarray(frame), (center[1],), (center[0],), r, mask=mask, bkgann=ann_rad
    )

    return flux[0]


def analyze_file(
    filename, aper_rad, ann_rad=None, force=False, model="gaussian", recenter=True, **kwargs
):
    path, outpath = get_paths(filename, suffix="analyzed", **kwargs)
    if not force and outpath.is_file() and path.stat().st_mtime < outpath.stat().st_mtime:
        return outpath

    frame, header = fits.getdata(path, header=True)

    ## fit PSF to center
    inds = cutout_slice(frame, window=30)
    model_fit = fit_model(frame, inds, model)

    # update header
    header["DPP_MOD"] = model, "PSF model name"
    header["DPP_MODA"] = model_fit["amplitude"], "PSF model amplitude"
    header["DPP_MODX"] = model_fit["x"], "[px] PSF model x"
    header["DPP_MODY"] = model_fit["y"], "[px] PSF model y"

    ## use PSF centers to recenter
    if recenter:
        ctr = frame_center(frame)
        offsets = ctr[0] - header["DPP_MODY"], ctr[1] - header["DPP_MODX"]
        frame = shift_frame(frame, offsets)
    else:
        ctr = np.array((header["DPP_MODY"], header["DPP_MODX"]))
    phot = safe_aperture_sum(frame, r=aper_rad, center=ctr, ann_rad=ann_rad)
    header["DPP_PHOT"] = phot, "[adu] Aperture photometry flux"
    header["DPP_PHOR"] = aper_rad, "[px] Aperture photometry radius"

    # print(f"cr={model_fit['y']} cc={model_fit['x']} amp={model_fit['amplitude']} flux={phot}")

    fits.writeto(outpath, frame, header=header, overwrite=True)
    return outpath
