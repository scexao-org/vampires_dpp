from functools import partial

import numpy as np

from vampires_dpp.analysis import safe_aperture_sum


def measure_star_pol(stokes_cube, stokes_err, header, field, aper_rad, ann_rad=None):
    phot_func = partial(safe_aperture_sum, r=aper_rad, ann_rad=ann_rad)
    IQ_phot, IQ_phot_err = phot_func(stokes_cube[0], err=stokes_err[0])
    Q_phot, Q_phot_err = phot_func(stokes_cube[1], err=stokes_err[1])
    IU_phot, IU_phot_err = phot_func(stokes_cube[2], err=stokes_err[2])
    U_phot, U_phot_err = phot_func(stokes_cube[3], err=stokes_err[3])

    dolp = np.hypot(Q_phot / IQ_phot, U_phot / IU_phot)
    # Partial derivatives
    d_dolp_dQ = Q_phot / (IQ_phot**2 * dolp)
    d_dolp_dU = U_phot / (IU_phot**2 * dolp)
    d_dolp_dIQ = -(Q_phot**2) / (IQ_phot**3 * dolp)
    d_dolp_dIU = -(U_phot**2) / (IU_phot**3 * dolp)
    # Propagated error
    dolp_err = np.sqrt(
        (d_dolp_dQ * Q_phot_err) ** 2
        + (d_dolp_dU * U_phot_err) ** 2
        + (d_dolp_dIQ * IQ_phot_err) ** 2
        + (d_dolp_dIU * IU_phot_err) ** 2
    )

    aolp = np.arctan2(U_phot, Q_phot)
    # Partial derivatives
    d_aolp_dQ = -U_phot / (Q_phot**2 + U_phot**2)
    d_aolp_dU = Q_phot / (Q_phot**2 + U_phot**2)
    # Propagated error
    aolp_err = np.sqrt((d_aolp_dQ * Q_phot_err) ** 2 + (d_aolp_dU * U_phot_err) ** 2)

    unit = header["BUNIT"]
    header[f"hierarch DPP PDI IQ FLUX {field}"] = IQ_phot, f"[{unit}] Phot. flux in IQ frame"
    header[f"hierarch DPP PDI IQ FLUX_ERR {field}"] = (
        IQ_phot_err,
        f"[{unit}] Phot. flux err in IQ frame",
    )
    header[f"hierarch DPP PDI Q FLUX {field}"] = Q_phot, f"[{unit}] Phot. flux in Q frame"
    header[f"hierarch DPP PDI Q FLUX_ERR {field}"] = (
        Q_phot_err,
        f"[{unit}] Phot. flux err in Q frame",
    )
    header[f"hierarch DPP PDI IU FLUX {field}"] = IU_phot, f"[{unit}] Phot. flux in IU frame"
    header[f"hierarch DPP PDI IU FLUX_ERR {field}"] = (
        IU_phot_err,
        f"[{unit}] Phot. flux err in IU frame",
    )
    header[f"hierarch DPP PDI U FLUX {field}"] = U_phot, f"[{unit}] Phot. flux in U frame"
    header[f"hierarch DPP PDI U FLUX_ERR {field}"] = (
        U_phot_err,
        f"[{unit}] Phot. flux err in U frame",
    )
    header[f"hierarch DPP PDI DOLP {field}"] = dolp, "Degree of linearly polarized flux"
    header[f"hierarch DPP PDI DOLP_ERR {field}"] = dolp_err, "Degree err of linearly polarized flux"
    header[f"hierarch DPP PDI AOLP {field}"] = aolp, "[deg] Angle of linearly polarized flux"
    header[f"hierarch DPP PDI AOLP_ERR {field}"] = aolp_err, "Angle err of linearly polarized flux"
    return header


def add_star_pol_hdul(hdul, aper_rad, ann_rad=None):
    stokes_cube = hdul[0].data
    stokes_err = hdul["ERR"].data
    header = hdul[0].header
    fields = [hdu.header["FIELD"] for hdu in hdul[2:]]
    for wl_idx, field in enumerate(fields):
        header = measure_star_pol(
            stokes_cube[wl_idx],
            stokes_err[wl_idx],
            header,
            field=field,
            aper_rad=aper_rad,
            ann_rad=ann_rad,
        )

    for hdu in hdul:
        hdu.header.update(header)

    return hdul
