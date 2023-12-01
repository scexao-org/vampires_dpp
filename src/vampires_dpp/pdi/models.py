from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.data import download_file
from numpy.typing import NDArray
from pydantic import BaseModel

from . import mueller_matrices as mm

MM_KEY = "1x0sUawjWmySrsnSQ70VRYfewpgXAcyQMyrt6eley0rE"
MM_URL = f"https://docs.google.com/spreadsheets/d/{MM_KEY}/gviz/tq?tqx=out:csv&sheet=downloadable"


def load_calibration_file(header):
    table = pd.read_csv(
        download_file(MM_URL, cache=True), header=0, index_col=0, dtype={"filter": str}
    )
    filt = header["FILTER01"]
    # closest match to Open is 675
    table_key = "675" if filt == "Open" else filt.replace("-50", "")
    return table.loc[table_key]


class EMCCDMuellerMatrix(BaseModel):
    name: str = "ideal"
    pbs_ratio: float = 1  # cam1 / cam2 ratio
    hwp_offset: float = 0  # deg
    hwp_phi: float = 0.5  # wave
    imr_offset: float = 0  # deg
    imr_phi: float = 0.5  # wave
    optics_diat: float = 0
    optics_theta: float = 0  # deg
    optics_phi: float = 0  # wave
    flc_theta: dict[str, float] = {"A": 0, "B": 45}  # deg
    flc_phi: float = 0.5  # wave

    @classmethod
    def load_calib_file(cls, header):
        table = load_calibration_file(header)
        flc_theta = {k: t + table["flc_theta"] for k, t in zip(("A", "B"), (0, 45), strict=True)}
        return cls(
            name=table.name,
            pbs_ratio=table["emgain"],
            hwp_offset=table["hwp_delta"],
            hwp_phi=table["hwp_phi"],
            imr_offset=table["imr_delta"],
            imr_phi=table["imr_phi"],
            optics_diat=table["optics_diat"],
            optics_theta=table["optics_theta"],
            optics_phi=table["optics_phi"],
            flc_theta=flc_theta,
            flc_phi=table["flc_phi"],
        )

    def evaluate(self, header: fits.Header, hwp_adi_sync: bool = True) -> NDArray:
        ## build up mueller matrix component by component

        # telescope
        alt = np.deg2rad(header["ALTITUDE"])
        M = mm.rotator(np.deg2rad(header["PA"]))
        M = mm.mirror() @ M
        M = mm.rotator(-alt) @ M

        # HWP
        # get adi sync offset
        if hwp_adi_sync:
            az = np.deg2rad(header["AZIMUTH"] - 180)
            hwp_sync_offset = mm.hwp_adi_sync_offset(alt=alt, az=az)
        else:
            hwp_sync_offset = 0
        # add instrumental offset
        hwp_theta = np.deg2rad(header["RET-ANG1"] + hwp_sync_offset + self.hwp_offset)
        M = mm.waveplate(hwp_theta, self.hwp_phi * 2 * np.pi) @ M

        # Image rotator
        imr_theta = np.deg2rad(header["D_IMRANG"] + self.imr_offset)
        M = mm.waveplate(imr_theta, self.imr_phi * 2 * np.pi) @ M

        # SCExAO optics
        M = (
            mm.generic(
                epsilon=self.optics_diat,
                theta=np.deg2rad(self.optics_theta),
                delta=self.optics_phi * 2 * np.pi,
            )
            @ M
        )

        # FLC
        flc_theta = np.deg2rad(self.flc_theta[header["U_FLC"]])
        M = mm.waveplate(flc_theta, self.flc_phi * 2 * np.pi) @ M

        # beamsplitter
        is_ordinary = header["U_CAMERA"] == 1
        M = mm.wollaston(is_ordinary) @ M
        if is_ordinary:
            M *= self.pbs_ratio

        return M.astype("f4")


class CMOSMuellerMatrix(BaseModel):
    name: str = "ideal"
    pbs_ratio: float = 1  # cam1 / cam2 ratio
    hwp_offset: float = 0  # deg
    hwp_phi: float = 0.5  # wave
    imr_offset: float = 0  # deg
    imr_phi: float = 0.5  # wave
    optics_diat: float = 0
    optics_theta: float = 0  # deg
    optics_phi: float = 0  # wave
    flc_theta: dict[str, float] = {"A": 0, "B": -45}  # deg
    flc_phi: float = 0.5  # wave

    def evaluate(self, header: fits.Header, hwp_adi_sync: bool = True) -> NDArray:
        ## build up mueller matrix component by component

        # telescope
        alt = np.deg2rad(header["ALTITUDE"])
        M = mm.rotator(np.deg2rad(header["PA"]))
        M = mm.mirror() @ M
        M = mm.rotator(-alt) @ M

        # HWP
        # get adi sync offset
        if hwp_adi_sync:
            az = np.deg2rad(header["AZIMUTH"] - 180)
            hwp_sync_offset = mm.hwp_adi_sync_offset(alt=alt, az=az)
        else:
            hwp_sync_offset = 0
        # add instrumental offset
        hwp_theta = np.deg2rad(header["RET-ANG1"] + hwp_sync_offset + self.hwp_offset)
        M = mm.waveplate(hwp_theta, self.hwp_phi * 2 * np.pi) @ M

        # Image rotator
        imr_theta = np.deg2rad(header["D_IMRANG"] + self.imr_offset)
        M = mm.waveplate(imr_theta, self.imr_phi * 2 * np.pi) @ M

        # SCExAO optics
        optics_mm = mm.generic(
            epsilon=self.optics_diat,
            theta=np.deg2rad(self.optics_theta),
            delta=self.optics_phi * 2 * np.pi,
        )
        M = optics_mm @ M

        # FLC
        flc_theta = np.deg2rad(self.flc_theta[header["U_FLC"]])
        M = mm.waveplate(flc_theta, self.flc_phi * 2 * np.pi) @ M

        # beamsplitter
        is_ordinary = header["U_CAMERA"] == 1
        M = mm.wollaston(is_ordinary) @ M
        if is_ordinary:
            M *= self.pbs_ratio

        return M.astype("f4")


def mueller_matrix_from_file(
    filename, outpath, force=False, hwp_adi_sync: bool = True, ideal: bool = False
):
    if (
        not force
        and Path(outpath).exists()
        and Path(filename).stat().st_mtime < Path(outpath).stat().st_mtime
    ):
        return outpath

    headers = []
    mms = []
    with fits.open(filename) as hdul:
        for hdu in hdul[1:]:
            headers.append(hdu.header)
            if "U_MBI" in hdu.header:
                if ideal:
                    mm_model = CMOSMuellerMatrix()
                else:
                    msg = "No calibrations completed for upgraded VAMPIRES, yet."
                    raise NotImplementedError(msg)
            else:
                if ideal:
                    mm_model = EMCCDMuellerMatrix()
                else:
                    mm_model = EMCCDMuellerMatrix.load_calib_file(hdu.header)
            mms.append(mm_model.evaluate(hdu.header, hwp_adi_sync=hwp_adi_sync))
        prim_hdu = fits.PrimaryHDU(np.array(mms), hdul[0].header)
    hdus = (fits.ImageHDU(cube, hdr) for cube, hdr in zip(mms, headers, strict=True))
    hdul = fits.HDUList([prim_hdu, *hdus])
    hdul.writeto(outpath, overwrite=True)
    return outpath
