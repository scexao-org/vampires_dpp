import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.utils.data import download_file
from numpy.typing import NDArray
from pydantic import BaseModel

from vampires_dpp.headers import sort_header

from . import mueller_matrices as mm

MM_KEY = "1i8TjHzQFMmxaUWrrqm1eYziyUanC6pweGGFzJPdfbiE"
MM_URL = f"https://docs.google.com/spreadsheets/d/{MM_KEY}/gviz/tq?tqx=out:csv&sheet=downloadable2"


def _clear_file():
    filename = download_file(MM_URL, cache=True)
    path = Path(filename)
    if path.exists():
        path.unlink()


# MBI_MM_DICT: Final[dict[str, str]] = {"F610": "625", "F670": "675", "F720": "725", "F760": "750"}


def load_calibration_file(header):
    table = pd.read_csv(
        download_file(MM_URL, cache=True), header=0, index_col=0, dtype={"filter": str}
    )
    filt = header["FILTER01"]
    if "MBI" in header["OBS-MOD"]:
        # table_key = MBI_MM_DICT[header["FIELD"]] if "FIELD" in header else "675"
        table_key = header["FIELD"]
    else:
        # closest match to Open is 675
        table_key = "675" if filt == "Open" else filt.replace("-50", "")

    return table.loc[table_key]


class VAMPIRESMuellerMatrix(BaseModel):
    name: str = "ideal"
    m3_diat: float = 0
    m3_offset: float = 0  # deg
    hwp_offset: float = 0  # deg
    hwp_phi: float = 0.5  # wave
    imr_offset: float = 0  # deg
    imr_phi: float = 0.5  # wave
    optics_diat: float = 0
    optics_theta: float = 0  # deg
    optics_phi: float = 0  # wave
    flc_theta: dict[str, float] = {"A": 0, "B": 45}  # deg
    flc_phi: float = 0.5  # wave

    def common_path_mm(self, pa, alt, az, hwp, imr, hwp_adi_sync=True):
        # telescope
        pa_theta = np.deg2rad(pa)
        m3_theta = np.deg2rad(self.m3_offset)
        m3 = mm.generic(epsilon=self.m3_diat, theta=m3_theta, delta=np.pi)
        alt_theta = np.deg2rad(alt)
        tel_mm = mm.rotator(-alt_theta) @ m3 @ mm.rotator(pa_theta)

        # HWP
        if hwp_adi_sync:
            az_theta = np.deg2rad(az - 180)
            hwp_adi_offset_rad = mm.hwp_adi_sync_offset(alt=alt_theta, az=az_theta)
        else:
            hwp_adi_offset_rad = 0
        hwp_theta = np.deg2rad(hwp + self.hwp_offset) + hwp_adi_offset_rad
        hwp_mm = mm.waveplate(hwp_theta, self.hwp_phi * 2 * np.pi)

        # Image rotator
        imr_theta = np.deg2rad(imr + self.imr_offset)
        imr_mm = mm.waveplate(imr_theta, self.imr_phi * 2 * np.pi)

        # SCExAO optics
        optics_mm = mm.generic(
            epsilon=self.optics_diat,
            theta=np.deg2rad(self.optics_theta),
            delta=self.optics_phi * 2 * np.pi,
        )
        return optics_mm @ imr_mm @ hwp_mm @ tel_mm

    def __call__(self, flc_state: str, camera: int, *args, **kwargs) -> NDArray:
        ## build up mueller matrix component by component
        cp_mm = self.common_path_mm(*args, **kwargs)

        # FLC
        flc_theta = np.deg2rad(self.flc_theta[flc_state])
        flc_mm = mm.waveplate(flc_theta, self.flc_phi * 2 * np.pi)

        # beamsplitter - horizontal/ordinary to camera 1
        is_ordinary = camera == 1
        pbs_mm = mm.wollaston(is_ordinary)

        M = pbs_mm @ flc_mm @ cp_mm
        return M.astype("f4")

    def from_header(self, header: fits.Header, **kwargs) -> NDArray:
        return self(
            pa=header["PA"],
            alt=header["ALTITUDE"],
            az=header["AZIMUTH"],
            hwp=header["RET-ANG1"],
            imr=header["D_IMRANG"],
            flc_state=header["U_FLC"],
            camera=header["U_CAMERA"],
            **kwargs,
        )


class EMCCDMuellerMatrix(VAMPIRESMuellerMatrix):
    @classmethod
    def load_calib_file(cls, header):
        table = load_calibration_file(header)
        flc_theta = {k: t + table["flc_theta"] for k, t in zip(("A", "B"), (0, 45), strict=True)}
        return cls(
            name=table.name,
            m3_diat=table["m3_diat"],
            m3_offset=table["m3_theta"],
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


class CMOSMuellerMatrix(VAMPIRESMuellerMatrix):
    flc_theta: dict[str, float] = {"A": 0, "B": 43}  # deg
    dichroic_diat: float = 0
    dichroic_theta: float = 0  # deg
    dichroic_phi: float = 0  # wave

    def __call__(self, use_flc: bool, flc_state: str, camera: int, *args, **kwargs) -> NDArray:
        ## build up mueller matrix component by component
        cp_mm = self.common_path_mm(*args, **kwargs)

        # FLC
        if use_flc:
            flc_theta = np.deg2rad(self.flc_theta[flc_state])
            flc_mm = mm.waveplate(flc_theta, self.flc_phi * 2 * np.pi)
        else:
            flc_mm = np.eye(4)

        dichroic_mm = mm.generic(
            np.deg2rad(self.dichroic_theta),
            self.dichroic_diat,
            np.deg2rad(self.dichroic_phi) * 2 * np.pi,
        )

        # beamsplitter
        is_ordinary = camera == 1
        pbs_mm = mm.wollaston(is_ordinary)

        M = pbs_mm @ dichroic_mm @ flc_mm @ cp_mm
        return M.astype("f4")

    def from_header(self, header: fits.Header, hwp_adi_sync: bool = True) -> NDArray:
        ## build up mueller matrix component by component
        actual_hwp_adi_sync = header["RET-MOD1"].strip() == "SYNCHRO_ADI"
        if hwp_adi_sync != actual_hwp_adi_sync:
            msg = f"You set HWP ADI sync to {hwp_adi_sync!r} but RET-MOD1 was {header['RET-MOD1'].strip()!r}"
            warnings.warn(msg, stacklevel=2)

        return self(
            pa=header["PA"],
            alt=header["ALTITUDE"],
            az=header["AZIMUTH"],
            hwp=header["RET-ANG1"],
            imr=header["D_IMRANG"],
            use_flc=header["U_FLCST"].strip() == "IN",
            flc_state=header["U_FLC"],
            camera=header["U_CAMERA"],
            hwp_adi_sync=hwp_adi_sync,
        )

    @classmethod
    def load_calib_file(cls, header):
        table = load_calibration_file(header)
        return cls(
            name=table.name,
            m3_diat=table["m3_diat"],
            m3_offset=table["m3_theta"],
            hwp_offset=table["hwp_delta"],
            hwp_phi=table["hwp_phi"],
            imr_offset=table["imr_delta"],
            imr_phi=table["imr_phi"],
            optics_diat=table["optics_diat"],
            optics_theta=table["optics_theta"],
            optics_phi=table["optics_phi"],
            dichroic_diat=table["dichroic_diat"],
            dichroic_theta=table["dichroic_theta"],
            dichroic_phi=table["dichroic_phi"],
        )


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
        for hdu in hdul[2:]:
            headers.append(hdu.header)
            if "U_MBI" in hdu.header:
                if ideal:
                    mm_model = CMOSMuellerMatrix()
                else:
                    mm_model = CMOSMuellerMatrix.load_calib_file(hdu.header)
            else:
                if ideal:
                    mm_model = EMCCDMuellerMatrix()
                else:
                    mm_model = EMCCDMuellerMatrix.load_calib_file(hdu.header)
            mms.append(mm_model.from_header(hdu.header, hwp_adi_sync=hwp_adi_sync))
        prim_hdu = fits.PrimaryHDU(np.array(mms), hdul[0].header)
    hdus = (fits.ImageHDU(cube, sort_header(hdr)) for cube, hdr in zip(mms, headers, strict=True))
    hdul = fits.HDUList([prim_hdu, *hdus])
    hdul.writeto(outpath, overwrite=True)
    return outpath
