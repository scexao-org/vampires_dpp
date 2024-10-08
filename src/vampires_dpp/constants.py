from typing import ClassVar, Final, Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from pydantic import BaseModel

from .util import wrap_angle

# Subaru location
SUBARU_LOC: Final[EarthLocation] = EarthLocation(lat=19.825504 * u.deg, lon=-155.4760187 * u.deg)


class InstrumentInfo(BaseModel):
    @property
    def pa_offset(self):
        pap_offset = -39
        return wrap_angle(self.pupil_offset - 180 + pap_offset)  # deg


class EMCCDVAMPIRES(InstrumentInfo):
    cam_num: Literal[1, 2]
    emgain: int
    gain: ClassVar[float] = 4.5  # e-/adu
    dark_current: ClassVar[float] = 1.5e-4  # e-/s/px
    bias: ClassVar[int] = 150

    filters: ClassVar[set[str]] = {
        "Open",
        "625-50",
        "675-50",
        "725-50",
        "750-50",
        "775-50",
        "Halpha",
        "Ha-Cont",
    }
    PIXEL_SCALE: ClassVar[dict[int, float]] = {1: 6.24, 2: 6.24}  # mas / px
    PUPIL_OFFSET: ClassVar[dict[int, float]] = {1: 140.4, 2: 140.4}  # deg

    @property
    def pixel_scale(self):
        return self.PIXEL_SCALE[self.cam_num]

    @property
    def pupil_offset(self):
        return wrap_angle(self.PUPIL_OFFSET[self.cam_num])

    @property
    def fullwell(self) -> float:
        fullwell = 180000 if self.emgain == 0 else min(800000, 2**16 * self.gain)
        return float(fullwell)

    @property
    def effgain(self) -> float:
        return self.gain / max(self.emgain, 1)

    @property
    def excess_noise_factor(self) -> float:
        return 1 if self.emgain == 0 else np.sqrt(2)

    @property
    def readnoise(self) -> float:
        # read noise is 89 e- in EM register,
        # 9.6 e- in conventional register
        return 9.6 if self.emgain == 0 else 89

    def get_psf_size(self, filt_name):
        if filt_name not in self.filters:
            msg = f"{filt_name} not recognized as a filter"
            raise ValueError(msg)
        ## TODO
        raise NotImplementedError()


class CMOSVAMPIRES(InstrumentInfo):
    cam_num: Literal[1, 2]
    readmode: Literal["fast", "slow"]
    dark_current: ClassVar[float] = 3.6e-3  # e-/s/px
    excess_noise_factor: ClassVar[float] = 1
    bias: ClassVar[int] = 200

    filters: ClassVar[set[str]] = {
        "Open",
        "625-50",
        "675-50",
        "725-50",
        "750-50",
        "775-50",
        "F610",
        "F670",
        "F720",
        "F760",
        "Halpha",
        "Ha-Cont",
        "SII",
        "SII-Cont",
    }

    VAMP_RN: ClassVar[dict[tuple[str, int], float]] = {
        ("fast", 1): 0.403,
        ("fast", 2): 0.399,
        ("slow", 1): 0.245,
        ("slow", 2): 0.220,
    }
    VAMP_GAIN: ClassVar[dict[str, float]] = {"fast": 0.103, "slow": 0.105}
    PIXEL_SCALE: ClassVar[dict[int, float]] = {1: 5.91, 2: 5.895}  # mas / px
    PUPIL_OFFSET: ClassVar[dict[int, float]] = {1: -38.40, 2: -38.58}  # deg

    @property
    def pixel_scale(self):
        return self.PIXEL_SCALE[self.cam_num]

    @property
    def pupil_offset(self):
        return wrap_angle(self.PUPIL_OFFSET[self.cam_num])

    @property
    def readnoise(self):
        return self.VAMP_RN[(self.readmode, self.cam_num)]

    @property
    def gain(self):
        return self.VAMP_GAIN[self.readmode]

    @property
    def effgain(self):
        return self.gain

    @property
    def fullwell(self):
        # 7000 e- from manual, but 2**16 is almost always lower
        return min(7000, 2**16 * self.gain)  # e-

    def get_psf_size(self, filt_name):
        if filt_name not in self.filters:
            msg = f"{filt_name} not recognized as a filter"
            raise ValueError(msg)
        ## TODO
        raise NotImplementedError()
