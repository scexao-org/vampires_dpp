from multiprocessing import cpu_count
from typing import ClassVar, Final, Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from pydantic import BaseModel

# limit default nproc since many operations are
# throttled by file I/O
DEFAULT_NPROC = min(cpu_count(), 8)
# Subaru location
SUBARU_LOC: Final[EarthLocation] = EarthLocation(lat=19.825504 * u.deg, lon=-155.4760187 * u.deg)


class InstrumentInfo(BaseModel):
    @property
    def pa_offset(self):
        return self.pupil_offset - 180 - 39  # deg


class EMCCDVAMPIRES(InstrumentInfo):
    cam_num: Literal[1, 2]
    emgain: int
    pixel_scale: ClassVar[float] = 6.24  # mas / px
    pupil_offset: ClassVar[float] = 140.4  # deg
    readnoise: ClassVar[float] = 82  # e-
    gain: ClassVar[float] = 4.5  # e-/adu

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

    @property
    def fullwell(self):
        if self.emgain == 0:
            # with EM gain off, using 180K register
            fullwell = 180_000  # e-
        else:
            # with EM gain on, limited by 16-bit register almost always
            fullwell = min(800_000, 2**16 * self.gain)
        return fullwell

    @property
    def effgain(self):
        return self.gain / self.emgain

    def get_psf_size(self, filt_name):
        if not filt_name in self.filters:
            raise ValueError(f"{filt_name} not recognized as a filter")
        ## TODO
        raise NotImplementedError()


class CMOSVAMPIRES(InstrumentInfo):
    cam_num: Literal[1, 2]
    readmode: Literal["fast", "slow"]
    pixel_scale: ClassVar[float] = 6.018378804429752  # mas / px
    pupil_offset: ClassVar[float] = -41.323163723676146  # deg

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
        if not filt_name in self.filters:
            raise ValueError(f"{filt_name} not recognized as a filter")
        ## TODO
        raise NotImplementedError()
