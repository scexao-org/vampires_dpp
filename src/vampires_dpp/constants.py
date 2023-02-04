from multiprocessing import cpu_count

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

# important parameters
PIXEL_SCALE = 6.24  # mas / px
PUPIL_OFFSET = 140.4  # deg
# Subaru location - DO NOT CHANGE!
SUBARU_LOC = EarthLocation(lat=19.825504 * u.deg, lon=-155.4760187 * u.deg)

FILTER_ANGULAR_SIZE = {
    "Open": np.rad2deg(700e-9 / 7.79) * 3.6e6,
    "625-50": np.rad2deg(625e-9 / 7.79) * 3.6e6,
    "675-50": np.rad2deg(675e-9 / 7.79) * 3.6e6,
    "725-50": np.rad2deg(725e-9 / 7.79) * 3.6e6,
    "750-50": np.rad2deg(750e-9 / 7.79) * 3.6e6,
    "775-50": np.rad2deg(775e-9 / 7.79) * 3.6e6,
}

# limit default nproc since many operations are
# throttled by file I/O
DEFAULT_NPROC = min(cpu_count(), 8)
