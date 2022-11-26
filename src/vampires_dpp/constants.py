import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord

# important parameters
PIXEL_SCALE = 6.24  # mas / px
PUPIL_OFFSET = 140.4  # deg
# Subaru location - DO NOT CHANGE!
SUBARU_LOC = EarthLocation(lat=19.825504 * u.deg, lon=-155.4760187 * u.deg)
