from astropy.io import fits
from astropy.time import Time
from pathlib import Path

from vampires_dpp.fixes import fix_header

TEST_DIR = Path(__file__).parent
TEST_FILE = Path(TEST_DIR, "data", "VMPA00021059.fits")


def test_fix_header(tmp_path):
    output = tmp_path / f"{TEST_FILE.stem}_fix.fits"
    path = fix_header(TEST_FILE, output=output)
    assert path == output
    hdr = fits.getheader(path)
    for key in ("UT-STR", "UT-END", "HST-STR", "HST-END"):
        # bad colon delimiter fixed
        assert hdr[key].count(":") == 2
        assert hdr[key].count(".") == 1
    for key in ("UT", "HST"):
        assert _test_timestamp_iso(hdr, key)
    assert _test_timestamp_mjd(hdr)


def test_fix_header_skip(tmp_path):
    output = tmp_path / f"{TEST_FILE.stem}_fix.fits"
    path = fix_header(TEST_FILE, output=output)
    # manually change header without changing filename
    data, hdr = fits.getdata(path, header=True)
    orig_mjd = hdr["MJD"]
    hdr["MJD"] = orig_mjd + 0.1
    fits.writeto(path, data, header=hdr, overwrite=True)
    path = fix_header(TEST_FILE, output=output, skip=True)
    hdr = fits.getheader(path)
    assert hdr["MJD"] == orig_mjd + 0.1


def _test_timestamp_iso(hdr, key):
    date = hdr["DATE-OBS"]
    key_str = f"{key}-STR"
    t_str = Time(f"{date}T{hdr[key_str]}", format="fits", scale="ut1")
    key_end = f"{key}-END"
    t_end = Time(f"{date}T{hdr[key_end]}", format="fits", scale="ut1")
    t_typ = Time(f"{date}T{hdr[key]}", format="fits", scale="ut1")
    return t_typ > t_str and t_typ < t_end


def _test_timestamp_mjd(hdr):
    t_str = Time(hdr["MJD-STR"], format="mjd", scale="ut1")
    t_end = Time(hdr["MJD-END"], format="mjd", scale="ut1")
    t_typ = Time(hdr["MJD"], format="mjd", scale="ut1")
    return t_typ > t_str and t_typ < t_end
