import numpy as np
import pytest
from astropy.io import fits

from vampires_dpp.calib.calib_files import make_background_file, make_flat_file

rng = np.random.default_rng(4796)


def _cmos_header():
    """Minimal FITS header that satisfies fix_header for CMOS cam1 (pre-NBS)."""
    hdr = fits.Header()
    hdr["U_CAMERA"] = 1
    hdr["U_DETMOD"] = "slow"
    hdr["MJD"] = 60000.0
    hdr["D_IMRPAP"] = 0.0
    hdr["EXPTIME"] = 0.001
    hdr["FILTER01"] = "750-50"
    hdr["FILTER02"] = "Open"
    return hdr


def _emccd_header():
    """Minimal FITS header that satisfies fix_header for EMCCD cam1."""
    hdr = fits.Header()
    hdr["U_CAMERA"] = 1
    hdr["U_EMGAIN"] = 300
    hdr["MJD"] = 60000.0
    hdr["D_IMRPAP"] = 0.0
    hdr["EXPTIME"] = 0.001
    hdr["FILTER01"] = "750-50"
    hdr["FILTER02"] = "Open"
    return hdr


class TestCalibrationFrames:
    @pytest.fixture()
    def background_frame(self, tmp_path):
        path = tmp_path / "master_back_cam1.fits"
        back = np.zeros((512, 512), dtype="f4")
        back_err = np.zeros((512, 512), dtype="f4")
        hdr = _cmos_header()
        hdul = fits.HDUList(
            [fits.PrimaryHDU(back, header=hdr), fits.ImageHDU(back_err, name="ERR")]
        )
        hdul.writeto(path, overwrite=True)
        return path

    @pytest.fixture()
    def flat_cube(self, tmp_path):
        data = rng.normal(loc=1.5e4, scale=10, size=(100, 512, 512))
        data = rng.poisson(data)
        path = tmp_path / "flat_file_cam1.fits"
        hdr = _cmos_header()
        fits.writeto(path, data.astype("uint16"), header=hdr)
        return path

    @pytest.fixture()
    def flat_cube_emccd(self, tmp_path):
        data = rng.normal(loc=1.5e4, scale=10, size=(100, 512, 512))
        data = rng.poisson(data)
        path = tmp_path / "flat_file_cam1.fits"
        hdr = _emccd_header()
        fits.writeto(path, data.astype("uint16"), header=hdr)
        return path

    def test_make_background_file(self, tmp_path):
        cube = rng.normal(loc=200, scale=10, size=(100, 512, 512))
        path = tmp_path / "back_file_cam1.fits"
        hdr = _cmos_header()
        fits.writeto(path, cube.astype("uint16"), header=hdr)
        outpath = make_background_file(path)
        assert outpath == path.with_stem(f"{path.stem}_coll")
        c, h = fits.getdata(outpath, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 200, rtol=1e-2)

        name = tmp_path / "master_back_cam1.fits"
        make_background_file(path, outname=name)
        c, h = fits.getdata(name, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 200, rtol=1e-2)

    def test_make_flat_file(self, flat_cube_emccd):
        outpath = make_flat_file(flat_cube_emccd)
        assert outpath == flat_cube_emccd.with_stem(f"{flat_cube_emccd.stem}_coll")
        c, h = fits.getdata(outpath, header=True)
        assert c.dtype == np.dtype(">f4")
        assert "BACKFILE" not in h
        assert np.isclose(np.median(c), 1)

    def test_make_flat_file_with_back(self, tmp_path, background_frame, flat_cube):
        outpath = make_flat_file(
            flat_cube, back_filename=background_frame, outname=tmp_path / "master_flat_cam1.fits"
        )
        assert outpath == tmp_path / "master_flat_cam1.fits"
        c, h = fits.getdata(outpath, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 1)
        assert h["BACKFILE"] == background_frame.name
