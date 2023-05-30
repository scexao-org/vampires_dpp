import numpy as np
import pytest
from astropy.io import fits

from vampires_dpp.calibration import calibrate_file, make_background_file, make_flat_file


class TestCalibrationFrames:
    @pytest.fixture()
    def background_frame(self, tmp_path):
        path = tmp_path / "master_back_cam1.fits"
        back = np.random.randn(512, 512) + 200
        fits.writeto(path, back.astype("uint16"), overwrite=True)
        return path

    @pytest.fixture()
    def flat_cube(self, tmp_path):
        data = 10 * np.random.randn(100, 512, 512) + 1.5e4
        # add photon noise
        data = np.random.poisson(data)
        path = tmp_path / "flat_file_cam1.fits"
        fits.writeto(path, data.astype("uint16"))
        return path

    def test_make_background_file(self, tmp_path):
        cube = 10 * np.random.randn(100, 512, 512) + 200
        path = tmp_path / "back_file_cam1.fits"
        fits.writeto(path, cube.astype("uint16"))
        outpath = make_background_file(path)
        assert outpath == path.with_stem(f"{path.stem}_collapsed")
        c, h = fits.getdata(outpath, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 200, rtol=1e-2)
        name = tmp_path / "master_back_cam1.fits"
        make_background_file(path, outname=name)
        c, h = fits.getdata(name, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 200, rtol=1e-2)

    def test_make_flat_file(self, flat_cube):
        outpath = make_flat_file(flat_cube)
        assert outpath == flat_cube.with_name(f"{flat_cube.stem}_collapsed{flat_cube.suffix}")
        c, h = fits.getdata(
            outpath,
            header=True,
        )
        assert c.dtype == np.dtype(">f4")
        assert "DPP_BACK" not in h
        assert np.isclose(np.median(c), 1)

    def test_make_flat_file_with_back(self, tmp_path, background_frame, flat_cube):
        outpath = make_flat_file(
            flat_cube, back_filename=background_frame, outname=tmp_path / "master_flat_cam1.fits"
        )
        assert outpath == tmp_path / "master_flat_cam1.fits"
        c, h = fits.getdata(outpath, header=True)
        assert c.dtype == np.dtype(">f4")
        assert np.isclose(np.median(c), 1)
        assert h["DPP_BACK"] == background_frame.name


class TestCalibrate:
    @pytest.fixture()
    def background_frame(self):
        data = np.random.randn(512, 512) + 200
        return data.astype("f4")

    @pytest.fixture()
    def flat_frame(self):
        return np.ones((512, 512), dtype="f4")

    @pytest.fixture()
    def data_cube_cam1(self, background_frame):
        cube = 10 * np.random.randn(102, 512, 512) + background_frame
        cube += np.random.poisson(2e4, (102, 512, 512))
        return np.flipud(cube).astype("uint16")

    @pytest.fixture()
    def data_cube_cam2(self, background_frame):
        cube = 10 * np.random.randn(102, 512, 512) + background_frame
        cube += np.random.poisson(2.3e4, (102, 512, 512))
        return cube.astype("uint16")

    def test_calibrate_files_dark(self, tmp_path, data_cube_cam1, data_cube_cam2, dark_frame):
        # save data to disk
        dark_path = tmp_path / "master_dark_cam1.fits"
        fits.writeto(dark_path, dark_frame)
        path1 = tmp_path / "data_cube_cam1.fits"
        fits.writeto(path1, data_cube_cam1)
        path2 = tmp_path / "data_cube_cam2.fits"
        fits.writeto(path2, data_cube_cam2)

        outpath1 = calibrate_file(path1, dark=dark_path, discard=2)
        outpath2 = calibrate_file(path2, dark=dark_path, suffix="_cal", discard=2)

        assert outpath1 == path1.with_name(f"{path1.stem}_calib{path1.suffix}")
        assert outpath2 == path2.with_name(f"{path2.stem}_cal{path2.suffix}")

        calib1, hdr1 = fits.getdata(outpath1, header=True)
        calib2, hdr2 = fits.getdata(outpath2, header=True)
        assert calib1.dtype == np.dtype(">f4")
        assert calib2.dtype == np.dtype(">f4")
        assert calib1.shape[0] == 100
        assert calib2.shape[0] == 100
        assert np.allclose(np.median(calib1, axis=(1, 2)), 2e4, rtol=1e-3)
        assert np.allclose(np.median(calib2, axis=(1, 2)), 2.3e4, rtol=1e-3)
        # if flip didn't work, they won't add together
        assert np.allclose(np.median(calib1 + calib2, axis=(1, 2)), 4.3e4, rtol=1e-3)
        assert hdr1["MDARK"] == dark_path.name
        assert hdr2["MDARK"] == dark_path.name

    def test_calibrate_files_dark_and_flat(
        self, tmp_path, data_cube_cam1, data_cube_cam2, dark_frame, flat_frame
    ):
        # save data to disk
        dark_path = tmp_path / "master_dark_cam1.fits"
        fits.writeto(dark_path, dark_frame)
        flat_path = tmp_path / "master_flat_cam1.fits"
        fits.writeto(flat_path, flat_frame)
        path1 = tmp_path / "data_cube_cam1.fits"
        fits.writeto(path1, data_cube_cam1)
        path2 = tmp_path / "data_cube_cam2.fits"
        fits.writeto(path2, data_cube_cam2)

        outpath1 = calibrate_file(path1, dark=dark_path, flat=flat_path, discard=2)
        outpath2 = calibrate_file(path2, dark=dark_path, flat=flat_path, suffix="_cal", discard=2)

        assert outpath1 == path1.with_name(f"{path1.stem}_calib{path1.suffix}")
        assert outpath2 == path2.with_name(f"{path2.stem}_cal{path2.suffix}")

        calib1, hdr1 = fits.getdata(outpath1, header=True)
        calib2, hdr2 = fits.getdata(outpath2, header=True)
        assert calib1.dtype == np.dtype(">f4")
        assert calib2.dtype == np.dtype(">f4")
        assert calib1.shape[0] == 100
        assert calib2.shape[0] == 100
        assert np.allclose(np.median(calib1, axis=(1, 2)), 2e4, rtol=1e-3)
        assert np.allclose(np.median(calib2, axis=(1, 2)), 2.3e4, rtol=1e-3)
        # if flip didn't work, they won't add together
        assert np.allclose(np.median(calib1 + calib2, axis=(1, 2)), 4.3e4, rtol=1e-3)
        assert hdr1["MDARK"] == dark_path.name
        assert hdr2["MDARK"] == dark_path.name
        assert hdr1["MFLAT"] == flat_path.name
        assert hdr2["MFLAT"] == flat_path.name
