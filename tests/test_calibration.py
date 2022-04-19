from vampires_dpp.calibration import (
    deinterleave,
    deinterleave_file,
    make_dark_file,
    make_flat_file,
    calibrate,
    calibrate_file,
)
from astropy.io import fits
import numpy as np
import pytest


class TestDeinterleave:
    @pytest.fixture()
    def cube(self):
        pos = np.ones(10)
        neg = -pos
        yield np.vstack((pos, neg, pos, neg, pos, neg))

    def test_deinterleave(self, cube):
        set1, set2 = deinterleave(cube)
        assert np.allclose(set1, 1)
        assert np.allclose(set2, -1)

    def test_deinterleave_files(self, tmp_path, cube):
        path = tmp_path / "cube.fits"
        fits.writeto(path, cube)
        deinterleave_file(path)
        assert path.with_name(f"cube_FLC1{path.suffix}").exists()
        assert path.with_name(f"cube_FLC2{path.suffix}").exists()
        flc1, hdr1 = fits.getdata(
            path.with_name(f"cube_FLC1{path.suffix}"), header=True
        )
        flc2, hdr2 = fits.getdata(
            path.with_name(f"cube_FLC2{path.suffix}"), header=True
        )
        assert np.allclose(flc1, 1)
        assert hdr1["U_FLCSTT"] == 1
        assert np.allclose(flc2, -1)
        assert hdr2["U_FLCSTT"] == 2


class TestCalibrationFrames:
    @pytest.fixture()
    def dark_frame(self, tmp_path):
        path = tmp_path / "master_dark_cam1.fits"
        dark = np.random.randn(512, 512) + 200
        fits.writeto(path, dark.astype("uint16"), overwrite=True)
        return path

    @pytest.fixture()
    def flat_cube(self, tmp_path):
        data = 10 * np.random.randn(100, 512, 512) + 1.5e4
        # add photon noise
        data = np.random.poisson(data)
        path = tmp_path / "flat_file_cam1.fits"
        fits.writeto(path, data.astype("uint16"))
        return path

    def test_make_dark_file(self, tmp_path):
        cube = 10 * np.random.randn(100, 512, 512) + 200
        path = tmp_path / "dark_file_cam1.fits"
        fits.writeto(path, cube.astype("uint16"))
        make_dark_file(path)
        c, h = fits.getdata(
            path.with_name(f"{path.stem}_master_dark{path.suffix}"), header=True
        )
        assert np.isclose(np.median(c), 200, rtol=1e-2)
        make_dark_file(path, output=tmp_path / "master_dark_cam1.fits")
        c, h = fits.getdata(tmp_path / "master_dark_cam1.fits", header=True)
        assert np.isclose(np.median(c), 200, rtol=1e-2)

    def test_make_flat_file(self, flat_cube):
        make_flat_file(flat_cube)
        c, h = fits.getdata(
            flat_cube.with_name(f"{flat_cube.stem}_master_flat{flat_cube.suffix}"),
            header=True,
        )
        assert "VPP_DARK" not in h
        assert np.isclose(np.median(c), 1)

    def test_make_flat_file_with_dark(self, tmp_path, dark_frame, flat_cube):
        make_flat_file(
            flat_cube, dark=dark_frame, output=tmp_path / "master_flat_cam1.fits"
        )
        c, h = fits.getdata(tmp_path / "master_flat_cam1.fits", header=True)
        assert np.isclose(np.median(c), 1)
        assert h["VPP_DARK"] == dark_frame.name


class TestCalibrate:
    @pytest.fixture()
    def dark_frame(self):
        data = np.random.randn(512, 512) + 200
        return data.astype("f4")

    @pytest.fixture()
    def flat_frame(self):
        return np.ones((512, 512), dtype="f4")

    @pytest.fixture()
    def data_cube_cam1(self, dark_frame):
        cube = 10 * np.random.randn(100, 512, 512) + dark_frame
        cube += np.random.poisson(2e4, (100, 512, 512))
        return cube.astype("uint16")

    @pytest.fixture()
    def data_cube_cam2(self, dark_frame):
        cube = 10 * np.random.randn(100, 512, 512) + dark_frame
        cube += np.random.poisson(2.3e4, (100, 512, 512))
        cube = np.flip(cube, axis=1)
        return cube.astype("uint16")

    def test_calibrate(self, data_cube_cam1, data_cube_cam2, dark_frame, flat_frame):
        calib1 = calibrate(data_cube_cam1, dark=dark_frame, flat=flat_frame)
        calib2 = calibrate(data_cube_cam2, dark=dark_frame, flat=flat_frame, flip=True)
        assert np.allclose(np.median(calib1, axis=(1, 2)), 2e4, rtol=1e-3)
        assert np.allclose(np.median(calib2, axis=(1, 2)), 2.3e4, rtol=1e-3)
        # if flip didn't work, they won't add together
        assert np.allclose(np.median(calib1 + calib2, axis=(1, 2)), 4.3e4, rtol=1e-3)
