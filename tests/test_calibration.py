from vampires_dpp.calibration import deinterleave, deinterleave_file
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
