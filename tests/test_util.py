from pathlib import Path

import numpy as np
import pytest
from vampires_dpp.paths import get_paths
from vampires_dpp.util import average_angle, check_version

rng = np.random.default_rng(4796)


def test_average_angle():
    # generate random angles
    xs = rng.uniform(-1, 1, size=100)
    ys = rng.uniform(-1, 1, size=100)
    angles = np.arctan2(ys, xs)
    angles_deg = np.rad2deg(angles)
    expected = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    expected_deg = np.rad2deg(expected)
    assert np.allclose(expected_deg, average_angle(angles_deg))


@pytest.mark.parametrize(
    ("cver", "vpver", "exp"),
    (
        ("0.1.0", "0.1.0", True),
        ("0.1.0", "0.2.0", False),
        ("0.2.2", "0.2.0", False),
        ("0.2.2", "0.2.4", True),
        ("0.3.0", "0.2.0", False),
        ("1.0.0", "1.0.0", True),
        ("1.2.0", "1.0.0", False),
        ("1.0.0", "1.2.0", True),
        ("1.2.3", "1.2.0", False),
        ("1.2.3", "1.2.5", True),
        ("1.2.3", "2.0.0", False),
    ),
)
def test_check_version(cver, vpver, exp):
    assert check_version(cver, vpver) == exp


@pytest.mark.parametrize(
    ("name", "kwds", "expected"),
    [
        ("test.fits", dict(outname="test_out.fits"), "test_out.fits"),
        ("test.fits", dict(output_directory="test"), "test/test.fits"),
        ("test.fits", dict(output_directory="test", outname="test_out.fits"), "test/test_out.fits"),
        ("test.fits", dict(output_directory="test", suffix="calib"), "test/test_calib.fits"),
        ("test.fits", dict(suffix="calib"), "test_calib.fits"),
        ("test.fits", dict(suffix="coef", filetype=".csv"), "test_coef.csv"),
        ("test.fits.fz", dict(outname="test_out.fits"), "test_out.fits"),
        ("test.fits.fz", dict(output_directory="test"), "test/test.fits"),
        (
            "test.fits.fz",
            dict(output_directory="test", outname="test_out.fits"),
            "test/test_out.fits",
        ),
        ("test.fits.fz", dict(output_directory="test", suffix="calib"), "test/test_calib.fits"),
        ("test.fits.fz", dict(suffix="calib"), "test_calib.fits"),
        ("test.fits.fz", dict(suffix="coef", filetype=".csv"), "test_coef.csv"),
        ("test.fits.gz", dict(outname="test_out.fits"), "test_out.fits"),
        ("test.fits.gz", dict(output_directory="test"), "test/test.fits"),
        (
            "test.fits.gz",
            dict(output_directory="test", outname="test_out.fits"),
            "test/test_out.fits",
        ),
        ("test.fits.gz", dict(output_directory="test", suffix="calib"), "test/test_calib.fits"),
        ("test.fits.gz", dict(suffix="calib"), "test_calib.fits"),
        ("test.fits.gz", dict(suffix="coef", filetype=".csv"), "test_coef.csv"),
    ],
)
def test_get_paths(name, kwds, expected):
    name = "test.fits"
    path, outpath = get_paths(name, **kwds)
    assert path == Path(name)
    assert outpath == Path(expected)
