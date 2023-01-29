import numpy as np
import pytest

from vampires_dpp.util import average_angle, check_version


def test_average_angle():
    # generate random angles
    xs = np.random.rand(100) * 2 - 1
    ys = np.random.rand(100) * 2 - 1
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
