import numpy as np
import pytest

from vampires_dpp.util import average_angle


def test_average_angle():
    # generate random angles
    xs = np.random.rand(100) * 2 - 1
    ys = np.random.rand(100) * 2 - 1
    angles = np.arctan2(ys, xs)
    angles_deg = np.rad2deg(angles)
    expected = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    expected_deg = np.rad2deg(expected)
    assert np.allclose(expected_deg, average_angle(angles_deg))
