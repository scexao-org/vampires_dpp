import pytest
from vampires_dpp.util import average_angle, flc_inds
import numpy as np


def test_flc_inds_good():
    states = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4])
    inds = flc_inds(states, 2)
    assert np.allclose(states[inds], states)

    states = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    inds = flc_inds(states, 4)
    assert np.allclose(states[inds], states)


def test_flc_inds_filter():
    states = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4])
    inds = flc_inds(states, 2)
    assert np.allclose(inds, [6, 7, 8, 9, 10, 11, 12, 13])
    assert np.allclose(states[inds], [1, 1, 2, 2, 3, 3, 4, 4])

    states = np.array(
        [
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            4,
            4,
        ]
    )
    inds = flc_inds(states, 4)
    assert np.allclose(inds, range(16))
    assert np.allclose(states[inds], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])


def test_average_angle():
    # generate random angles
    xs = np.random.rand(100) * 2 - 1
    ys = np.random.rand(100) * 2 - 1
    angles = np.arctan2(ys, xs)
    angles_deg = np.rad2deg(angles)
    expected = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    expected_deg = np.rad2deg(expected)
    assert np.allclose(expected_deg, average_angle(angles_deg))
