import pytest
from vampires_dpp.util import frame_center, flc_inds
import numpy as np


@pytest.mark.parametrize(
    "frame,center",
    [
        (np.empty((10, 10)), (4.5, 4.5)),
        (np.empty((11, 11)), (5, 5)),
        (np.empty((100, 11, 11)), (5, 5)),
        (np.empty((10, 100, 16, 11)), (7.5, 5)),
    ],
)
def test_frame_center(frame, center):
    fcenter = frame_center(frame)
    assert fcenter[0] == center[0]
    assert fcenter[1] == center[1]


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
