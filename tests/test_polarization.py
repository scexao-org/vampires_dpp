import numpy as np
from vampires_dpp.polarization import pol_inds


def test_pol_inds_good():
    states = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4])
    inds = pol_inds(states, 2)
    assert np.allclose(states[inds], states)

    states = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    inds = pol_inds(states, 4)
    assert np.allclose(states[inds], states)


def test_pol_inds_filter():
    states = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4])
    inds = pol_inds(states, 2)
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
    inds = pol_inds(states, 4)
    assert np.allclose(inds, range(16))
    assert np.allclose(states[inds], [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
