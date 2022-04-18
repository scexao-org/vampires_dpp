import numpy as np

from vampires_dpp.image_processing import shift_frame


def test_shift_frame():
    array = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    shift_down = shift_frame(array, (-1, 0))
    assert np.allclose(shift_down, np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))

    shift_up = shift_frame(array, (1, 0))
    assert np.allclose(shift_up, np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]))

    shift_left = shift_frame(array, (0, -1))
    assert np.allclose(shift_left, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]))

    shift_right = shift_frame(array, (0, 1))
    assert np.allclose(shift_right, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
