import numpy as np
import pytest

from vampires_dpp.image_processing import shift_frame, derotate_frame, frame_center


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


def test_derotate_frame():
    array = np.zeros((5, 5))
    array[2, 1] = 1

    cw_90 = derotate_frame(array, 90)
    expected = np.zeros_like(array)
    expected[1, 2] = 1
    assert np.allclose(cw_90, expected)

    ccw_90 = derotate_frame(array, -90)
    expected = np.zeros_like(array)
    expected[3, 2] = 1
    assert np.allclose(ccw_90, expected)


def test_derotate_frame_offset():
    array = np.zeros((7, 5))
    array[2, 1] = 1

    cw_90 = derotate_frame(array, 90, center=(2, 2), mode="constant", cval=0)
    expected = np.zeros_like(array)
    expected[1, 2] = 1
    assert np.allclose(cw_90, expected)


def test_derotate_frame_kwargs():
    array = np.zeros((5, 5))
    array[2, 1] = 1

    cw_90 = derotate_frame(array, 90, order=1, mode="symmetric")
    expected = np.zeros_like(array)
    expected[1, 2] = 1
    assert np.allclose(cw_90, expected)


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
