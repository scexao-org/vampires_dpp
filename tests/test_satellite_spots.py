import numpy as np

from vampires_dpp.satellite_spots import (
    window_centers,
    window_slice,
    window_indices,
    window_masks,
)


def test_window_centers():
    centers = window_centers((4, 4), radius=2)
    assert np.allclose(centers[0], [4, 6])
    assert np.allclose(centers[1], [6, 4])
    assert np.allclose(centers[2], [4, 2])
    assert np.allclose(centers[3], [2, 4])


def test_window_centers_offsets():
    centers = window_centers((40, 50), radius=10, theta=30)
    assert np.allclose(centers[0], [45, 58.66])
    assert np.allclose(centers[1], [48.66, 45])
    assert np.allclose(centers[2], [35, 41.34])
    assert np.allclose(centers[3], [31.34, 55])


def test_window_centers_two():
    centers = window_centers((50, 50), radius=10, theta=90, n=2)
    assert np.allclose(centers[0], [60, 50])
    assert np.allclose(centers[1], [40, 50])


def test_window_slice():
    frame = np.empty((101, 101))
    sy, sx = window_slice(frame, center=(80, 50), window=30)
    assert sy.start == 65
    assert sy.stop == 96
    assert sx.start == 35
    assert sx.stop == 66


def test_window_slice_window_shape():
    frame = np.empty((101, 101))
    sy, sx = window_slice(frame, center=(80, 50), window=(30, 40))
    assert sy.start == 65
    assert sy.stop == 96
    assert sx.start == 30
    assert sx.stop == 71


def test_window_slice_out_of_bounds():
    frame = np.empty((101, 101))
    sy, sx = window_slice(frame, center=(90, 90), window=30)
    assert sy.start == 75
    assert sy.stop == 101
    assert sx.start == 75
    assert sx.stop == 101

    sy, sx = window_slice(frame, center=(10, 90), window=30)
    assert sy.start == 0
    assert sy.stop == 26
    assert sx.start == 75
    assert sx.stop == 101

    sy, sx = window_slice(frame, center=(10, 10), window=30)
    assert sy.start == 0
    assert sy.stop == 26
    assert sx.start == 0
    assert sx.stop == 26

    sy, sx = window_slice(frame, center=(90, 10), window=30)
    assert sy.start == 75
    assert sy.stop == 101
    assert sx.start == 0
    assert sx.stop == 26


def test_window_indices():
    frame = np.empty((101, 101))
    inds = window_indices(frame, radius=10, window=30)


def test_window_masks():
    frame = np.zeros((101, 101))
    frame[60, 50] = 1
    frame[50, 60] = 1
    frame[40, 50] = 1
    frame[50, 40] = 1
    m = window_masks(frame, radius=10)
    assert frame[m].sum() == 4
