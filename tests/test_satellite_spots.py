import numpy as np

from vampires_dpp.satellite_spots import window_centers


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
