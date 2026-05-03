import numpy as np

from vampires_dpp.indexing import cutout_inds, frame_center


def test_cutout_inds():
    frame = np.empty((101, 101))
    inds = cutout_inds(frame, center=(80, 50), window=31)
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 64
    assert sy.stop == 96
    assert sx.start == 34
    assert sx.stop == 66


def test_cutout_inds_window_shape():
    frame = np.empty((101, 101))
    inds = cutout_inds(frame, center=(80, 50), window=(31, 40))
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 64
    assert sy.stop == 96
    assert sx.start == 30
    assert sx.stop == 70


def test_cutout_inds_out_of_bounds():
    frame = np.empty((101, 101))

    inds = cutout_inds(frame, center=(90, 90), window=31)
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 74
    assert sx.start == 74

    inds = cutout_inds(frame, center=(10, 90), window=31)
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 0
    assert sx.start == 74

    inds = cutout_inds(frame, center=(10, 10), window=31)
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 0
    assert sx.start == 0

    inds = cutout_inds(frame, center=(90, 10), window=31)
    sy, sx = inds[-2], inds[-1]
    assert sy.start == 74
    assert sx.start == 0


def test_frame_center():
    frame = np.empty((30, 45))
    ctr = frame_center(frame)
    expected = np.array((14.5, 22))
    np.testing.assert_allclose(ctr, expected)


def test_frame_center_nd():
    cube = np.empty((100, 45, 30))
    ctr = frame_center(cube)
    expected = np.array((22, 14.5))
    np.testing.assert_allclose(ctr, expected)

    cube = np.empty((100, 13, 45, 30))
    ctr = frame_center(cube)
    expected = np.array((22, 14.5))
    np.testing.assert_allclose(ctr, expected)
