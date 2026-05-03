import numpy as np
import pytest

from vampires_dpp.image_processing import (
    adaptive_sigma_clip_mask,
    derotate_frame,
    frame_center,
    radial_profile_image,
    shift_cube,
    shift_frame,
)

# ---------------------------------------------------------------------------
# shift_frame
# ---------------------------------------------------------------------------


def test_shift_frame_identity():
    frame = np.random.default_rng(0).standard_normal((32, 32))
    result = shift_frame(frame, (0, 0))
    assert np.allclose(result, frame, atol=1e-10)


@pytest.mark.parametrize("dy,dx", [(1, 0), (-1, 0), (0, 1), (0, -1), (2, -3)])
def test_shift_frame_integer(dy, dx):
    frame = np.zeros((16, 16))
    frame[8, 8] = 1.0
    result = shift_frame(frame, (dy, dx))
    # peak moves to (8+dy, 8+dx); Fourier shift uses periodic boundary
    expected_y = (8 + dy) % 16
    expected_x = (8 + dx) % 16
    peak = np.unravel_index(np.argmax(result), result.shape)
    assert peak == (expected_y, expected_x)


def test_shift_frame_subpixel_roundtrip():
    # Use a bandlimited signal (Gaussian PSF). Random noise has significant
    # Nyquist-frequency energy which accumulates aliasing error on round-trip.
    ny, nx = 64, 64
    y, x = np.ogrid[:ny, :nx]
    frame = np.exp(-((y - ny // 2) ** 2 + (x - nx // 2) ** 2) / (2 * 5**2))
    shift = (1.3, -2.7)
    roundtrip = shift_frame(shift_frame(frame, shift), (-shift[0], -shift[1]))
    assert np.allclose(roundtrip, frame, atol=1e-10)


def test_shift_frame_output_dtype():
    frame = np.ones((8, 8), dtype="f4")
    result = shift_frame(frame, (0.5, 0.5))
    assert result.dtype == np.float64


def test_shift_frame_nan_edges_no_propagation():
    # NaN at the edges (e.g. from registration padding) must not contaminate the interior
    frame = np.zeros((32, 32))
    frame[14:18, 14:18] = 1.0
    frame[:3, :] = np.nan
    frame[-3:, :] = np.nan
    result = shift_frame(frame, (1.0, 1.0))
    assert np.isfinite(result[15:19, 15:19]).all()


def test_shift_frame_nan_boundary_restored():
    # NaN boundary should move with the shift (periodic boundaries)
    frame = np.ones((32, 32))
    frame[:4, :] = np.nan
    result = shift_frame(frame, (2.0, 0.0))
    # rows 0-3 shift to rows 2-5; rows 0-1 are filled by the (finite) bottom wrap
    assert np.all(np.isnan(result[2:6, :]))
    assert np.all(np.isfinite(result[7:, :]))


# ---------------------------------------------------------------------------
# shift_cube
# ---------------------------------------------------------------------------


def test_shift_cube_matches_frame_by_frame():
    rng = np.random.default_rng(7)
    cube = rng.standard_normal((10, 32, 32))
    shifts = rng.uniform(-3, 3, size=(10, 2))

    vectorized = shift_cube(cube, shifts)
    sequential = np.stack([shift_frame(cube[i], shifts[i]) for i in range(10)])

    assert np.allclose(vectorized, sequential, atol=1e-10)


def test_shift_cube_identity():
    rng = np.random.default_rng(0)
    cube = rng.standard_normal((5, 16, 16))
    shifts = np.zeros((5, 2))
    result = shift_cube(cube, shifts)
    assert np.allclose(result, cube, atol=1e-10)


def test_shift_cube_shape():
    cube = np.ones((7, 24, 24))
    shifts = np.zeros((7, 2))
    assert shift_cube(cube, shifts).shape == cube.shape


def test_shift_cube_nan_edges_no_propagation():
    cube = np.zeros((5, 32, 32))
    cube[:, 14:18, 14:18] = 1.0
    cube[:, :3, :] = np.nan
    shifts = np.ones((5, 2))
    result = shift_cube(cube, shifts)
    assert np.isfinite(result[:, 15:19, 15:19]).all()


# ---------------------------------------------------------------------------
# derotate_frame
# ---------------------------------------------------------------------------


def test_derotate_frame():
    # Array must be large enough that interior pixels have >4px margin for Lanczos4
    array = np.zeros((33, 33))
    array[16, 12] = 1  # 4px from left edge, well inside

    cw_90 = derotate_frame(array, 90)
    peak = np.unravel_index(np.nanargmax(cw_90), cw_90.shape)
    assert peak == (12, 16)

    ccw_90 = derotate_frame(array, -90)
    peak = np.unravel_index(np.nanargmax(ccw_90), ccw_90.shape)
    assert peak == (20, 16)


def test_derotate_frame_offset():
    array = np.zeros((33, 33))
    array[16, 12] = 1

    cw_90 = derotate_frame(array, 90, center=(16, 16))
    peak = np.unravel_index(np.nanargmax(cw_90), cw_90.shape)
    assert peak == (12, 16)


# ---------------------------------------------------------------------------
# frame_center
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# radial_profile_image
# ---------------------------------------------------------------------------


def test_radial_profile_image_shape():
    frame = np.ones((32, 32))
    result = radial_profile_image(frame)
    assert result.shape == frame.shape


def test_radial_profile_image_constant():
    frame = np.full((32, 32), 5.0)
    result = radial_profile_image(frame)
    assert np.allclose(result, 5.0, atol=1e-10)


def test_radial_profile_image_radial_symmetry():
    ny, nx = 64, 64
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    frame = np.exp(-r / 10)
    result = radial_profile_image(frame, fwhm=1)
    assert np.allclose(result, frame, atol=0.05)


# ---------------------------------------------------------------------------
# adaptive_sigma_clip_mask
# ---------------------------------------------------------------------------


def test_adaptive_sigma_clip_mask_shape():
    data = np.random.default_rng(0).standard_normal((64, 64))
    mask = adaptive_sigma_clip_mask(data)
    assert mask.shape == data.shape
    assert mask.dtype == bool


def test_adaptive_sigma_clip_mask_clean():
    data = np.ones((32, 32))
    mask = adaptive_sigma_clip_mask(data, sigma=3)
    assert not mask.any()


def test_adaptive_sigma_clip_mask_detects_spike():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((64, 64))
    data[32, 32] = 1000.0
    mask = adaptive_sigma_clip_mask(data, sigma=5)
    assert mask[32, 32]


def test_adaptive_sigma_clip_mask_non_square():
    data = np.random.default_rng(0).standard_normal((48, 64))
    mask = adaptive_sigma_clip_mask(data)
    assert mask.shape == data.shape


def test_adaptive_sigma_clip_mask_non_divisible():
    data = np.random.default_rng(0).standard_normal((37, 53))
    mask = adaptive_sigma_clip_mask(data, boxsize=8)
    assert mask.shape == data.shape
