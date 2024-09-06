import numpy as np
import pytest
from vampires_dpp.image_registration import intersect_point

rng = np.random.default_rng()


class TestIntersectPoint:
    @pytest.mark.parametrize(
        ["points", "center"],
        [
            ([(-1, 0), (1, 0), (0, -1), (0, 1)], [0, 0]),
            ([(-1, 0), (1, 0), (0.5, -1), (0.5, 1)], [0.5, 0]),
            ([(-1, 0.5), (1, 0.5), (0, -1), (0, 1)], [0, 0.5]),
            ([(-1, 0.5), (1, 0.5), (-0.5, -1), (-0.5, 1)], [-0.5, 0.5]),
        ],
    )
    def test_intersect_point(self, points, center):
        point_arr = np.array(points)
        output = intersect_point(point_arr[:, 0], point_arr[:, 1])
        np.testing.assert_allclose(output, center)

    def test_intersect_point_arr(self):
        point_arr = rng.random((100, 4, 2))
        output = intersect_point(point_arr[..., 0], point_arr[..., 1])
        assert output.shape == (100, 2)

    @pytest.mark.xfail(
        reason="If unable to properly order points, will not form line pairs correctly"
    )
    @pytest.mark.parametrize(
        ("points", "center"),
        [
            ([(-1, -1), (1, 1), (-1, 1), (1, -1)], [0, 0]),
            ([(-1, 0), (-1, 1), (1, 0), (1, 1)], (0, 0)),
        ],
    )
    def test_intersect_x(self, points, center):
        point_arr = np.array(points)
        output = intersect_point(point_arr[:, 0], point_arr[:, 1])
        np.testing.assert_allclose(output, center)


class TestCentroidFinding:
    ...
