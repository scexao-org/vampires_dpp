import numpy as np
import pytest
from skimage.measure import centroid

from vampires_dpp.image_registration import offset_centroid


@pytest.fixture
def random_array(shape):
    np.random.seed(8865)
    return np.random.randn(shape)


def test_centroid_skimage():
    data = np.arange(1, 101).reshape(10, 10)
    dpp_centroid = offset_centroid(data, np.s_[0 : data.shape[-2], 0 : data.shape[-1]])
    skimage_centroid = centroid(data)
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)


def test_centroid_nan():
    data = np.arange(1, 101).reshape(10, 10)
    data[3, 5] = np.nan
    dpp_centroid = offset_centroid(data, np.s_[0 : data.shape[-2], 0 : data.shape[-1]])
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)


def test_centroid_skimage_rand(random_array):
    dpp_centroid = offset_centroid(
        random_array, np.s_[0 : random_array.shape[-2], 0 : random_array.shape[-1]]
    )
    skimage_centroid = centroid(random_array)
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)
