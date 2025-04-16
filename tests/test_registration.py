import numpy as np
import pytest
from skimage.measure import centroid
from vampires_dpp.registration import offset_centroids

rng = np.random.default_rng(4796)


@pytest.fixture
def random_array(shape):
    return rng.normal(size=shape)


def test_centroid_skimage():
    data = np.arange(1, 101).reshape(10, 10)
    dpp_centroid = offset_centroids(data, np.s_[0 : data.shape[-2], 0 : data.shape[-1]])
    skimage_centroid = centroid(data)
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)


def test_centroid_nan():
    data = np.arange(1, 101).reshape(10, 10)
    data[3, 5] = np.nan
    dpp_centroid = offset_centroids(data, np.s_[0 : data.shape[-2], 0 : data.shape[-1]])
    skimage_centroid = centroid(data)
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)


def test_centroid_skimage_rand(random_array):
    dpp_centroid = offset_centroids(
        random_array, np.s_[0 : random_array.shape[-2], 0 : random_array.shape[-1]]
    )
    skimage_centroid = centroid(random_array)
    np.testing.assert_allclose(dpp_centroid, skimage_centroid)
