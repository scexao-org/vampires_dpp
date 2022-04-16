import pytest
from vampires_dpp.util import  frame_center
import numpy as np

@pytest.mark.parametrize("frame,center", [
    (np.empty((10, 10)), (4.5, 4.5)),
    (np.empty((11, 11)), (5, 5)),
    (np.empty((100, 11, 11)), (5, 5)),
    (np.empty((10, 100, 16, 11)), (7.5, 5)),
])
def test_frame_center(frame, center):
    fcenter = frame_center(frame)
    assert fcenter[0] == center[0]
    assert fcenter[1] == center[1]