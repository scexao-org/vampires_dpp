from vampires_dpp.calibration import deinterleave
import numpy as np

def test_deinterleave():
    pos = np.ones(10)
    neg = -pos
    cube = np.vstack((pos, neg, pos, neg, pos, neg))
    set1, set2 = deinterleave(cube)
    assert np.allclose(set1, pos)
    assert np.allclose(set2, neg)
