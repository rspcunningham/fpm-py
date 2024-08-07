from fpm_py.utils import *
import numpy as np

def test_circle_like_easy():
    array = np.ones((5, 5))
    mask = circle_like(array)
    print(mask)
    assert mask.shape == array.shape
    assert mask.dtype == 'bool'
    assert mask[0, 0] == 0
    assert mask[2, 2] == 1
    assert mask[4, 4] == 0


def test_circle_like_hard():
    array = np.ones((5, 10))
    mask = circle_like(array)
    print(mask)
    assert mask.shape == array.shape
    assert mask.dtype == 'bool'
    assert mask[0, 0] == 0
    assert mask[3, 5] == 1
    assert mask[3, 7] == 0