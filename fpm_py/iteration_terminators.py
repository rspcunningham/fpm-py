
import numpy as np
from typing import Callable

TerminatorType = Callable[
    [np.ndarray, int], 
    bool
]

def iter_ceil(_: np.ndarray, i: int, max_iters: int = 10) -> bool:
    """
    Simple iteration terminator that stops after a given number of iterations.
    Args:
        _: The object array. Not used in this simple terminator.
        i: The current iteration number.
        max_iters: The maximum number of iterations, default to 10.
    """
    return i >= max_iters