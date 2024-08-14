
import torch
from typing import Callable

TerminatorType = Callable[
    [torch.Tensor, int], 
    bool
]
"""
Type alias for the iteration terminator function type. A function that takes an object tensor and an iteration number and returns a boolean indicating whether to stop iterating. A return value of True indicates that the iteration should stop.
"""

def iter_ceil(object: torch.Tensor, i: int, max_iters: int = 10) -> bool:
    """
    Simple iteration terminator that stops after a given number of iterations.
    Args:
        _: The object array. Not used in this simple terminator.
        i: The current iteration number.
        max_iters: The maximum number of iterations, default to 10.
    """
    return i >= max_iters
