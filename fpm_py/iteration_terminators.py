"""
This module contains functions that determine when to stop the iteration process. The functions are used as arguments in the `reconstruct` function in the `algorithm.py` module. The functions are simple and can be easily replaced with custom functions.

All functions must match the `TerminateType` type alias:
```python
TerminatorType = Callable[
    [torch.Tensor, int], 
    bool
]
```

"""


import torch
from typing import Callable

TerminatorType = Callable[
    [torch.Tensor, int], 
    bool
]

def iter_ceil(object: torch.Tensor, i: int, max_iters: int = 10) -> bool:
    """
    Simple iteration terminator that stops after a given number of iterations.
    Args:
        _: The object array. Not used in this simple terminator.
        i: The current iteration number.
        max_iters: The maximum number of iterations, default to 10.
    """
    return i >= max_iters
