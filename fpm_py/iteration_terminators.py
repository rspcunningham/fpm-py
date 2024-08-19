"""
This module contains functions that determine when to stop the iteration process. The functions are used as arguments in the `reconstruct` function in the `algorithm.py` module. The functions are simple and can be easily replaced with custom functions.

All Iteration Terminator functions can only accept one positional argument of type: TerminatorInputs. This structure is created and passed in by the reconstruction algorithm.

"""

import torch
from typing import Callable
from dataclasses import dataclass

@dataclass
class TerminatorInputs:
    """
    Dataclass for the inputs to the iteration terminator functions. All iteration terminator functions must accept this dataclass as the only positional argument.

    Attributes:
        object (torch.Tensor): The object array. Not used in this simple terminator.
        wave_fourier (torch.Tensor): The current Fourier domain of the wave. None on the first iteration.
        wave_fourier_new (torch.Tensor): The new Fourier domain of the wave. None on the first iteration.
        i (int): The current iteration number.
    
    """
    object: torch.Tensor
    wave_fourier: torch.Tensor
    wave_fourier_new: torch.Tensor
    i: int

TerminatorType = Callable[[TerminatorInputs], bool]

def iter_ceil(inputs: TerminatorInputs, max_iters: int = 10, **kwargs) -> bool:
    """
    Simple iteration terminator that stops after a given number of iterations.
    Args:
        inputs (TerminatorInputs): The inputs to the terminator function.
        max_iters (int): The maximum number of iterations to run.
    Returns:
        bool: True if the iteration number is greater than or equal to the maximum number of iterations
    """
    return inputs.i >= max_iters
