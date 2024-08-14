"""
This module contains the optimizers used in the FPM reconstruction process. The optimizers are used as arguments in the `reconstruct` function in the `algorithm.py` module. Optimizers update the object and pupil in each iteration of the reconstruction process.

All functions must match the `OptimizerType` type alias:
```python
OptimizerType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int], 
    tuple[torch.Tensor, torch.Tensor]
]
```

"""


import torch
from typing import Callable

import matplotlib.pyplot as plt

from .utils import overlap_matrices, ift, ft

OptimizerType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int], 
    tuple[torch.Tensor, torch.Tensor]
]

def quasi_second_order(
        object: torch.Tensor, 
        pupil: torch.Tensor, 
        wave_fourier: torch.Tensor, 
        wave_fourier_new: torch.Tensor, 
        x: int, 
        y: int,
        alpha_o: float = 1,
        mu_o: float = 1,
        alpha_p: float = 1,
        mu_p: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple gradient descent optimizer with learning rate and regularization hyperparams for object and pupil.
    
    Args:
        object (torch.Tensor): The object.
        pupil (torch.Tensor): The pupil.
        wave_fourier (torch.Tensor): The Fourier domain of the wave.
        wave_fourier_new (torch.Tensor): The new Fourier domain of the wave.
        x (int): Bottom row where pupil overlaps with object.
        y (int): Leftmost column where pupil overlaps with object.
        alpha_o (float): The learning rate for the object.
        mu_o (float): The regularization parameter for the object.
        alpha_p (float): The learning rate for the pupil.
        mu_p (float): The regularization parameter for the pupil.
    
    Returns:
        tuple: The updated object and pupil.
    """

    # Selects only the region of interest for object, leaves everything else alone
    object_region = object[x:x + pupil.shape[0], y:y + pupil.shape[1]]

    delta_wave = wave_fourier_new - wave_fourier

    if torch.sum(delta_wave) == 0:
        return object, pupil

    # Update the object with the correction term
    object = overlap_matrices(object, (
        alpha_o * torch.abs(pupil) * torch.conj(pupil) * delta_wave
    ) / (torch.max(torch.abs(pupil)) * (torch.abs(pupil) ** 2 + mu_o)), x, y)
    
    # Update the pupil with the correction term
    pupil += (
        alpha_p * torch.abs(object_region) * torch.conj(object_region) * delta_wave
    ) / (torch.max(torch.abs(object_region)) * (torch.abs(object_region) ** 2 + mu_p))

    return object, pupil

def tomas(object: torch.Tensor, pupil: torch.Tensor, wave_fourier: torch.Tensor, wave_fourier_new: torch.Tensor, x: int, y: int, alpha: float = 1, beta: float = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tomas' optimizer with learning rate and regularization hyperparams for object and pupil.
    
    Args:
        object (torch.Tensor): The object.
        pupil (torch.Tensor): The pupil.
        wave_fourier (torch.Tensor): The Fourier domain of the wave.
        wave_fourier_new (torch.Tensor): The new Fourier domain of the wave.
        x (int): Bottom row where pupil overlaps with object.
        y (int): Leftmost column where pupil overlaps with object.
    
    Returns:
        tuple: The updated object and pupil.
    """
    # Selects only the region of interest for object, leaves everything else alone
    object_region = object[x:x + pupil.shape[0], y:y + pupil.shape[1]]

    delta_wave = wave_fourier_new - wave_fourier

    if torch.sum(delta_wave) == 0:
        return object, pupil

    # Update the object with the correction term
    object = overlap_matrices(object, 1 / torch.max(torch.abs(pupil)) * torch.abs(pupil) * torch.conj(pupil) * delta_wave / (torch.abs(pupil) ** 2 + alpha), x, y)
    
    # Update the pupil with the correction term
   
    pupil += 1 / torch.max(torch.abs(object_region)) * torch.abs(object_region) * torch.conj(object_region) * delta_wave / (torch.abs(object_region) ** 2 + beta)

    return object, pupil