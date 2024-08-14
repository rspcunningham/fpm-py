"""
This module contains the optimizers used in the FPM reconstruction process. The optimizers are used as arguments in the `reconstruct` function in the `algorithm.py` module. Optimizers update the object and pupil in each iteration of the reconstruction process.

All Optimizer functions can only accept one positional argument of type: OptimizerInputs. This structure is created and passed in by the reconstruction algorithm.

"""


import torch
from typing import Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .utils import overlap_matrices, ift, ft

@dataclass
class OptimizerInputs:
    """
    Dataclass for the inputs to the optimizer functions. All optimizer functions must accept this dataclass as the only positional argument.

    Attributes:
        object (torch.Tensor): The object array.
        pupil (torch.Tensor): The pupil array.
        wave_fourier (torch.Tensor): The current Fourier domain of the wave.
        wave_fourier_new (torch.Tensor): The new Fourier domain of the wave.
        x (int): Bottom row where pupil overlaps with object.
        y (int): Leftmost column where pupil overlaps with object.
    
    """
    object: torch.Tensor
    pupil: torch.Tensor
    wave_fourier: torch.Tensor
    wave_fourier_new: torch.Tensor
    x: int
    y: int

OptimizerType = Callable[
    [OptimizerInputs], 
    tuple[torch.Tensor, torch.Tensor]
]

def quasi_second_order(
        inputs: OptimizerInputs,
        alpha_o: float = 1,
        mu_o: float = 1,
        alpha_p: float = 1,
        mu_p: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple gradient descent optimizer with learning rate and regularization hyperparams for object and pupil.
    
    Args:
        inputs (OptimizerInputs): The inputs to the optimizer function.
        alpha_o (float): Regularization hyperparameter for object.
        mu_o (float): Learning rate hyperparameter for object.
        alpha_p (float): Regularization hyperparameter for pupil.
        mu_p (float): Learning rate hyperparameter for pupil.
    
    Returns:
        tuple: The updated object and pupil.
    """

    # Selects only the region of interest for object, leaves everything else alone
    object_region = inputs.object[inputs.x:inputs.x + inputs.pupil.shape[0], inputs.y:inputs.y + inputs.pupil.shape[1]]

    delta_wave = inputs.wave_fourier_new - inputs.wave_fourier

    if torch.sum(delta_wave) == 0:
        return inputs.object, inputs.pupil

    # Update the object with the correction term
    inputs.object = overlap_matrices(inputs.object, (
        alpha_o * torch.abs(inputs.pupil) * torch.conj(inputs.pupil) * delta_wave
    ) / (torch.max(torch.abs(inputs.pupil)) * (torch.abs(inputs.pupil) ** 2 + mu_o)), inputs.x, inputs.y)
    
    # Update the pupil with the correction term
    inputs.pupil += (
        alpha_p * torch.abs(object_region) * torch.conj(object_region) * delta_wave
    ) / (torch.max(torch.abs(object_region)) * (torch.abs(object_region) ** 2 + mu_p))

    return inputs.object, inputs.pupil

def tomas(inputs: OptimizerInputs, alpha: float = 1, beta: float = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tomas' optimizer with learning rate and regularization hyperparams for object and pupil.

    Args:
        inputs (OptimizerInputs): The inputs to the optimizer function.
        alpha (float): The learning rate for the object.
        beta (float): The learning rate for the pupil.
    
    Returns:
        tuple: The updated object and pupil.
    """
    # Selects only the region of interest for object, leaves everything else alone
    object_region = inputs.object[inputs.x:inputs.x + inputs.pupil.shape[0], inputs.y:inputs.y + inputs.pupil.shape[1]]

    delta_wave = inputs.wave_fourier_new - inputs.wave_fourier

    if torch.sum(delta_wave) == 0:
        return inputs.object, inputs.pupil

    # Update the object with the correction term
    inputs.object = overlap_matrices(inputs.object, 1 / torch.max(torch.abs(inputs.pupil)) * torch.abs(inputs.pupil) * torch.conj(inputs.pupil) * delta_wave / (torch.abs(inputs.pupil) ** 2 + alpha), inputs.x, inputs.y)
    
    # Update the pupil with the correction term
   
    inputs.pupil += 1 / torch.max(torch.abs(object_region)) * torch.abs(object_region) * torch.conj(object_region) * delta_wave / (torch.abs(object_region) ** 2 + beta)

    return inputs.object, inputs.pupil