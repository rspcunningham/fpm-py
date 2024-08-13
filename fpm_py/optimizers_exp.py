import torch
from typing import Callable

import matplotlib.pyplot as plt

from .utils import overlap_matrices, ift, ft

DEBUG = False


OptimizerType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int], 
    tuple[torch.Tensor, torch.Tensor]
]

def simple_grad_descent(
        object: torch.Tensor, 
        pupil: torch.Tensor, 
        wave_fourier: torch.Tensor, 
        wave_fourier_new: torch.Tensor, 
        x: int, 
        y: int,
        alpha_o: float = 1,
        mu_o: float = 0,
        alpha_p: float = 1,
        mu_p: float = 0
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

    if DEBUG:
        
        print(f'{pupil.shape=}')
        print(f'{object_region=}')

        plt.imshow(torch.abs(ift(wave_fourier)).cpu().numpy())
        plt.title("wave_fourier (opt)")
        plt.show()

        plt.imshow(torch.abs(ift(wave_fourier_new)).cpu().numpy())
        plt.title("wave_fourier_new (opt)")
        plt.show()

        plt.imshow(torch.abs(ift(delta_wave)).cpu().numpy())
        plt.title("delta_wave (opt)")
        plt.show()


    # Update the object with the correction term
    object = overlap_matrices(object, (
        alpha_o * torch.abs(pupil) * torch.conj(pupil) * delta_wave
    ) / (torch.max(torch.abs(pupil)) * (torch.abs(pupil) ** 2 + mu_o)), x, y)
    
    if DEBUG:
        plt.imshow(torch.abs(ift(object)).cpu().numpy())
        plt.title("object (opt)")
        plt.show()
    
    # Update the pupil with the correction term
    
    
    pupil += (
        alpha_p * torch.abs(object_region) * torch.conj(object_region) * delta_wave
    ) / (torch.max(torch.abs(object_region)) * (torch.abs(object_region) ** 2 + mu_p))

    if DEBUG:
        plt.imshow(torch.abs(ift(pupil)).cpu().numpy())
        plt.title("pupil (opt)")
        plt.show()

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
    if DEBUG:
        print(f'{object_region=}')
        print(f'{torch.max(torch.abs(object_region))=}')
        print(f'{torch.abs(object_region)=}')
        print(f'{torch.conj(object_region)=}')
        print(f'{delta_wave=}')
        print(f'{torch.abs(object_region) ** 2=}')
        print(f'{1 / torch.max(torch.abs(object_region))=}')
    
    pupil_delta = 1 / torch.max(torch.abs(object_region)) * torch.abs(object_region) * torch.conj(object_region) * delta_wave / (torch.abs(object_region) ** 2 + beta)

    if DEBUG:
        plt.imshow(torch.abs(ift(pupil_delta)).cpu().numpy())
        plt.title("pupil_delta (opt)")
        plt.show()

    pupil += pupil_delta

    return object, pupil