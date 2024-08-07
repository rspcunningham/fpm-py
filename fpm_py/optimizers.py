import numpy as np
from typing import Callable

from .utils import overlap_matrices

OptimizerType = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int], 
    tuple[np.ndarray, np.ndarray]
]

def simple_grad_descent(
        object: np.ndarray, 
        pupil: np.ndarray, 
        wave_fourier: np.ndarray, 
        wave_fourier_new: np.ndarray, 
        x: int, 
        y: int,
        alpha_o = 1,
        mu_o = 1,
        alpha_p = 1,
        mu_p = 1
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple gradient descent optimizer with learning rate and regularization hyperparams for object and pupil.
    
    Args:
        object (ndarray): The object.
        pupil (ndarray): The pupil.
        wave_fourier (ndarray): The Fourier domain of the wave.
        wave_fourier_new (ndarray): The new Fourier domain of the wave.
        x (int): bottom row where pupil overlaps with object.
        y (int): leftmost column where pupil overlaps with object.
        alpha_o (float): The learning rate for the object.
        mu_o (float): The regularization parameter for the object.
        alpha_p (float): The learning rate for the pupil.
        mu_p (float): The regularization parameter for the pupil.
    Returns:
        tuple: The updated object and pupil.
    """

    # Selects only the region of interest for O, leaves everything else alone
    pupil_dims = np.asarray(np.shape(pupil))
    object_region = object[x:x+pupil_dims[0], y:y+pupil_dims[1]]

    delta_wave = wave_fourier_new - wave_fourier

    object = overlap_matrices(object, (
        alpha_o * np.abs(pupil) * np.conj(pupil) * delta_wave
    ) / (np.max(np.abs(pupil)) * (np.abs(pupil) ** 2 + mu_o)), x, y)
    
    # Update the pupil with the correction term
    pupil += (
        alpha_p * np.abs(object_region) * np.conj(object_region) * delta_wave
    ) / (np.max(np.abs(object_region)) * (np.abs(object_region) ** 2 + mu_p))

    return object, pupil