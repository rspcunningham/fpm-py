import numpy as np
from .utils import *
from .data import *
from .optimizers import *
from .iteration_terminators import *

def reconstruct(
        stack: list[ImageCapture],
        effective_magnification: float, 
        output_scale_factor: int = 10, 
        pupil_0: np.ndarray = None, 
        iteration_terminator: TerminatorType = iter_ceil,
        optimizer: OptimizerType = simple_grad_descent
    ) -> np.ndarray[np.complex128]:

    """
    Reconstructs the object from a stack of captured images.
    Args:
        stack (list[ImageCapture]): The stack of captured images.
        effective_magnification (float): The effective magnification.
        output_scale_factor (int): The scale factor for the output image.
        pupil_0 (ndarray): The initial pupil.
        iteration_terminator (TerminatorType): The iteration termination condition.
        optimizer (OptimizerType): The optimizer function.
    Returns:
        ndarray: The reconstructed object.
    """

    if pupil_0 is None:
        print("Pupil not provided. Creating a circular pupil matching images in stack.")
        pupil_0 = circle_like(stack[0].image)

    if stack[0].image.shape != pupil_0.shape:
        raise ValueError("The pupil and image sizes do not match.")

    # initialize values
    image_size = np.array(stack[0].image.shape)
    output_image_size = image_size * output_scale_factor
    du = effective_magnification / image_size
    fourier_center = np.round(output_image_size / 2).astype(np.int32)
    pupil_binary = pupil_0 > 0

    i = 0

    while not iteration_terminator(object, i):
        for j, data in enumerate(stack):
           
            if object is None:
                # Initialize object with first image
                object = np.zeros(output_image_size).astype(np.complex128)
                pupil = pupil_0
                x, y = kvector_to_x_y(fourier_center, image_size, du, data.k_vector)

                wave_fourier_new = ft(np.sqrt(data.image))

                overlap_matrices(object, wave_fourier_new * pupil, x, y)
                continue

            x, y = kvector_to_x_y(fourier_center, image_size, du, data.k_vector)
            
            wave_fourier = object[x:x+image_size[0], y:y+image_size[1]] * pupil
            wave_spatial = ift(wave_fourier)

            image_estimated = np.abs(wave_spatial) ** 2
            image_measured = data.image

            wave_fourier_new = ft(
                np.sqrt(image_measured) * wave_spatial / (np.sqrt(image_estimated) + np.finfo(np.float64).eps)
                )
            
            pupil = pupil * pupil_binary # constrain pupil to binary mask
            
            object, pupil = optimizer(object, pupil, wave_fourier, wave_fourier_new, x, y)

        i += 1
    
    return ift(object)