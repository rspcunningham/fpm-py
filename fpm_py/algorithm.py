"""
The algorithm module contains the core algorithm for reconstructing an object from a series of images. The main function is reconstruct, which takes an `ImageSeries` object and returns the reconstructed object in the spatial domain (as a complex value).
"""

from .utils import *
from .data import *
from .optimizers import *
from .iteration_terminators import *

import torch

def reconstruct(
    image_series: ImageSeries,
    output_scale_factor: int = None,
    pupil_0: torch.Tensor = None,
    iteration_terminator: TerminatorType = iter_ceil,
    optimizer: OptimizerType = tomas
) -> torch.Tensor:
    """
    Core algorithm. Reconstructs an object from a series of images.

    Args:
        image_series (ImageSeries): The series of images to reconstruct.
        output_scale_factor (int): The scale factor of the output image. If not provided, the minimum needed size will be calculated.
        pupil_0 (torch.Tensor): The initial calculation of the pupil function. If not provided, a boolean circular pupil will be created.
        iteration_terminator (TerminatorType): The function that determines when to stop iterating. If not provided, the default is to iterate 10 times.
        optimizer (OptimizerType): The optimizer function that updates the object and pupil. If not provided, the default is the tomas optimizer.
    
    Returns:
        torch.Tensor: The reconstructed object in the spatial domain.

    """


    # Create a default circular pupil if not provided
    if pupil_0 is None:
        print("Pupil not provided. Creating a circular pupil matching images in stack.")
        pupil_0 = circle_like(image_series.image_stack[0].image)
     
    # Check if pupil and image sizes match
    if not torch.equal(image_series.image_size, torch.tensor(pupil_0.shape, device=image_series.device)):
        raise ValueError("The pupil and image sizes do not match.")
    
    # assign the minimum needed output image size
    if output_scale_factor is None:
        k_span = image_series.image_size // 2
        total = k_span + image_series.max_k // image_series.du
        output_image_size = image_series.image_size * total // k_span + 1
        output_image_size = [int(x.item()) for x in output_image_size]
    else: 
        output_image_size = [x * output_scale_factor for x in image_series.image_size]
    
    # Convert pupil_0 to PyTorch tensor and move to device
    pupil_0.to(image_series.device)

    # Initialize values
    fourier_center = (output_image_size[0] // 2, output_image_size[1] // 2)
    pupil_binary = pupil_0.abs() > 0

    # Initialize object and iteration counter
    object = None
    i = 0

    # Main reconstruction loop
    while not iteration_terminator(object, i):
        for _, data in enumerate(image_series.image_stack):

            image = data.image
            k_vector = data.k_vector

            if object is None:
                # Initialize object with first image
                object = torch.zeros(output_image_size, dtype=torch.complex64, device=image_series.device)
                pupil = pupil_0

                x, y = kvector_to_x_y(fourier_center, image_series.image_size, image_series.du, k_vector)

                wave_fourier_new = ft(torch.sqrt(image))
                object = overlap_matrices(object, wave_fourier_new * pupil, x, y)
                
                continue

            # Calculate x and y coordinates for the current image
            x, y = kvector_to_x_y(fourier_center, image_series.image_size, image_series.du, k_vector)


            # Extract the relevant part of the object and multiply by pupil
            wave_fourier = object[x:x+image_series.image_size[0], y:y+image_series.image_size[1]] * pupil

            # Transform to spatial domain
            wave_spatial = ift(wave_fourier)

            # Calculate estimated and measured images
            image_estimated = torch.abs(wave_spatial) ** 2

            image_measured = image
            image_measured = torch.clamp(image_measured, min=0)

            # Update wave_fourier based on measured and estimated images
            wave_fourier_new = ft(
                torch.sqrt(image_measured) * wave_spatial / (torch.sqrt(image_estimated) + torch.finfo(torch.float32).eps)
            )
            #wave_fourier_new = ft(torch.sqrt(image_measured))

            # Constrain pupil to binary mask
            pupil = pupil * pupil_binary

            # Optimize object and pupil
            object, pupil = optimizer(object, pupil, wave_fourier, wave_fourier_new, x, y)


        # Increment iteration counter
        i += 1

    # Return the final reconstructed object in spatial domain
    return ift(object)