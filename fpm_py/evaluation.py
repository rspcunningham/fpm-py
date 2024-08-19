"""
The evaluation module contains various quantiative metrics for assessing the performance of the reconstruction algorithm
"""

from .utils import *
from .data import *
from .optimizers import *
from .iteration_terminators import *
from .algorithm import *

import numpy as np
import cv2

def sail(reference_image: np.ndarray, recon_image: np.ndarray) -> float:
    """
    SAIL (Spectrum Alignment Index with Localization) evaluation metric. Value between 0 and 1 to depict the algorithm performance based on the original and reconstructed image.

    Args:
        original_image (np.ndarray): The original image used as reference.
        recon_image (np.ndarray): The reconstructed image as a result of the algorithm. This is the image being evaluated.
    
    Returns:
        float: SAIL value between 0 and 1.
    """

    # Step 1: Check that images are aligned
    if reference_image.shape != recon_image.shape:
        # Resize recon_image to match reference_image shape
        recon_image = cv2.resize(recon_image, (reference_image.shape[1], reference_image.shape[0]))
    
    # Step 2: Convert both images to frequency domain using a 2D Fourier Transform
    original_fft = np.fft.fft2(reference_image)
    recon_fft = np.fft.fft2(recon_image)
    
    # Step 3: Extract phase information from the Fourier Transforms
    original_phase = np.angle(original_fft)
    recon_phase = np.angle(recon_fft)
    
    # Step 4: Calculate the radial frequencies (concentric rings)
    height, width = reference_image.shape
    center_y, center_x = height // 2, width // 2
    y, x = np.indices((height, width))
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2) 
    
    # Step 5: Define number of rings (frequency bands) and compute FRC for each ring based on phase information
    num_rings = min(center_x, center_y)
    frc_values = []
    
    for r in range(1, num_rings):
        # Create a mask for the current ring
        mask = (radius >= r) & (radius < r + 1)
        
        # Calculate the phase difference between the two images for the ring
        phase_diff = np.exp(1j * (original_phase[mask] - recon_phase[mask]))
        
        # Calculate FRC using the phase correlation
        num = np.abs(np.sum(phase_diff))
        denom = np.sqrt(np.sum(np.abs(original_phase[mask])**2) * np.sum(np.abs(recon_phase[mask])**2))
        
        if denom == 0:
            frc_values.append(0)
        else:
            frc_values.append(num / denom)
    
    frc_values = np.array(frc_values)
    
    # Step 6: Generate the FRC curve (using the radial frequencies)
    spatial_frequencies = np.arange(1, num_rings)  # Corresponds to the radius of each ring
    
    # Step 7: Compute Area Under the Curve (AUC) of the FRC curve using trapezoidal rule
    auc = np.trapz(frc_values, spatial_frequencies)
    # Normalize the AUC by the maximum possible AUC (which would be the area if FRC = 1 for all rings)
    max_auc = np.trapz(np.ones_like(frc_values), spatial_frequencies)
    
    # Step 8: Return the normalized AUC as the SAIL score
    sail = auc / max_auc
    return sail


def rmse(reference_image: np.ndarray, recon_image: np.ndarray) -> float:
    """
    RMSE (Root Mean Squared Error) evaluation metric. Value represents the error between the reference and reconstructed image.

    Args:
        reference_image (np.ndarray): The original image used as reference.
        recon_image (np.ndarray): The reconstructed image as a result of the algorithm.
    
    Returns:
        float: RMSE value.
    """
    # Step 1: Ensure images are the same size
    if reference_image.shape != recon_image.shape:
        recon_image = cv2.resize(recon_image, (reference_image.shape[1], reference_image.shape[0]))

    # Step 2: Calculate the RMSE
    error = np.square(reference_image - recon_image).mean()
    rmse_value = np.sqrt(error)
    
    return rmse_value