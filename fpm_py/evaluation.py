"""
The evaluation module contains various quantiative metrics for assessing the performance of the reconstruction algorithm
"""

from .utils import *
from .data import *
from .optimizers import *
from .iteration_terminators import *
from .algorithm import *

from scipy.integrate import trapz

def sail(original_image: ImageCapture, recon_image: ImageCapture) -> float:
    """
    SAIL (Spectrum Alignment Index with Localization) evaluation metric. Value between 0 and 1 to depict the algorithm performance based on the original and reconstructed image.

    Args:
        original_image (ImageCapture): The original image used as reference.
        recon_image (ImageCapture): The reconstructed image as a result of the algorithm. This is the image being evaluated.
    
    Returns:
        float: SAIL value between 0 and 1.
    """

    # Step 1: Check that images are aligned, what do we do if they are not aligned? Something with that output scale factor?
    assert original_image.shape == recon_image.shape, "Original and reconstructed images must be of the same size."
    
    # Step 2: Convert both images to frequency domain using a 2D Fourier Transform
    original_fft, recon_fft = ft(original_image), ft(recon_image)
    
    # Step 3: Calculate the radial frequencies (concentric rings)
    height, width = original_image.shape
    center_y, center_x = height // 2, width // 2
    y, x = np.indices((height, width))
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2) 
    
    # Step 4: Define number of rings (frequency bands) and compute FRC for each ring
    num_rings = min(center_x, center_y)
    frc_values = []
    
    for r in range(1, num_rings):
        # Create a mask for the current ring
        mask = (radius >= r) & (radius < r + 1)
        
        # Calculate FRC for the ring
        num = np.sum(original_fft[mask] * np.conj(recon_fft[mask]))
        denom = np.sqrt(np.sum(np.abs(original_fft[mask])**2) * np.sum(np.abs(recon_fft[mask])**2))
        
        if denom == 0:
            frc_values.append(0)
        else:
            frc_values.append(np.abs(num / denom))
    
    frc_values = np.array(frc_values)
    
    # Step 5: Generate the FRC curve (using the radial frequencies)
    spatial_frequencies = np.arange(1, num_rings)  # Corresponds to the radius of each ring
    
    # Step 6: Compute Area Under the Curve (AUC) of the FRC curve using trapezoidal rule
    auc = trapz(frc_values, spatial_frequencies)
    # Normalize the AUC by the maximum possible AUC (which would be the area if FRC = 1 for all rings)
    max_auc = trapz(np.ones_like(frc_values), spatial_frequencies)
    
    # Step 7: Return the normalized AUC as the SAIL score
    sail = auc / max_auc
    return sail


