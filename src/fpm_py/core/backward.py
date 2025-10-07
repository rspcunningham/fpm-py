from __future__ import annotations

from typing import Optional, List
import numpy as np
from numpy.typing import NDArray
import torch


from ..utils.math_utils import ft, ift  # Only import what we need
from .structs import ImageCapture, ImageSeries, AcquisitionSettings  # `AcquisitionSettings` holds .du
from ..utils.data_utils import auto_place, best_device  # device selection and placement


"""
Backward Model
"""

__all__ = ["kvectors_to_image_series", "spatial_to_kvectors"]

def spatial_to_kvectors(
    led_positions: NDArray[np.float64],  # Shape (N, 3) for x, y, z positions
    wavelength: float,          # Light wavelength in meters
    pixel_size: float,          # Camera pixel size in meters
    sample_to_lens_distance: float,  # Distance in meters
    lens_to_sensor_distance: float,  # Distance in meters
) -> torch.Tensor:
    """
    Convert physical LED positions in 3D space to k-vectors in frequency space.
    
    Args:
        led_positions: Array of LED positions in 3D space (x, y, z), with (0, 0, 0) at optical axis
        wavelength: Light wavelength in meters
        pixel_size: Camera pixel size in meters
        sample_to_lens_distance: Distance from sample to lens in meters
        lens_to_sensor_distance: Distance from lens to sensor in meters
        
    Returns:
        Tensor of k-vectors in frequency space, shape (N, 2)
    """
    device = best_device()
    # Extract LED positions
    x, y, z = led_positions[:, 0], led_positions[:, 1], led_positions[:, 2]
    
    # Calculate illumination angles (from LED position to sample)
    # Note: z is the optical axis, and we're calculating angles from this axis
    radial_distance = np.sqrt(x**2 + y**2)
    # Angle from the optical axis for each LED
    theta = np.arctan2(radial_distance, z)
    # Azimuthal angle around the optical axis
    phi = np.arctan2(y, x)
    
    # Convert to direction cosines (sin(θ))
    sin_theta = np.sin(theta)
    
    # Calculate x and y components of the k-vector
    # k = (2π/λ) * sin(θ)
    k_magnitude = 2 * np.pi * sin_theta / wavelength
    kx = k_magnitude * np.cos(phi)
    ky = k_magnitude * np.sin(phi)
    
    # Scale by pixel size and account for magnification
    magnification = lens_to_sensor_distance / sample_to_lens_distance
    kx_scaled = kx * pixel_size / magnification
    ky_scaled = ky * pixel_size / magnification
    
    # Combine into k-vectors
    k_vectors = np.stack((kx_scaled, ky_scaled), axis=1)
    
    # Convert to torch tensor
    return torch.tensor(k_vectors, dtype=torch.float32, device=device)


def kvectors_to_image_series(
    obj: torch.Tensor,           # Complex object field
    k_vectors: torch.Tensor,     # Shape (N, 2) of k-vectors
    wavelength: float = 550e-9,  # Default wavelength (green)
    pixel_size: float = 1e-6,    # Default pixel size (1μm)
    magnification: float = 1.0,   # System magnification (default=1.0)
    pupil_radius: Optional[int] = None,  # Pupil radius in pixels (NA-related)
    pupil: Optional[torch.Tensor] = None  # Optional custom pupil function
) -> ImageSeries:
    """
    Simulate FPM captures directly from k-vectors.
    
    Args:
        obj: Complex object field
        k_vectors: Tensor of k-vectors in frequency space
        pupil_radius: Radius of the pupil in pixels
        wavelength: Light wavelength in meters
        pixel_size: Camera pixel size in meters
        pupil: Optional custom pupil function (overrides pupil_radius if provided)
        magnification: System magnification (only used for du calculation if du=None)
        
    Returns:
        ImageSeries containing all simulated captures
    """
    device = obj.device
    H, W = obj.shape
    
    # Effective pixel size in object plane
    effective_pixel_size = pixel_size / magnification
    # Calculate du: wavelength / (N * pixel_size)
    # Using min(H, W) as the relevant dimension to ensure properly scaled results
    # note that du in this context is the scale between k-space in pixels and in radians, NOT the sampling interval in k-space
    du = wavelength / (min(H, W) * effective_pixel_size)
    print(f"Calculated du = {du:.6f} from optical parameters")
    
    # Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device) - H // 2,
        torch.arange(W, device=device) - W // 2,
        indexing="ij"
    )
    
    # Create pupil mask (binary circle by default)
    if pupil is None:
        if pupil_radius is None:
            # Default to a reasonable pupil size if not specified
            pupil_radius = min(H, W) // 10
        
        print(f"Using pupil with radius {pupil_radius} pixels")
        r_squared = x_grid**2 + y_grid**2
        pupil_mask = (r_squared <= pupil_radius**2).to(torch.complex64)
    else:
        pupil_mask = pupil.to(device).to(torch.complex64)
        print("Using custom pupil function")
    
    # Print k-vector information
    k_min = k_vectors.min(dim=0)[0]
    k_max = k_vectors.max(dim=0)[0]
    print(f"K-vectors range: ({k_min[0].item():.2f}, {k_min[1].item():.2f}) to "
          f"({k_max[0].item():.2f}, {k_max[1].item():.2f})")
    
    # Create list to store captures
    captures: List[ImageCapture] = []
    
    # Process each k-vector (illumination angle)
    for kv in k_vectors:
        # Extract k-vector components
        kx, ky = kv[0].item(), kv[1].item()
        
        # Create phase ramp in spatial domain (represents tilted wavefront)
        # exp(j * (kx*x + ky*y)) in the physical model
        phase_ramp = torch.exp(1j * (kx * x_grid + ky * y_grid))
        
        # Multiply object by phase ramp (simulating angled illumination)
        obj_illuminated = obj * phase_ramp
        
        # Get Fourier transform of illuminated object
        obj_F_illuminated = ft(obj_illuminated)
        
        # Apply pupil at center of Fourier space (optical transfer function)
        spectrum_masked = obj_F_illuminated * pupil_mask
        
        # Convert back to spatial domain
        wave_spatial = ift(spectrum_masked)
        
        # Camera only records intensity (squared magnitude)
        intensity = (wave_spatial.abs() ** 2).float()
        
        # Store as an ImageCapture with corresponding k-vector
        captures.append(ImageCapture(image=intensity, k_vector=kv))
    
    # Pack into ImageSeries
    settings = AcquisitionSettings(
        du=du,
        pixel_size=pixel_size,
        wavelength=wavelength
    )
    
    series = ImageSeries(captures=captures, settings=settings)
    return auto_place(series)  # Handle device placement consistently
