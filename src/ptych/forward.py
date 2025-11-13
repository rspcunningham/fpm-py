"""
The FP forward model: O, P, {k_i}  →  {I_i}
"""
from typing import Callable, cast
from functools import partial
import torch
import torch.nn.functional as F

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))
fftshift = cast(Callable[..., torch.Tensor], torch.fft.fftshift)
ifftshift = cast(Callable[..., torch.Tensor], torch.fft.ifftshift)

def forward_model(
    object_tensor: torch.Tensor,
    pupil_tensor: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    downsample_factor: int = 1
) -> torch.Tensor:
    """
    Forward model - returns images at each k-space location given an object

    Args:
        object_tensor (torch.Tensor): Object [H, W]
        pupil_tensor (torch.Tensor): Pupil tensor [H, W]
        kx (torch.Tensor): Wavevector shift(s) in x direction. Tensor [B]
        ky (torch.Tensor): Wavevector shift(s) in y direction. Tensor [B]
        downsampling_factor (int): Downsampling factor for the output images

    Returns:
        torch.Tensor: Predicted intensities [B, H, W]
    """

    H, W = object_tensor.shape
    dtype = object_tensor.dtype

    # Create coordinate grids [H, W]
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Create phase ramps for all k-vectors at once
    # Phase ramp: exp(i * 2π * (kx*x + ky*y) / N)
    # Shape: [B, H, W]
    kx_normalized = kx.view(-1, 1, 1) / W  # Normalize by image size
    ky_normalized = ky.view(-1, 1, 1) / H

    phase = 2 * torch.pi * (kx_normalized * x_grid[None, :, :] + ky_normalized * y_grid[None, :, :])
    phase_ramps = torch.exp(1j * phase.to(dtype))  # [B, H, W]


    # Apply phase ramps to object (multiply in spatial domain = shift in frequency domain)
    tilted_objects = object_tensor[None, :, :] * phase_ramps  # [B, H, W]

    # Batch FFT all tilted objects
    objects_fourier = fftshift(fft2(tilted_objects), dim=(-2, -1))  # [B, H, W]

    # Apply pupil filter (broadcast over batch dimension)
    filtered_fourier = pupil_tensor[None, :, :] * objects_fourier  # [B, H, W]

    # Batch inverse FFT
    complex_image_fields = ifft2(filtered_fourier)  # [B, H, W]

    # Compute intensities
    predicted_intensities = torch.abs(complex_image_fields)**2  # [B, H, W]

    if downsample_factor > 1:
        predicted_intensities = F.avg_pool2d(predicted_intensities, kernel_size=downsample_factor, stride=downsample_factor)

    return predicted_intensities
