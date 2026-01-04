"""
The FP forward model: O, P, {k_i}  →  {I_i}
"""
from typing import Callable, cast
from functools import partial
import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))
fftshift = cast(Callable[..., torch.Tensor], torch.fft.fftshift)
ifftshift = cast(Callable[..., torch.Tensor], torch.fft.ifftshift)

def forward_model(
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx: Float[torch.Tensor, "B"],
    ky: Float[torch.Tensor, "B"],
    downsample_factor: int = 1
) -> Complex[torch.Tensor, "B N/{downsample_factor} N/{downsample_factor}"]:
    """
    Forward model - returns images at each k-space location given an object

    Args:
        object_tensor (torch.Tensor): Object tensor [N, N] (0, 1)
        pupil_tensor (torch.Tensor): Pupil tensor [N, N]
        kx (torch.Tensor): Wavevector shift(s) in x direction, normalized. Tensor [B] (-0.5, 0.5)
        ky (torch.Tensor): Wavevector shift(s) in y direction, normalized. Tensor [B] (-0.5, 0.5)
        downsampling_factor (int): Downsampling factor for the output images

    Returns:
        torch.Tensor: Predicted intensities [B, N/downsample_factor, N/downsample_factor]
    """

    N, _ = object_tensor.shape
    dtype = object_tensor.dtype
    kx_reshaped = kx.view(-1, 1, 1)
    ky_reshaped = ky.view(-1, 1, 1)

    # Create coordinate grids [N, N]
    coords = torch.arange(N, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

    # Create phase ramps for all k-vectors at once
    # Phase ramp: exp(i * 2π * (kx*x + ky*y) / N)
    # Shape: [B, N, N]
    phase = 2 * torch.pi * (kx_reshaped * x_grid[None, :, :] + ky_reshaped * y_grid[None, :, :])
    phase_ramps = torch.exp(1j * phase.to(dtype))  # [B, N, N]


    # Apply phase ramps to object (multiply in spatial domain = shift in frequency domain)
    tilted_objects = object_tensor[None, :, :] * phase_ramps  # [B, N, N]

    # Batch FFT all tilted objects
    objects_fourier = fftshift(fft2(tilted_objects), dim=(-2, -1))  # [B, N, N]

    # Apply pupil filter (broadcast over batch dimension)
    filtered_fourier = pupil_tensor[None, :, :] * objects_fourier  # [B, N, N]

    # Batch inverse FFT
    complex_image_fields = ifft2(filtered_fourier)  # [B, N, N]

    # Compute intensities
    predicted_intensities = torch.abs(complex_image_fields)**2  # [B, N, N]

    if downsample_factor > 1:
        predicted_intensities = F.avg_pool2d(predicted_intensities, kernel_size=downsample_factor, stride=downsample_factor)

    return predicted_intensities
