from typing import Callable, cast
from functools import partial
import torch
from jaxtyping import Complex, Float

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))

def forward_model(
    object_tensor: Complex[torch.Tensor, "T N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx: Float[torch.Tensor, "B"],
    ky: Float[torch.Tensor, "B"]
) -> Float[torch.Tensor, "T B N N"]:
    """
    Forward model - returns images at each k-space location given an object

    Args:
        object_tensor (torch.Tensor): Object tensor [T, N, N] (0, 1)
        pupil_tensor (torch.Tensor): Pupil tensor [N, N] -- DC at [0, 0]
        kx (torch.Tensor): Wavevector shift(s) in x direction, normalized. Tensor [B] (-0.5, 0.5)
        ky (torch.Tensor): Wavevector shift(s) in y direction, normalized. Tensor [B] (-0.5, 0.5)

    Returns:
        torch.Tensor: Predicted intensities [T, B, N, N]
    """

    T, N, _ = object_tensor.shape
    dtype = object_tensor.dtype
    device = object_tensor.device
    kx_reshaped = kx.view(-1, 1, 1)
    ky_reshaped = ky.view(-1, 1, 1)

    # Create coordinate grids [N, N]
    coords = torch.arange(N, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing='ij')

    # Create phase ramps for all k-vectors at once
    # Phase ramp: exp(i * 2π * (kx*x + ky*y) / N)
    # Shape: [B, N, N]
    phase = 2 * torch.pi * (kx_reshaped * x_grid[None, :, :] + ky_reshaped * y_grid[None, :, :])
    phase_ramps = torch.exp(1j * phase.to(dtype))  # [B, N, N]


    # Apply phase ramps to object (multiply in spatial domain = shift in frequency domain)
    tilted_objects = object_tensor[:, None] * phase_ramps[None]  # [T, B, N, N]

    # Batch FFT all tilted objects
    objects_fourier = fft2(tilted_objects)  # [T, B, N, N]

    # Apply pupil filter (broadcast over tile and batch dimensions)
    filtered_fourier = pupil_tensor * objects_fourier  # [T, B, N, N]

    # Batch inverse FFT
    complex_image_fields = ifft2(filtered_fourier)  # [T, B, N, N]

    # Compute intensities
    predicted_intensities = torch.abs(complex_image_fields)**2  # [T, B, N, N]

    return predicted_intensities
