"""
The FP forward model: O, P, {k_i}  →  {I_i}
"""
from typing import Callable, cast
import torch

fft2 = cast(Callable[[torch.Tensor], torch.Tensor], torch.fft.fft2)
fftshift = cast(Callable[[torch.Tensor], torch.Tensor], torch.fft.fftshift)
ifft2 = cast(Callable[[torch.Tensor], torch.Tensor], torch.fft.ifft2)
ifftshift = cast(Callable[[torch.Tensor], torch.Tensor], torch.fft.ifftshift)

def forward_model(object_tensor: torch.Tensor, pupil_tensor: torch.Tensor, kx: int, ky: int) -> torch.Tensor:
    """
    Forward model - returns images at each k-space location given an object

    Args:
        object_tensor (torch.Tensor): Object [H, W]
        pupil_tensor (torch.Tensor): Pupil tensor [H, W]
        kx (int): Wavevector shift in x direction in cycles/pixel
        ky (int): Wavevector shift in y direction in cycles/pixel

    Returns:
        torch.Tensor: Forward model result.
    """

    # Step 1: Compute the Fourier transform of the object
    object_fourier = fftshift(fft2(object_tensor))

    # Step 2: circularly shift the spectrum by the wavevector shifts
    object_fourier_shifted = torch.roll(object_fourier, shifts=(ky, kx), dims=(0, 1))

    filtered_object_fourier_shifted = pupil_tensor * object_fourier_shifted

    complex_image_field = ifft2(filtered_object_fourier_shifted)

    predicted_intensity = torch.abs(complex_image_field)**2

    return predicted_intensity
