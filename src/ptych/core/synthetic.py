import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ptych.core.forward import forward_model

def synthesize_captures(
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    downsample_ratio: int,
) -> Float[torch.Tensor, "B n n"]:
    """
    Generate synthetic captures by running forward model and downsampling.

    Args:
        object_tensor: Complex object tensor [N, N]
        pupil_tensor: Complex pupil tensor [N, N]
        kx_batch: Normalized k-vectors in x direction [B]
        ky_batch: Normalized k-vectors in y direction [B]
        downsample_ratio: Factor to downsample by (N / n)

    Returns:
        Synthetic captures [B, n, n] as float intensities
    """
    # Renormalize k-vectors from n-grid to N-grid
    kx_batch = kx_batch / downsample_ratio
    ky_batch = ky_batch / downsample_ratio

    # Run forward model at full resolution
    intensities = forward_model(object_tensor, pupil_tensor, kx_batch, ky_batch)  # [B, N, N]

    # Downsample using average pooling
    if downsample_ratio > 1:
        # Add channel dimension for avg_pool2d: [B, 1, N, N]
        intensities = intensities.unsqueeze(1)
        intensities = F.avg_pool2d(intensities, kernel_size=downsample_ratio)
        intensities = intensities.squeeze(1)  # [B, n, n]

    return intensities
