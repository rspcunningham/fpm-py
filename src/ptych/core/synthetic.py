import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ptych.core.physics import FPMForwardModel


def synthesize_captures(
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    object_to_capture_ratio: int,
) -> Float[torch.Tensor, "B n n"]:
    """
    Generate synthetic captures by running forward model and downsampling.

    Args:
        object_tensor: Complex object tensor [N, N]
        pupil_tensor: Complex pupil tensor [N, N]
        kx_batch: Normalized k-vectors in x direction [B]
        ky_batch: Normalized k-vectors in y direction [B]
        object_to_capture_ratio: Linear ratio between object and capture grids (N / n)

    Returns:
        Synthetic captures [B, n, n] as float intensities
    """
    # Renormalize k-vectors from n-grid to N-grid
    kx_batch = kx_batch / object_to_capture_ratio
    ky_batch = ky_batch / object_to_capture_ratio

    # Run forward model at full resolution (add T=1 batch dim)
    forward_model = FPMForwardModel(object_tensor.shape[-1]).to(object_tensor.device)
    predicted_intensities = forward_model(
        object_tensor[None],
        pupil_tensor[None],
        kx_batch,
        ky_batch,
    )  # [1, B, N, N]
    predicted_intensities = predicted_intensities.squeeze(0)  # [B, N, N]

    # Reduce full-resolution intensities to the capture grid with average pooling
    if object_to_capture_ratio > 1:
        # Add channel dimension for avg_pool2d: [B, 1, N, N]
        predicted_intensities = predicted_intensities.unsqueeze(1)
        predicted_intensities = F.avg_pool2d(
            predicted_intensities,
            kernel_size=object_to_capture_ratio,
        )
        predicted_intensities = predicted_intensities.squeeze(1)  # [B, n, n]

    return predicted_intensities
