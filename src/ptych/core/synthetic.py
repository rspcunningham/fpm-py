import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ptych.core.forward import FPMForwardModel


def synthesize_captures(
    object_tensor: Complex[torch.Tensor, "object_height object_width"],
    pupil_tensor: Complex[torch.Tensor, "object_height object_width"],
    illumination_kx: Float[torch.Tensor, "illumination"],
    illumination_ky: Float[torch.Tensor, "illumination"],
    object_to_capture_ratio: int,
) -> Float[torch.Tensor, "illumination height width"]:
    """
    Generate synthetic captures by running forward model and downsampling.

    Args:
        object_tensor: Complex object tensor [object_height, object_width]
        pupil_tensor: Complex pupil tensor [object_height, object_width]
        illumination_kx: Normalized k-vectors in x direction [illumination]
        illumination_ky: Normalized k-vectors in y direction [illumination]
        object_to_capture_ratio: Linear ratio between object and capture grids

    Returns:
        Synthetic captures [illumination, height, width] as float intensities
    """
    # Renormalize k-vectors from n-grid to N-grid
    illumination_kx = illumination_kx / object_to_capture_ratio
    illumination_ky = illumination_ky / object_to_capture_ratio

    # Run forward model at full resolution with one synthetic patch batch element.
    forward_model = FPMForwardModel(object_tensor.shape[-1]).to(object_tensor.device)
    predicted_intensities = forward_model(
        object_tensor[None],
        pupil_tensor[None],
        illumination_kx,
        illumination_ky,
    )  # [1, illumination, object_height, object_width]
    predicted_intensities = predicted_intensities.squeeze(0)

    # Reduce full-resolution intensities to the capture grid with average pooling
    if object_to_capture_ratio > 1:
        # Add channel dimension for avg_pool2d.
        predicted_intensities = predicted_intensities.unsqueeze(1)
        predicted_intensities = F.avg_pool2d(
            predicted_intensities,
            kernel_size=object_to_capture_ratio,
        )
        predicted_intensities = predicted_intensities.squeeze(1)

    return predicted_intensities
