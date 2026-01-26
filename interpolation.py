import torch
import torch.nn.functional as F
from jaxtyping import Float


def interpolate_green(data: Float[torch.Tensor, "B N N"]) -> Float[torch.Tensor, "B N N"]:
    """
    Interpolate R/B positions in a Bayer mosaic using cardinal neighbors.

    Assumes RGGB pattern where green pixels are at positions where (x+y) is odd.
    Vectorized over batch dimension.

    Args:
        data: [B, N, N] tensor of single-channel Bayer mosaic images

    Returns:
        [B, N, N] tensor with R/B positions interpolated from green neighbors
    """
    B, H, W = data.shape

    # Create mask for R/B positions (where x+y is even)
    y_idx = torch.arange(H, device=data.device).view(-1, 1)
    x_idx = torch.arange(W, device=data.device).view(1, -1)
    rb_mask = ((x_idx + y_idx) % 2 == 0)  # [H, W]

    # Pad with edge replication: [B, N, N] -> [B, 1, N, N] -> pad -> [B, 1, N+2, N+2]
    padded = F.pad(data.unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)  # [B, H+2, W+2]

    # Cardinal neighbor averages
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    avg = (up + down + left + right) / 4  # [B, H, W]

    # Apply interpolation only at R/B positions
    result = data.clone()
    result[:, rb_mask] = avg[:, rb_mask]

    return result
