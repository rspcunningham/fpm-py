"""alg_utils.py
Shared utilities for the FPM forward model.

These are pulled out of *forward.py* to avoid duplication and keep the
reconstruction routine concise.  Nothing here touches external state,
so importing is inexpensive.
"""
from __future__ import annotations

import torch

__all__ = [
    "ft",
    "ift",
    "kvector_to_x_y",
    "overlap_matrices",
    "circle_like",
]

# -----------------------------------------------------------------------------
#  Fourier helpers
# -----------------------------------------------------------------------------


def ft(x: torch.Tensor) -> torch.Tensor:
    """Centered 2-D forward FFT (no gradient tracking)."""

    with torch.no_grad():
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))


def ift(x: torch.Tensor) -> torch.Tensor:
    """Centered 2-D inverse FFT (no gradient tracking)."""

    with torch.no_grad():
        return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x)))


# -----------------------------------------------------------------------------
#  Geometry helpers
# -----------------------------------------------------------------------------


def kvector_to_x_y(
    fourier_center: tuple[int, int],
    image_size: torch.Tensor,
    du: float,
    k_vector: torch.Tensor,
    obj_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Map a *k-vector* to integer crop coordinates inside the global spectrum.
    
    Args:
        fourier_center: Center coordinates (row, col) of the global spectrum
        image_size: Size of the sub-image to be placed [H, W]
        du: conversion factor from k-space to pixel units
        k_vector: Physical k-vector for this sub-image
        obj_size: Optional size of target object for bounds validation [H, W]
        
    Returns:
        (x, y): Top-left coordinates for placing the sub-image in global space
        
    Raises:
        AssertionError: If the calculated coordinates would index outside
                        the target object bounds (when obj_size is provided)
    """

    # `k_vector` is physical units; convert to discrete pixels then round.
    shift = torch.round(k_vector / du).to(torch.int64)
    image_center = image_size // 2

    x = int(fourier_center[0] + shift[0].item() - image_center[0].item())
    y = int(fourier_center[1] + shift[1].item() - image_center[1].item())
    
    # Validate indices if obj_size is provided
    if obj_size is not None:
        # Check x bounds
        assert 0 <= x <= obj_size[0] - image_size[0].item(), (
            f"x index {x} out of bounds [0, {obj_size[0] - image_size[0].item()}]. "
            f"Consider increasing output size to accommodate k-vector {k_vector}."
        )
        # Check y bounds
        assert 0 <= y <= obj_size[1] - image_size[1].item(), (
            f"y index {y} out of bounds [0, {obj_size[1] - image_size[1].item()}]. "
            f"Consider increasing output size to accommodate k-vector {k_vector}."
        )
    
    return x, y


# -----------------------------------------------------------------------------
#  Array stitching helpers
# -----------------------------------------------------------------------------


def overlap_matrices(
    larger: torch.Tensor,
    smaller: torch.Tensor,
    bottom: int,
    left: int,
) -> torch.Tensor:
    """Add *smaller* into *larger* in-place at (bottom, left).

    The *bottom*/*left* indices follow the original MATLAB-style convention
    used in the historical code: *(bottom, left)* refers to the bottom-right
    corner of the crop.  The function converts this to a top-left corner and
    performs an in-place +=.
    """

    rows, cols = smaller.shape[-2:]
    start_row = bottom - rows + 1
    start_col = left

    if (
        start_row < 0
        or start_col < 0
        or start_row + rows > larger.shape[-2]
        or start_col + cols > larger.shape[-1]
    ):
        raise ValueError(f"Smaller matrix cannot be placed at the specified position. "
                         f"start_row: {start_row}, start_col: {start_col}, "
                         f"rows: {rows}, cols: {cols}, "
                         f"larger shape: {larger.shape}, "
                         f"smaller shape: {smaller.shape}")

    larger[start_row : start_row + rows, start_col : start_col + cols].add_(smaller)
    return larger


def circle_like(array: torch.Tensor) -> torch.Tensor:
    """Return a complex64 circular mask of the same shape as *array*."""

    mask = torch.zeros(array.shape, dtype=torch.bool, device=array.device)
    center_y, center_x = torch.tensor(mask.shape, device=array.device) // 2
    radius = min(center_y, center_x)

    yy = torch.arange(mask.shape[0], device=array.device).view(-1, 1)
    xx = torch.arange(mask.shape[1], device=array.device).view(1, -1)
    dist = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = dist <= radius
    return mask.to(torch.complex64)
