"""
Utility functions for the FPM reconstruction process.
"""

import torch

# 2D Fourier transform
def ft(x: torch.Tensor) -> torch.Tensor:
    """
    Simple macro for 2D Fourier transform.
    Args:
    x (torch.Tensor): The input image.
    Returns:
    torch.Tensor: The Fourier transform of the input image."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

# 2D Inverse Fourier transform
def ift(x: torch.Tensor) -> torch.Tensor:
    """
    Simple macro for 2D Inverse Fourier transform.
    Args:
    x (torch.Tensor): The input Fourier domain image.
    Returns:
    torch.Tensor: The inverse Fourier transform of the input image.
    """
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x)))

def kvector_to_x_y(fourier_center: tuple[int, int], image_size: tuple[int, int], du: float, k_vector: torch.Tensor) -> tuple[int, int]:
    """
    Converts k-vector to x and y coordinates in the spatial domain.

    Args:
    fourier_center (tuple[int, int]): The center of the Fourier domain image.
    image_size (tuple[int, int]): The size of the image.
    du (float): The pixel size in the Fourier domain.
    k_vector (torch.Tensor): The k-vector associated with the image.

    Returns:
    tuple[int, int]: The x and y coordinates in the spatial domain.
    
    """
    fourier_shift = (k_vector[0] // du, k_vector[1] // du)
    image_center = (image_size[0] // 2, image_size[1] // 2)
    x = fourier_center[0] + fourier_shift[0] - image_center[0]
    y = fourier_center[1] + fourier_shift[1] - image_center[1]

    return int(x), int(y)

def overlap_matrices(larger: torch.Tensor, smaller: torch.Tensor, bottom: int, left: int) -> torch.Tensor:
    """
    Adds a smaller matrix to a larger matrix at the specified position.
    Args:
    larger (torch.Tensor): The larger matrix.
    smaller (torch.Tensor): The smaller matrix.
    bottom (int): The bottom row index to place the smaller matrix.
    left (int): The left column index to place the smaller matrix.
    Returns:
    torch.Tensor: The updated larger matrix.
    """
    # Get the dimensions of the smaller matrix
    rows, cols = smaller.shape
    # Calculate the starting indices in the larger matrix
    start_row = bottom
    start_col = left
    # Ensure the indices are within the bounds of the larger matrix
    if start_row < 0 or start_col < 0 or start_row + rows > larger.shape[0] or start_col + cols > larger.shape[1]:
        print(start_row, start_col, smaller.shape, larger.shape)
        raise ValueError("Matrix B cannot be placed at the specified position in matrix A.")
    
    # Add smaller matrix to the larger matrix
    larger[start_row:start_row + rows, start_col:start_col + cols] += smaller
    return larger

def circle_like(array: torch.Tensor) -> torch.Tensor:
    """
    Creates a complex-valued circular mask with the same shape as the input array.
    Args:
    array (torch.Tensor): The input array.
    Returns:
    torch.Tensor: The circular mask.
    """
    mask = torch.zeros(array.shape, dtype=torch.bool, device=array.device)
    center_y, center_x = torch.tensor(mask.shape, device=array.device) // 2
    radius = min(center_y, center_x)

    y = torch.arange(mask.shape[0], device=array.device).view(-1, 1)
    x = torch.arange(mask.shape[1], device=array.device).view(1, -1)

    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = distance <= radius
    mask = mask.to(torch.complex64)
    return mask