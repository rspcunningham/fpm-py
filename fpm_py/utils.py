import torch

# 2D Fourier transform
def ft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

# 2D Inverse Fourier transform
def ift(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x)))

def kvector_to_x_y(fourier_center: torch.Tensor, image_size: torch.Tensor, du: torch.Tensor, k_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts k-vector to x and y coordinates in the spatial domain.
    Args:
    fourier_center (Tensor): The center of the Fourier domain.
    image_size (Tensor): The size of the image.
    du (Tensor): The effective magnification.
    k_vector (Tensor): The k-vector.
    Returns:
    tuple: The x and y coordinates in the spatial domain.
    """
    fourier_shift = torch.round(k_vector / du).to(torch.int32)
    x, y = (fourier_center + fourier_shift - torch.floor(image_size / 2)).to(torch.int32)
    return x, y

def overlap_matrices(larger: torch.Tensor, smaller: torch.Tensor, bottom: int, left: int) -> torch.Tensor:
    """
    Adds a smaller matrix to a larger matrix at the specified position.
    Args:
    larger (Tensor): The larger matrix.
    smaller (Tensor): The smaller matrix.
    bottom (int): The bottom row index to place the smaller matrix.
    left (int): The left column index to place the smaller matrix.
    Returns:
    Tensor: The updated larger matrix.
    """
    # Get the dimensions of the smaller matrix
    rows, cols = smaller.shape
    # Calculate the starting indices in the larger matrix
    start_row = bottom - rows + 1
    start_col = left
    # Ensure the indices are within the bounds of the larger matrix
    if start_row < 0 or start_col < 0 or start_row + rows > larger.shape[0] or start_col + cols > larger.shape[1]:
        raise ValueError("Matrix B cannot be placed at the specified position in matrix A.")
    
    # Add smaller matrix to the larger matrix
    larger[start_row:start_row + rows, start_col:start_col + cols] += smaller
    return larger

def circle_like(array: torch.Tensor) -> torch.Tensor:
    """
    Creates a circular mask with the same shape as the input array.
    Args:
    array (Tensor): The input array.
    Returns:
    Tensor: The circular mask.
    """
    mask = torch.zeros(array.shape, dtype=torch.bool, device=array.device)
    center_y, center_x = torch.tensor(mask.shape, device=array.device) // 2
    radius = min(center_y, center_x)
    y, x = torch.ogrid[:mask.shape[0], :mask.shape[1]]
    y = y.to(array.device)
    x = x.to(array.device)
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = distance <= radius
    return mask