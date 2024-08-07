import numpy as np

"""2D Fourier transform"""
ft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

"""2D Inverse Fourier transform"""
ift = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def kvector_to_x_y(fourier_center, image_size, du, k_vector):
    """
    Converts k-vector to x and y coordinates in the spatial domain.
    Args:
        fourier_center (ndarray): The center of the Fourier domain.
        image_size (ndarray): The size of the image.
        du (float): The effective magnification.
        k_vector (ndarray): The k-vector.
    Returns:
        tuple: The x and y coordinates in the spatial domain.
    """
    fourier_shift = np.round(k_vector / du).astype(np.int32)
    x, y = (fourier_center + fourier_shift - np.floor(image_size / 2)).astype(np.int32)
    return x, y


def overlap_matrices(larger: np.ndarray, smaller: np.ndarray, bottom: int, left: int):
    """
    Adds a smaller matrix to a larger matrix at the specified position.
    Args:
        larger (ndarray): The larger matrix.
        smaller (ndarray): The smaller matrix.
        bottom (int): The bottom row index to place the smaller matrix.
        left (int): The left column index to place the smaller matrix.
    Returns:
        ndarray: The updated larger matrix.
    """
    # Get the dimensions of the smaller matrix
    rows, cols = smaller.shape
    
    # Calculate the starting indices in A
    start_row = bottom - rows + 1
    start_col = left
    
    # Ensure the indices are within the bounds of A
    if start_row < 0 or start_col < 0 or start_row + rows > larger.shape[0] or start_col + cols > larger.shape[1]:
        raise ValueError("Matrix B cannot be placed at the specified position in matrix A.")
    
    # Add matrix B to matrix A
    larger[start_row:start_row + rows, start_col:start_col + cols] += smaller

    return larger

def circle_like(array: np.ndarray) -> np.ndarray:
    """
    Creates a circular mask with the same shape as the input array.
    Args:
        array (ndarray): The input array.
    Returns:
        ndarray: The circular mask.
    """

    mask = np.zeros(array.shape, dtype='bool')
    center_y, center_x = np.array(mask.shape) // 2
    radius = min(center_y, center_x)

    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    mask = distance <= radius

    return mask