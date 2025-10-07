from ..core.structs import ImageSeries
import torch
from PIL import Image
import numpy as np

"""
Tensor device helpers
"""

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def auto_place(series: ImageSeries) -> ImageSeries:
    return series.to(best_device())

"""
Image Loading Helper
"""

def image_to_tensor(
    image_path: str,
    *,
    to_complex: bool = False,
    device: torch.device | str | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load an image file and convert it to a torch.Tensor.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
    to_complex : bool, optional
        If True, converts the image to a complex tensor with zero imaginary part.
        Default is False.
    device : torch.device or str, optional
        Device to place the tensor on. If None, uses CUDA if available, 
        then MPS (Apple Silicon), then CPU.
    normalize : bool, optional
        If True, normalizes pixel values to range [0, 1]. Default is True.
        
    Returns
    -------
    torch.Tensor
        The image as a torch.Tensor.
    """
    
    
    # Determine device if not specified
    if device is None:
        device = best_device()
    
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")
    
    # Convert to numpy array and then to tensor
    img_np = np.array(img, copy=True)
    
    # Normalize if required
    if normalize:
        img_tensor = torch.from_numpy(img_np).float().to(device) / 255.0
    else:
        img_tensor = torch.from_numpy(img_np).float().to(device)
    
    # Convert to complex tensor if requested
    if to_complex:
        img_tensor = img_tensor.to(torch.complex64)  # Zero imaginary part
    
    return img_tensor
