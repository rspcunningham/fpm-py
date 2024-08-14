"""
This module contains the `ImageCapture` and `ImageSeries` classes, which are the base datatypes used to store and manipulate images and image series in the FPM framework. The `ImageCapture` class represents a single image capture, while the  ImageSeries  class represents a series of images captured with the same optical magnification and pixel size. The `ImageSeries` class also contains methods for saving and loading image series data to and from disk.
"""

from dataclasses import dataclass, field
import numpy as np
import torch

# Set up the device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

@dataclass
class ImageCapture:
    """
    A single image capture with associated k-vector.

    Args:
        image (torch.Tensor): The image data
        k_vector (torch.Tensor): The k-vector associated with the image

    Raises:
        ValueError: If the image is not 2D or the k_vector is not 1x2

    """
    image: torch.Tensor
    k_vector: torch.Tensor

    def __post_init__(self):
        # Ensure image is 2D
        if self.image.dim() != 2:
            raise ValueError("image must be a 2D tensor (single channel image)")

        # Ensure k_vector is 1x2
        if self.k_vector.shape != (2,):
            raise ValueError("k_vector must be a tensor with shape [k_x, k_y]")
        
        # Ensure both tensors are on the same device
        device = self.image.device
        self.k_vector = self.k_vector.to(device)
        
        # Ensure correct data types
        self.image = self.image.to(torch.float32)
        self.k_vector = self.k_vector.to(torch.float32)

        
@dataclass
class ImageSeries: 
    """
    A series of images captured with the same optical magnification and pixel size.
    
    Args:
        image_stack (list[ImageCapture]): A list of ImageCapture objects
        optical_magnification (float): The optical magnification of the images
        pixel_size (float): The physical size of a pixel in the image, in micrometers
        effective_magnification (float): The effective magnification of the images. If not provided, it will be calculated as optical_magnification / pixel_size
    
    Raises:
        ValueError: If the images in the stack do not have the same shape
        ValueError: If effective_magnification is not provided and both optical_magnification and pixel_size are not provided
    """
    image_stack: list[ImageCapture]
    optical_magnification: float = None
    pixel_size: float = None
    effective_magnification: float = None
    device: torch.device = field(init=False)
    du: float = field(init=False)
    image_size: tuple[int, int] = field(init=False)
    max_k: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.effective_magnification is None:
            try: 
                self.effective_magnification = self.optical_magnification / self.pixel_size
            except:
                raise ValueError("effective_magnification or both optical_magnification and pixel_size must be provided as floats")
            
        self.device = get_device()
        
        self.du = self.effective_magnification / self.image_stack[0].image.shape[0]

        # Ensure all images have the same shape
        self.image_size = self.image_stack[0].image.shape
        for image in self.image_stack[1:]:
            if image.image.shape != self.image_size:
                raise ValueError("All images in the stack must have the same shape")
            
        self.image_size = torch.tensor(self.image_size, device=self.device)
        
        # ensure all images are on the same device
        self.device = self.image_stack[0].image.device
        for image in self.image_stack[1:]:
            image.image = image.image.to(self.device)

        # Calculate the maximum k values
        k_vectors = torch.stack([item.k_vector for item in self.image_stack])
        self.max_k = torch.max(torch.abs(k_vectors), dim=0)[0]
        
    def save(self, path: str):
        """
        Save the ImageSeries object to disk.

        Args:
            path (str): The path to save the object to

        """
        self.device = None
        torch.save(self, path)
    
    @staticmethod
    def load(path: str):
        """
        Load an ImageSeries object from disk.
        
        Args:
            path (str): The path to load the object from
        """
        dataset = torch.load(path)
        dataset.device = get_device()
        return dataset

    @staticmethod
    def from_dict(path: str):
        """
        Load an ImageSeries object saved as a dictionary.

        Args:
            path (str): The path to load the object from

        Returns:
            ImageSeries: The loaded ImageSeries object

        """
        torch.serialization.add_safe_globals(['image_stack', 'optical_magnification', 'pixel_size'])
        data = torch.load(path, weights_only=True)
        image_stack = [
            ImageCapture(item["image"].to(device=get_device()), item["k_vector"].to(device=get_device())) 
            for item in data["image_stack"]
        ]
        return ImageSeries(image_stack, data["optical_magnification"], data["pixel_size"])
    
    def to_dict(self):
        """
        Convert the ImageSeries object to a dictionary.
        """
        return {
            "image_stack": [{"image": item.image.cpu().numpy().tolist(), "k_vector": item.k_vector.cpu().numpy().tolist()} for item in self.image_stack],
            "optical_magnification": self.optical_magnification,
            "pixel_size": self.pixel_size,
            "effective_magnification": self.effective_magnification,
            "device": str(self.device),
            "du": self.du,
            "image_size": self.image_size
        }