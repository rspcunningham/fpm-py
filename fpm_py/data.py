"""
This module contains the `ImageCapture` and `ImageSeries` classes, which are the base datatypes used to store and manipulate images and image series in the FPM framework. The `ImageCapture` class represents a single image capture, while the  ImageSeries  class represents a series of images captured with the same optical magnification and pixel size. The `ImageSeries` class also contains methods for saving and loading image series data to and from disk.
"""

from dataclasses import dataclass, field
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
        device = get_device()
        self.image = self.image.to(device)
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
        sensor_pixel_size (float): The physical size of a pixel in the image, in micrometers
        object_pixel_size (float): The effective magnification of the images. If not provided, it will be calculated as optical_magnification / sensor_pixel_size
    
    Raises:
        ValueError: If the images in the stack do not have the same shape
        ValueError: If object_pixel_size is not provided and both optical_magnification and sensor_pixel_size are not provided
    """
    image_stack: list[ImageCapture]
    wavelength: float = None
    numerical_aperture: float = None
    optical_magnification: float = None
    sensor_pixel_size: float = None
    object_pixel_size: float = None
    device: torch.device = field(init=False)
    du: float = field(init=False)
    image_size: tuple[int, int] = field(init=False)
    max_k: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.object_pixel_size is None:
            try: 
                self.object_pixel_size = self.optical_magnification / self.sensor_pixel_size
            except:
                raise ValueError("object_pixel_size or both optical_magnification and sensor_pixel_size must be provided as floats")
            
        self.device = get_device()
        
        self.du = self.object_pixel_size / self.image_stack[0].image.shape[0]

        # Ensure all images have the same shape
        self.image_size = self.image_stack[0].image.shape
        for image in self.image_stack[1:]:
            if image.image.shape != self.image_size:
                raise ValueError("All images in the stack must have the same shape")
            
        self.image_size = torch.tensor(self.image_size, device=self.device)
        
        # Calculate the maximum k values
        k_vectors = torch.stack([item.k_vector for item in self.image_stack])
        self.max_k = torch.max(torch.abs(k_vectors), dim=0)[0]

    @staticmethod
    def from_dict(data: dict):
        """
        Load an ImageSeries object saved as a dictionary.

        Args:
            path (str): The path to load the object from

        Returns:
            ImageSeries: The loaded ImageSeries object

        """
        image_stack = [
            ImageCapture(item["image"].to(device=get_device()), item["k_vector"].to(device=get_device())) 
            for item in data["image_stack"]
        ]
        return ImageSeries(image_stack, data["optical_magnification"], data["sensor_pixel_size"])
    
    @staticmethod
    def load(path: str):
        
        torch.serialization.add_safe_globals(['image_stack', 'optical_magnification', 'sensor_pixel_size'])
        data = torch.load(path, weights_only=True)

        return ImageSeries.from_dict(data)
    
    def to_dict(self):
        """
        Convert the ImageSeries object to a dictionary.
        """
        return {
            "image_stack": [{"image": item.image.cpu().numpy().tolist(), "k_vector": item.k_vector.cpu().numpy().tolist()} for item in self.image_stack],
            "optical_magnification": self.optical_magnification,
            "sensor_pixel_size": self.sensor_pixel_size,
            "object_pixel_size": self.object_pixel_size,
            "device": str(self.device),
            "du": self.du,
            "image_size": self.image_size
        }
    
    def save(self, path):

        data = self.to_dict()
        torch.save(data, path)
