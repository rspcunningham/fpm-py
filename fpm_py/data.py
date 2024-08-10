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

class ImageCapture:
    image: torch.Tensor
    k_vector: torch.Tensor

    def __post_init__(self):
        # Ensure image is 2D
        if self.image.dim() != 2:
            raise ValueError("image must be a 2D tensor (single channel image)")

        # Ensure k_vector is 1x2
        if self.k_vector.shape != (1, 2):
            raise ValueError("k_vector must be a tensor with shape (1, 2)")
        
        # Ensure both tensors are on the same device
        device = self.image.device
        self.k_vector = self.k_vector.to(device)
        
        # Ensure correct data types
        self.image = self.image.to(torch.float32)
        self.k_vector = self.k_vector.to(torch.float32)

        
@dataclass
class ImageSeries: 
    image_stack: list[ImageCapture]
    optical_magnification: float = None
    pixel_size: float = None
    effective_magnification: float = None
    device: torch.device = field(init=False)
    du: float = field(init=False)
    image_size: tuple[int, int] = field(init=False)

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
        
        # ensure all images are on the same device
        self.device = self.image_stack[0].image.device
        for image in self.image_stack[1:]:
            image.image = image.image.to(self.device)
        
        

        