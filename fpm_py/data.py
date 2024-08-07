from dataclasses import dataclass, field
import numpy as np

@dataclass
class ImageCapture:
    image: np.ndarray
    k_vector: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))

    def __post_init__(self):
        if self.k_vector.shape != (1, 2):
            raise ValueError("k_vector must be an array with shape (1, 2)")