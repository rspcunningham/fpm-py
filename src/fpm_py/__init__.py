"""FPM: Fourier Ptychographic Microscopy

A Python library for Fourier ptychographic microscopy simulation and reconstruction.
"""

__version__ = "2.0.0"

from .core.forward import reconstruct
from .core.backward import kvectors_to_image_series, spatial_to_kvectors
from .utils.data_utils import image_to_tensor

__all__ = [
    "reconstruct",
    "kvectors_to_image_series",
    "spatial_to_kvectors",
    "image_to_tensor",
]
