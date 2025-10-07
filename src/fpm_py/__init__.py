"""FPM: Fourier Ptychographic Microscopy

A Python library for Fourier ptychographic microscopy simulation and reconstruction.
"""

__version__ = "2.0.0"

from .core.forward import reconstruct
from .core.backward import kvectors_to_image_series, spatial_to_kvectors
from .analysis.metrics import plot_comparison_with_histograms
from .utils.data_utils import image_to_tensor, best_device

__all__ = [
    "reconstruct",
    "kvectors_to_image_series",
    "spatial_to_kvectors",
    "plot_comparison_with_histograms",
    "image_to_tensor",
    "best_device",
]
