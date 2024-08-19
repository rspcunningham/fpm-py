"""
Fourier ptychography package for Python.

This package provides tools for performing Fourier ptychography microscopy (FPM) reconstruction in Python.

The package is organized into the following modules:
- `algorithm`: Contains the main reconstruction algorithm.
- `data`: Contains the `ImageCapture` and `ImageSeries` classes for storing image data.
- `iteration_terminators`: Contains functions that determine when to stop the iteration process.
- `optimizers`: Contains the optimizers used in the FPM reconstruction process.
- `utils`: Contains utility functions used throughout the package.

"""

from .algorithm import reconstruct
from .data import ImageCapture, ImageSeries
