"""Demo script for generating synthetic FPM captures from bars.png."""

import numpy as np
import torch
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study

BASE_DIR = "demo"

def create_circular_mask(size: int, radius_fraction: float) -> np.ndarray:
    """Create a circular mask centered in the array."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size * radius_fraction / 2
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius**2
    return mask

# Load bars.png and convert to grayscale float [0, 1]
img = Image.open(f"{BASE_DIR}/gold.png").convert("L")
amplitude = np.array(img, dtype=np.float32) / 255.0

# Create object tensor: phase proportional to amplitude
# Scale phase to [0, 2*pi] range
phase = amplitude * 2 * np.pi
object_tensor = torch.from_numpy(amplitude * np.exp(1j * phase)).to(torch.complex64)

# Create pupil tensor
N = object_tensor.shape[0]

# Base: 0.5 amplitude, 0 phase everywhere
pupil_amplitude = np.full((N, N), 0.5, dtype=np.float32)
pupil_phase = np.zeros((N, N), dtype=np.float32)

# Central circle: 10% of tensor width, full amplitude (1.0)
circle_mask = create_circular_mask(N, 0.10)
pupil_amplitude[circle_mask] = 1.0

pupil_tensor = torch.from_numpy(pupil_amplitude * np.exp(1j * pupil_phase)).to(
    torch.complex64
)

# Set downsample factor
downsample_ratio = 4

# Run synthetic study generation
generate_synthetic_study(
    dir_path=BASE_DIR,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor,
    downsample_ratio=downsample_ratio,
)
