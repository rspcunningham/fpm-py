"""Demo script for generating synthetic FPM captures from bars.png."""

import numpy as np
import torch
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study
from ptych.core.zernike import precompute_zernike_basis, make_zernike_pupil

# Load gold.png and convert to grayscale float [0, 1]
img = Image.open(f"demo/gold.png").convert("L")
amplitude = np.array(img, dtype=np.float32) / 255.0

# Create object tensor: phase proportional to amplitude
# Scale phase to [0, 2*pi] range
phase = amplitude * 2 * np.pi
object_tensor = torch.from_numpy(amplitude * np.exp(1j * phase)).to(torch.complex64)

# Create pupil tensor using Zernike basis
N = object_tensor.shape[0]

# Precompute Zernike basis (rad_fraction=0.15 matches current 0.30/2 radius)
basis = precompute_zernike_basis(N, rad_fraction=0.15)

# Define Zernike coefficients
phase_coeffs = torch.zeros(basis.num_phase_terms)  # No aberrations
amp_coeffs = torch.zeros(basis.num_amp_terms)
amp_coeffs[0] = 1.0  # Piston = uniform amplitude

# Generate pupil (use_softplus=False for exact amplitude)
pupil_tensor = make_zernike_pupil(phase_coeffs, amp_coeffs, basis, use_softplus=False)

object_amplitude_u8 = np.asarray(
    pupil_tensor.real / pupil_tensor.real.max() * 255, dtype=np.uint8
)
Image.fromarray(object_amplitude_u8).save(f"tmp/test/object_result.png")

# Set downsample factor
downsample_ratio = 4

# Run synthetic study generation
generate_synthetic_study(
    dir_path="tmp/test",
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor,
    downsample_ratio=downsample_ratio,
)
