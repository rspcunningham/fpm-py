import numpy as np
import torch
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study
from ptych.core.zernike import precompute_zernike_basis, make_zernike_pupil

study_dir = "tmp/usaf_binary"

# Load gold.png and convert to grayscale float [0, 1]
img = Image.open(f"{study_dir}/ideal.png").convert("L")
amplitude = np.array(img, dtype=np.float32) / 255.0

# Center-crop to a square so the current synthetic pipeline receives NxN tensors.
height, width = amplitude.shape
crop_size = min(height, width)
top = (height - crop_size) // 2
left = (width - crop_size) // 2
amplitude = amplitude[top:top + crop_size, left:left + crop_size]

# Create object tensor: phase proportional to amplitude
# Scale phase to [0, 2*pi] range
phase = amplitude * 2 * np.pi
object_tensor = torch.from_numpy(amplitude * np.exp(1j * phase)).to(torch.complex64)

# Create pupil tensor using Zernike basis
N = object_tensor.shape[0]

# Precompute Zernike basis
basis = precompute_zernike_basis(N)

# Define Zernike coefficients
phase_coeffs = torch.zeros(basis.num_phase_terms)  # No aberrations
amp_coeffs = torch.zeros(basis.num_amp_terms)
amp_coeffs[0] = 1.0  # Piston = uniform amplitude
rad_fraction = 0.15  # rad_fraction=0.15 matches current 0.30/2 radius

# Generate pupil (use_softplus=False for exact amplitude)
pupil_tensor = make_zernike_pupil(phase_coeffs, amp_coeffs, basis, rad_fraction, use_softplus=False)

# Set downsample factor
downsample_ratio = 4

# Run synthetic study generation
generate_synthetic_study(
    dir_path=study_dir,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor
)
