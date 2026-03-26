import json
import numpy as np
import torch
from typing import cast
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study
from ptych.data.parse import parse_manifest
from ptych.core.pupil import make_ideal_pupil, make_zernike_pupil

study_dir = "demo/synthetic"

# Load manifest to derive optical parameters and downsample ratio
with open(f"{study_dir}/info.json") as f:
    manifest = parse_manifest(cast(dict[str, object], json.load(f)))

# Load ideal.png and convert to grayscale float [0, 1]
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

# Derive downsample ratio from object size vs manifest capture dimensions
N = object_tensor.shape[0]
capture_size = manifest.capture_dimensions.height
assert N % capture_size == 0, (
    f"Object size ({N}) must be integer multiple of capture size ({capture_size})"
)
downsample_ratio = N // capture_size

pupil_params = make_ideal_pupil(
    N=N,
    NA=0.13,
    wavelength_m=manifest.captures[0].wavelength,
    sensor_pixel_size_m=manifest.sensor_pixel_size,
    magnification=manifest.magnification,
    downsample_ratio=downsample_ratio,
)
pupil_tensor = make_zernike_pupil(
    pupil_params.phase_coeffs, pupil_params.amp_coeffs,
    pupil_params.basis, pupil_params.rad_fraction, use_softplus=False,
)

# Run synthetic study generation
generate_synthetic_study(
    dir_path=study_dir,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor
)
