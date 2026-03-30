from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from ptych import PtychStudy
from ptych.data.synthetic import generate_synthetic_study
from ptych.core.pupil import make_ideal_pupil, make_zernike_pupil

IDEAL_IMAGE_PATH = Path("demo_images/ideal.png")
OUTPUT_DIR = Path("results/synthetic_usaf_test")

# Load a real study to use its info.json for generating synthetic data
study = PtychStudy.load("usaf-test")

# Load ideal.png and convert to grayscale float [0, 1]
img = Image.open(IDEAL_IMAGE_PATH).convert("L")
amplitude: npt.NDArray[np.float32] = np.asarray(img, dtype=np.float32) / np.float32(255.0)

# Center-crop to a square since the current synthetic pipeline requires NxN square tensors.
image_shape = cast(tuple[int, int], amplitude.shape)
height = image_shape[0]
width = image_shape[1]
crop_size = min(height, width)
top = (height - crop_size) // 2
left = (width - crop_size) // 2
amplitude = amplitude[top:top + crop_size, left:left + crop_size]

synthetic_manifest = replace(
    study.manifest,
    capture_dimensions=replace(
        study.manifest.capture_dimensions,
        width=crop_size,
        height=crop_size,
    ),
)

# Create object tensor: phase proportional to amplitude
# Scale phase to [0, 2*pi] range
phase = amplitude * 2 * np.pi
object_tensor = torch.from_numpy(amplitude * np.exp(1j * phase)).to(torch.complex64)

# Derive the object-to-capture ratio from object size vs manifest capture dimensions
N = object_tensor.shape[0]
capture_size = synthetic_manifest.capture_dimensions.height
assert N % capture_size == 0, (
    f"Object size ({N}) must be integer multiple of capture size ({capture_size})"
)
object_to_capture_ratio = N // capture_size
print(f"Using object-to-capture ratio: {object_to_capture_ratio}")

pupil_params = make_ideal_pupil(
    object_grid_size=N,
    numerical_aperture=0.13,
    wavelength_m=synthetic_manifest.captures[0].wavelength,
    sensor_pixel_size_m=synthetic_manifest.sensor_pixel_size,
    magnification=synthetic_manifest.magnification,
    object_to_capture_ratio=object_to_capture_ratio,
)
pupil_tensor = make_zernike_pupil(
    pupil_params.phase_coeffs, pupil_params.amp_coeffs,
    pupil_params.basis, pupil_params.rad_fraction, use_softplus=False,
)

# Run synthetic study generation
generate_synthetic_study(
    manifest=synthetic_manifest,
    output_dir=OUTPUT_DIR,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor,
)
