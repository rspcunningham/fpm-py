from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from ptych import PtychStudy
from ptych.data.synthetic import generate_synthetic_study
from ptych.core.pupil import Pupil, pupil_cutoff_cyc_per_px_from_optics
from ptych.data.types import is_illuminated_capture

IDEAL_IMAGE_PATH = Path("demo_images/ideal.png")
OUTPUT_DIR = Path("results/synthetic_usaf_test")

# Load a real study to use its info.json for generating synthetic data
study = PtychStudy.load("usaf-test")

# Load ideal.png and convert to grayscale float [0, 1]
img = Image.open(IDEAL_IMAGE_PATH).convert("L")
amplitude: npt.NDArray[np.float32] = np.asarray(img, dtype=np.float32) / np.float32(
    255.0
)

# Center-crop to a square since the current synthetic pipeline requires NxN square tensors.
image_shape = cast(tuple[int, int], amplitude.shape)
height = image_shape[0]
width = image_shape[1]
crop_size = min(height, width)
top = (height - crop_size) // 2
left = (width - crop_size) // 2
amplitude = amplitude[top : top + crop_size, left : left + crop_size]

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

pupil_capture = next(
    capture
    for capture in synthetic_manifest.captures
    if is_illuminated_capture(capture)
)
pupil_params = Pupil(
    object_grid_size=N,
    pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_from_optics(
        numerical_aperture=synthetic_manifest.numerical_aperture,
        wavelength_m=pupil_capture.wavelength,
        sensor_pixel_size_m=synthetic_manifest.sensor_pixel_size,
        magnification=synthetic_manifest.magnification,
        object_to_capture_ratio=object_to_capture_ratio,
    ),
)
with torch.no_grad():
    pupil_tensor = pupil_params()[0]

# Run synthetic study generation
captures = generate_synthetic_study(
    manifest=synthetic_manifest,
    output_dir=OUTPUT_DIR,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor,
)

print(f"Generated captures tensor shape: {tuple(captures.shape)}")
