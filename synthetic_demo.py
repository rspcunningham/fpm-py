import numpy as np
import torch
from PIL import Image

from ptych.data.synthetic import generate_synthetic_study
from ptych.core.pupil import make_ideal_pupil

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

downsample_ratio = 4

# create ideal pupil
N = object_tensor.shape[0]

pupil_tensor = make_ideal_pupil(
    N=N,
    NA=0.30,
    wavelength_m=0.30,
    sensor_pixel_size_m=1e-6,
    magnification=1.0,
    downsample_ratio=downsample_ratio,
)

# Run synthetic study generation
generate_synthetic_study(
    dir_path=study_dir,
    object_tensor=object_tensor,
    pupil_tensor=pupil_tensor
)
