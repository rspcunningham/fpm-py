import torch
import numpy as np
from PIL import Image

from ptych import solve_inverse, PtychStudy

BASE_DIR = "./demo"

study = PtychStudy.from_disk(BASE_DIR)

# Initialize object and pupil (512x512, amplitude 0.5, phase 0)
object = torch.full((512, 512), 0.5, dtype=torch.complex64)
pupil = torch.full((512, 512), 0.5, dtype=torch.complex64)

object, pupil, metrics = solve_inverse(
    study.captures,
    object,
    pupil,
    study.kx_batch,
    study.ky_batch,
    torch_device="mps",
)

# Save object result as PNG
object_amplitude: np.ndarray[tuple[int, int], np.dtype[np.float32]] = object.abs().cpu().numpy()
# Normalize to 0-255 range
object_amplitude_u8 = np.asarray(
    object_amplitude / object_amplitude.max() * 255, dtype=np.uint8
)
Image.fromarray(object_amplitude_u8).save(f"{BASE_DIR}/object_result.png")
