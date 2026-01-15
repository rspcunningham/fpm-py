import torch
import numpy as np
from PIL import Image

from ptych import solve_inverse, PtychStudy

BASE_DIR = "./demo"

study = PtychStudy.from_disk(BASE_DIR)

# load ground truth image to get necessary dims
gold_img = Image.open(f"{BASE_DIR}/gold.png").convert("L")
gold_amplitude = np.array(gold_img, dtype=np.float32) / 255.0

# create initial object, amplitude == 0.5, phase == 0
obj_np = np.ones_like(gold_amplitude) / 2
object = torch.from_numpy(obj_np).to(torch.complex64)

# create pupil tensor identical to object
pupil = object.clone()

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
