import json
from pathlib import Path

import numpy as np

from ptych import PtychStudy, solve_study
from ptych.core.metric_plots import save_metrics_summary
from ptych.data.utils import get_default_device

dataset = "usaf-test-dark"

# Problem definition
# MPS FFT throughput is shape-sensitive. In local profiling, crop sizes 384, 416,
# 448, 480, and 512 were efficient; 400 and 432 were comparatively slow.
CROP_SIZE = 416

# Reconstruction model settings
OBJECT_TO_CAPTURE_RATIO = 2
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Memory/scaling settings
PATCH_SIZE = 416
PATCH_BATCH_SIZE = 16
ILLUMINATION_CHUNK_SIZE = 145

# Optimization and runtime settings
TORCH_DEVICE = get_default_device()
EPOCHS = 500

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load only the reconstruction crop from the cached dataset.
study = PtychStudy.load(dataset, crop_size=CROP_SIZE)

# Run reconstruction.
print(f"Using torch device: {TORCH_DEVICE}")
result = solve_study(
    study,
    patch_size=PATCH_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    pupil_phase_radial_order=PUPIL_PHASE_RADIAL_ORDER,
    pupil_amplitude_radial_order=PUPIL_AMPLITUDE_RADIAL_ORDER,
    epochs=EPOCHS,
    device=TORCH_DEVICE,
    patch_batch_size=PATCH_BATCH_SIZE,
    illumination_chunk_size=ILLUMINATION_CHUNK_SIZE,
)

# Save reconstruction artifacts.
metrics_plot_path = OUTPUT_DIR / "reconstruction_metrics.png"
object_path = OUTPUT_DIR / "object.npy"
capture_path = OUTPUT_DIR / "capture_0.npy"
metrics_json_path = OUTPUT_DIR / "metrics.json"

save_metrics_summary(
    result.metrics,
    path=metrics_plot_path,
)

np.save(object_path, result.object.cpu().numpy())
np.save(capture_path, result.capture_0.cpu().numpy())
with metrics_json_path.open("w") as file:
    json.dump(result.metrics, file)
print("Reconstruction complete!")
print(f"Reconstructed object tensor: {result.object.shape}")
print(f"Saved object: {object_path}")
print(f"Saved capture: {capture_path}")
print(f"Saved metrics JSON: {metrics_json_path}")
print(f"Saved metrics plot: {metrics_plot_path}")
