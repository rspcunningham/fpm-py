import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ptych import ImageCrop, PtychStudy, SolverLearningRates, solve_study
from ptych.core.metric_plots import save_metrics_summary
from ptych.data.utils import get_default_device

# Select dataset
dataset = "usaf-test-dark"

CROP_SIZE = 416  # should be the same as PATCH_SIZE if practical
CROP_TOP = 1024
CROP_LEFT = 1440

# Reconstruction model settings
OBJECT_TO_CAPTURE_RATIO = 4  # prefer 2 or 4
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Memory/scaling settings
PATCH_SIZE = 416  # prefer power of 2 or 384, 416, 448, 480, 512
PATCH_BATCH_SIZE = 16
ILLUMINATION_CHUNK_SIZE = 145

# Runtime settings
EPOCHS = 60
LEARNING_RATES = SolverLearningRates(
    object=1e-1,
    pupil=1e-2,
    illumination_gains=1e-1,
    darkfield_backgrounds=1e-2,
    darkfield_scatter=3e-2,
)

# Output directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = Path(f"results/{dataset}-{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

study = PtychStudy.load(
    dataset,
    crop=ImageCrop(top=CROP_TOP, left=CROP_LEFT, width=CROP_SIZE, height=CROP_SIZE),
)

# Get GPU
TORCH_DEVICE = get_default_device()
print(f"Using torch device: {TORCH_DEVICE}")

# Run reconstruction
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
    learning_rates=LEARNING_RATES,
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
