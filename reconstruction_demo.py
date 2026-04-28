import json
from pathlib import Path

import numpy as np

from ptych import PtychStudy, solve_study
from ptych.core.metric_plots import save_metrics_summary

dataset = "usaf-test-dark"
# dataset = "malaria-test"

# Reconstruction geometry settings
PATCH_SIZE = 300
CROP_SIZE = 300

OBJECT_TO_CAPTURE_RATIO = 1
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
PATCH_BATCH_SIZE = 16
EPOCHS = 10

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}-small-gains")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load only the reconstruction crop from the cached dataset.
study = PtychStudy.load(dataset, crop_size=CROP_SIZE)

# Run reconstruction.
result = solve_study(
    study,
    patch_size=PATCH_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    pupil_phase_radial_order=PUPIL_PHASE_RADIAL_ORDER,
    pupil_amplitude_radial_order=PUPIL_AMPLITUDE_RADIAL_ORDER,
    epochs=EPOCHS,
    device=TORCH_DEVICE,
    patch_batch_size=PATCH_BATCH_SIZE,
)

# Save reconstruction artifacts.
save_metrics_summary(
    result.metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

np.save(OUTPUT_DIR / "object.npy", result.object.cpu().numpy())
np.save(OUTPUT_DIR / "capture_0.npy", result.capture_0.cpu().numpy())
with (OUTPUT_DIR / "metrics.json").open("w") as file:
    json.dump(result.metrics, file)
print("Reconstruction complete!")
print(f"Reconstructed object tensor: {result.object.shape}")
print(f"Results written to {OUTPUT_DIR}")
