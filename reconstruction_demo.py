from pathlib import Path

import numpy as np

from ptych import PtychStudy, solve_study
from ptych.core.metric_plots import save_metrics_summary

dataset = "usaf-test-dark"

# Reconstruction geometry settings
PATCH_SIZE = 256
CROP_SIZE = 256

OBJECT_TO_CAPTURE_RATIO = 4
NUM_PHASE_TERMS = 10
NUM_AMPLITUDE_TERMS = 10

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
BATCH_SIZE = 16
EPOCHS = 200

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load only the reconstruction crop from the cached dataset.
study = PtychStudy.load(dataset, crop_size=CROP_SIZE)

# Run reconstruction.
result = solve_study(
    study,
    patch_size=PATCH_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    pupil_num_phase_terms=NUM_PHASE_TERMS,
    pupil_num_amplitude_terms=NUM_AMPLITUDE_TERMS,
    epochs=EPOCHS,
    device=TORCH_DEVICE,
    batch_size=BATCH_SIZE,
)

# Save reconstruction artifacts.
save_metrics_summary(
    result.metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

np.save(OUTPUT_DIR / "object.npy", result.object.cpu().numpy())
np.save(OUTPUT_DIR / "capture_0.npy", result.capture_0.cpu().numpy())
print("Reconstruction complete!")
print(f"Reconstructed object tensor: {result.object.shape}")
