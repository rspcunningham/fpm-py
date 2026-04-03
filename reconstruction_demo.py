from pathlib import Path

import torch

from ptych import CaptureRegion, PtychStudy, solve_study
from ptych.core.pupil import make_ideal_pupil
from preview_utils import save_preview_png, save_tensor, save_metrics_summary

from viewer.qtlib import show_grayscale_subplots

dataset = "malaria-test"

# Load dataset from nextcloud storage
study = PtychStudy.load(dataset)

# Reconstruction geometry settings
TILE_SIZE = 256
CROP_SIZE = 256
#CAPTURE_SELECTION = [(0, 37)]  # all captures
CAPTURE_SELECTION = None

OBJECT_TO_CAPTURE_RATIO = 4
NUMERICAL_APERTURE = 0.13  # Used to generate the initial pupil guess; still a free parameter.
NUM_PHASE_TERMS = 10
NUM_AMP_TERMS = 10

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
TILE_BATCH_SIZE = 16
EPOCHS = 150

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}-2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare the initial pupil guess.
pupil = make_ideal_pupil(
    object_grid_size=TILE_SIZE * OBJECT_TO_CAPTURE_RATIO,
    numerical_aperture=NUMERICAL_APERTURE,
    wavelength_m=study.manifest.captures[0].wavelength,
    sensor_pixel_size_m=study.manifest.sensor_pixel_size,
    magnification=study.manifest.magnification,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    num_phase_terms=NUM_PHASE_TERMS,
    num_amp_terms=NUM_AMP_TERMS,
)

capture_region = CaptureRegion.centered_square(
    width=study.captures.shape[2],
    height=study.captures.shape[1],
    size=CROP_SIZE,
)


def save_checkpoint(epoch: int, merged_object: torch.Tensor) -> None:
    save_tensor(merged_object, OUTPUT_DIR / f"checkpoint_epoch_{epoch:04d}.npy")

show_grayscale_subplots(study.captures)

# Run reconstruction.
result = solve_study(
    study,
    pupil,
    capture_selector=CAPTURE_SELECTION,
    capture_region=capture_region,
    tile_size=TILE_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    epochs=EPOCHS,
    torch_device=TORCH_DEVICE,
    tile_batch_size=TILE_BATCH_SIZE,
    on_checkpoint=save_checkpoint,
    checkpoint_interval=50,
)

# Save reconstruction artifacts.
save_metrics_summary(
    result.batch_metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

save_tensor(result.stitched_object, OUTPUT_DIR / "stitched_object.npy")
save_preview_png(
    result.stitched_object,
    OUTPUT_DIR / "stitched_object.png",
    mode="intensity",
)
save_preview_png(
    result.stitched_object,
    OUTPUT_DIR / "stitched_phase.png",
    mode="phase",
)
print(f"Stitched result shape: {result.stitched_object.shape}")
