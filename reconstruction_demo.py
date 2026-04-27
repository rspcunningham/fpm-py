from pathlib import Path

from preview_utils import save_metrics_summary, save_preview_png, save_tensor
from ptych import CaptureRegion, PtychStudy, solve_study
from ptych.core.pupil import Pupil, radius_fraction_from_optics
from ptych.data.types import is_illuminated_capture

dataset = "usaf-test-dark"

# Load dataset from nextcloud storage
study = PtychStudy.load(dataset)

# Reconstruction geometry settings
TILE_SIZE = 128
CROP_SIZE = 128
CAPTURE_SELECTION = None

OBJECT_TO_CAPTURE_RATIO = 4
NUMERICAL_APERTURE = (
    0.13  # Used to generate the initial pupil guess; still a free parameter.
)
NUM_PHASE_TERMS = 10
NUM_AMP_TERMS = 10

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
TILE_BATCH_SIZE = 16
EPOCHS = 50

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare the initial pupil guess.
study_wavelength = next(
    capture for capture in study.manifest.captures if is_illuminated_capture(capture)
).wavelength

pupil = Pupil(
    object_grid_size=TILE_SIZE * OBJECT_TO_CAPTURE_RATIO,
    num_phase_terms=NUM_PHASE_TERMS,
    num_amp_terms=NUM_AMP_TERMS,
    radius_fraction=radius_fraction_from_optics(
        numerical_aperture=NUMERICAL_APERTURE,
        wavelength_m=study_wavelength,
        sensor_pixel_size_m=study.manifest.sensor_pixel_size,
        magnification=study.manifest.magnification,
        object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    ),
)

capture_region = CaptureRegion.centered_square(
    width=study.captures.shape[2],
    height=study.captures.shape[1],
    size=CROP_SIZE,
)

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
)

# Save reconstruction artifacts.
save_metrics_summary(
    result.batch_metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

stitched_object = result.reconstruction
save_tensor(stitched_object, OUTPUT_DIR / "stitched_object.npy")
save_preview_png(
    stitched_object,
    OUTPUT_DIR / "stitched_object.png",
    mode="intensity",
)
save_preview_png(
    stitched_object,
    OUTPUT_DIR / "stitched_phase.png",
    mode="phase",
)
print(f"Reconstruction tensor: {result.reconstruction.shape}")
print(f"Stitched result shape: {stitched_object.shape}")
