from pathlib import Path

from preview_utils import save_metrics_summary, save_tensor
from ptych import CaptureRegion, PtychStudy, solve_study
from ptych.core.pupil import radius_fraction_from_optics
from ptych.data.types import is_illuminated_capture

dataset = "usaf-test-dark"

# Load dataset from nextcloud storage
study = PtychStudy.load(dataset)

# Reconstruction geometry settings
PATCH_SIZE = 256
CROP_SIZE = 256

OBJECT_TO_CAPTURE_RATIO = 4
NUMERICAL_APERTURE = (
    0.13  # Used to generate the initial pupil guess; still a free parameter.
)
NUM_PHASE_TERMS = 10
NUM_AMP_TERMS = 10

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
BATCH_SIZE = 16
EPOCHS = 200

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepare the initial pupil guess.
study_wavelength = next(
    capture for capture in study.manifest.captures if is_illuminated_capture(capture)
).wavelength

pupil_radius_fraction = radius_fraction_from_optics(
    numerical_aperture=NUMERICAL_APERTURE,
    wavelength_m=study_wavelength,
    sensor_pixel_size_m=study.manifest.sensor_pixel_size,
    magnification=study.manifest.magnification,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
)

capture_region = CaptureRegion.centered_square(
    width=study.captures.shape[2],
    height=study.captures.shape[1],
    size=CROP_SIZE,
)

# Run reconstruction.
result = solve_study(
    study,
    capture_region=capture_region,
    patch_size=PATCH_SIZE,
    object_to_capture_ratio=OBJECT_TO_CAPTURE_RATIO,
    pupil_radius_fraction=pupil_radius_fraction,
    pupil_num_phase_terms=NUM_PHASE_TERMS,
    pupil_num_amp_terms=NUM_AMP_TERMS,
    epochs=EPOCHS,
    device=TORCH_DEVICE,
    batch_size=BATCH_SIZE,
)

# Save reconstruction artifacts.
save_metrics_summary(
    result.metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

reconstruction = result.reconstruction
save_tensor(reconstruction, OUTPUT_DIR / "reconstruction.npy")
save_tensor(result.raw_object_amplitude, OUTPUT_DIR / "raw_object_amplitude.npy")
save_tensor(result.raw_object_phase, OUTPUT_DIR / "raw_object_phase.npy")

print(f"Reconstruction tensor: {result.reconstruction.shape}")
print(f"Raw object amplitude tensor: {result.raw_object_amplitude.shape}")
print(f"Raw object phase tensor: {result.raw_object_phase.shape}")
print(f"Stitched result shape: {reconstruction.shape}")
