import json
from pathlib import Path

import numpy as np

from ptych import PtychStudy, solve_study
from ptych.core.metric_plots import save_metrics_summary

dataset = "usaf-test-dark"

# Reconstruction geometry settings
PATCH_SIZE = 400
CROP_SIZE = 400

OBJECT_TO_CAPTURE_RATIO = 2
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
PATCH_BATCH_SIZE = 16
EPOCHS = 100

# MTF-preserving display/output sharpening after the artifact-suppressed solve.
EDGE_SHARPEN_THRESHOLD_PERCENTILE = 90
EDGE_SHARPEN_SIGMA_PX = 3.0
EDGE_SHARPEN_AMOUNT = 2.25
EDGE_MASK_SIGMA_PX = 1.0
EDGE_MASK_SLOPE = 8.0

# Output directory
OUTPUT_DIR = Path(f"results/{dataset}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _fft_gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    height, width = image.shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    y_frequency, x_frequency = np.meshgrid(fy, fx, indexing="ij")
    gaussian = np.exp(
        -2.0
        * np.pi**2
        * sigma_px**2
        * (np.square(x_frequency) + np.square(y_frequency))
    )
    return np.fft.ifft2(np.fft.fft2(image) * gaussian).real


def _edge_sharpen_object(object_array: np.ndarray) -> np.ndarray:
    object_intensity = np.square(np.abs(object_array))
    low_clip, high_clip = np.percentile(object_intensity, [0.1, 99.9])

    edge_base = _fft_gaussian_blur(object_intensity, EDGE_MASK_SIGMA_PX)
    y_gradient, x_gradient = np.gradient(edge_base)
    edge_strength = np.sqrt(np.square(x_gradient) + np.square(y_gradient))
    edge_threshold = np.percentile(
        edge_strength,
        EDGE_SHARPEN_THRESHOLD_PERCENTILE,
    )
    edge_width = np.percentile(edge_strength, 99.5) - edge_threshold
    edge_mask = 1.0 / (
        1.0
        + np.exp(
            -EDGE_MASK_SLOPE * (edge_strength - edge_threshold) / (edge_width + 1e-12)
        )
    )
    edge_mask = _fft_gaussian_blur(edge_mask, EDGE_MASK_SIGMA_PX)

    blurred_intensity = _fft_gaussian_blur(object_intensity, EDGE_SHARPEN_SIGMA_PX)
    sharpened_intensity = object_intensity + EDGE_SHARPEN_AMOUNT * edge_mask * (
        object_intensity - blurred_intensity
    )
    sharpened_intensity = np.clip(sharpened_intensity, low_clip, high_clip)
    amplitude_scale = np.sqrt(
        np.maximum(sharpened_intensity, 0.0) / (object_intensity + 1e-12)
    )
    return (object_array * amplitude_scale).astype(object_array.dtype, copy=False)


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
object_array = result.object.cpu().numpy()
object_array = _edge_sharpen_object(object_array)
save_metrics_summary(
    result.metrics,
    path=OUTPUT_DIR / "reconstruction_metrics.png",
)

np.save(OUTPUT_DIR / "object.npy", object_array)
np.save(OUTPUT_DIR / "capture_0.npy", result.capture_0.cpu().numpy())
with (OUTPUT_DIR / "metrics.json").open("w") as file:
    json.dump(result.metrics, file)
print("Reconstruction complete!")
print(f"Reconstructed object tensor: {result.object.shape}")
print(f"Results written to {OUTPUT_DIR}")
