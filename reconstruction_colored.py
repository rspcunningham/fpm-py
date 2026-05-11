import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ptych import ImageCrop, PtychStudy, SolverLearningRates, solve_study
from ptych.core.metric_plots import save_metrics_summary
from ptych.data.types import VALID_COLOR_CHANNELS
from ptych.data.utils import get_default_device

# Select dataset
dataset = "20250821-160234-USAF-Colour"

CROP_SIZE = 512
CROP_TOP = 1024
CROP_LEFT = 1104

# Reconstruction model settings
OBJECT_TO_CAPTURE_RATIO = 4
PUPIL_PHASE_RADIAL_ORDER = 3
PUPIL_AMPLITUDE_RADIAL_ORDER = 0

# Memory/scaling settings
PATCH_SIZE = 256
PATCH_BATCH_SIZE = 1
ILLUMINATION_CHUNK_SIZE = 32

# Runtime settings
EPOCHS = 8
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


studies = PtychStudy.load_by_channel(
    dataset,
    crop=ImageCrop(top=CROP_TOP, left=CROP_LEFT, width=CROP_SIZE, height=CROP_SIZE),
)
missing_channels = [
    channel for channel in VALID_COLOR_CHANNELS if channel not in studies
]
if missing_channels:
    raise ValueError(f"Dataset is missing channels: {missing_channels}")

# Get GPU
TORCH_DEVICE = get_default_device()
print(f"Using torch device: {TORCH_DEVICE}")

rgb_channels: list[np.ndarray] = []
metrics_by_channel: dict[str, object] = {}
wavelengths_by_channel: dict[str, float] = {}
for channel in VALID_COLOR_CHANNELS:
    study = studies[channel]
    wavelength = study.capture_metadata[0].wavelength
    wavelengths_by_channel[channel] = wavelength
    channel_dir = OUTPUT_DIR / channel
    channel_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solving {channel} channel ({wavelength:g} m)")
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

    metrics_by_channel[channel] = result.metrics
    save_metrics_summary(
        result.metrics,
        path=channel_dir / "reconstruction_metrics.png",
    )
    np.save(channel_dir / "object_complex.npy", result.object.cpu().numpy())
    np.save(channel_dir / "capture_0.npy", result.capture_0.cpu().numpy())

    intensity = result.object.abs().square()
    intensity = intensity / intensity.max().clamp_min(1e-12)
    rgb_channels.append(intensity.cpu().numpy().astype(np.float32))

object_path = OUTPUT_DIR / "object.npy"
metrics_json_path = OUTPUT_DIR / "metrics.json"
wavelengths_json_path = OUTPUT_DIR / "wavelengths.json"

rgb_object = np.stack(rgb_channels, axis=-1)
np.save(object_path, rgb_object)
with metrics_json_path.open("w") as file:
    json.dump(metrics_by_channel, file)
with wavelengths_json_path.open("w") as file:
    json.dump(wavelengths_by_channel, file, indent=2)

print("Color reconstruction complete!")
print(f"RGB object tensor: {rgb_object.shape}")
print(f"Saved RGB object: {object_path}")
print(f"Saved metrics JSON: {metrics_json_path}")
print(f"Saved wavelengths JSON: {wavelengths_json_path}")
