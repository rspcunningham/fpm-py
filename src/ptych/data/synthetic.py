from pathlib import Path
import json
from typing import cast

import numpy as np
import torch
from jaxtyping import Complex

from ptych.core.synthetic import synthesize_captures
from ptych.data.parse import parse_manifest
from ptych.data.utils import compute_k_camera

def generate_synthetic_study(
    dir_path: str | Path,
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
) -> None:
    """
    Generate synthetic captures from info.json and save to captures/ directory.

    Loads the manifest from dir_path/info.json, computes k-vectors from LED
    positions, runs the forward model, and saves .npy files to dir_path/captures/.

    Args:
        dir_path: Directory containing info.json
        object_tensor: Complex object tensor [N, N]
        pupil_tensor: Complex pupil tensor [N, N]
        downsample_ratio: Factor to downsample by (N / n)
    """
    dir_path = Path(dir_path)

    # Load and parse manifest
    manifest_path = dir_path / "info.json"
    with open(manifest_path) as f:
        manifest = parse_manifest(cast(dict[str, object], json.load(f)))

    # Filter out darkfield captures
    valid_captures = [cap for cap in manifest.captures if cap.led_positions]

    # === Temporary assertions (matching study.py) ===
    wavelengths = [cap.wavelength for cap in valid_captures]
    assert len(set(wavelengths)) == 1, (
        f"All captures must have the same wavelength. Found: {set(wavelengths)}"
    )
    wavelength = wavelengths[0]

    for i, cap in enumerate(valid_captures):
        assert len(cap.led_positions) == 1, (
            f"Multi-LED captures are not supported yet. "
            f"Capture {i} ({cap.filename}) has {len(cap.led_positions)} LED positions."
        )

    # === End temporary assertions ===

    # Get the downsample ratio from the ideal object and target capture size
    width, height = manifest.capture_dimensions.width, manifest.capture_dimensions.height
    ratio_x = object_tensor.shape[1] / width
    ratio_y = object_tensor.shape[0] / height
    if ratio_x != ratio_y:
        raise ValueError(f"Downsample ratios in x and y do not match: {ratio_x} vs {ratio_y}. Please ensure the desired capture size is the same aspect ratio as the object tensor.")
    if ratio_x != int(ratio_x):
        raise ValueError(f"Downsample ratio is not an integer: {ratio_x}. Please ensure the desired capture size is an integer fraction of the object tensor size.")

    # Compute k-vectors from LED positions
    k_vectors = [
        compute_k_camera(
            cap.led_positions[0],
            wavelength,
            manifest.sensor_pixel_size,
            manifest.magnification,
        )
        for cap in valid_captures
    ]
    kx_batch = torch.tensor([kx for kx, _ in k_vectors])
    ky_batch = torch.tensor([ky for _, ky in k_vectors])

    # Generate synthetic captures
    captures = synthesize_captures(
        object_tensor, pupil_tensor, kx_batch, ky_batch, int(ratio_x)
    )

    # Save to captures/ directory
    captures_dir = dir_path / "captures"
    captures_dir.mkdir(exist_ok=True)

    for i, cap in enumerate(valid_captures):
        img = captures[i].detach().cpu().numpy()
        np.save(captures_dir / cap.filename, img)

    print(f"Generated {len(valid_captures)} synthetic captures in {captures_dir}")
