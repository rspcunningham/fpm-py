"""Generate synthetic FPM captures from ground truth object and pupil."""
from pathlib import Path
import json
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ptych.core.forward import forward_model
from ptych.data.parse import parse_manifest
from ptych.utils import compute_k_camera


def synthesize_captures(
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    downsample_ratio: int,
) -> Float[torch.Tensor, "B n n"]:
    """
    Generate synthetic captures by running forward model and downsampling.

    Args:
        object_tensor: Complex object tensor [N, N]
        pupil_tensor: Complex pupil tensor [N, N]
        kx_batch: Normalized k-vectors in x direction [B]
        ky_batch: Normalized k-vectors in y direction [B]
        downsample_ratio: Factor to downsample by (N / n)

    Returns:
        Synthetic captures [B, n, n] as float intensities
    """
    # Run forward model at full resolution
    intensities = forward_model(object_tensor, pupil_tensor, kx_batch, ky_batch)  # [B, N, N]

    # Downsample using average pooling
    if downsample_ratio > 1:
        # Add channel dimension for avg_pool2d: [B, 1, N, N]
        intensities = intensities.unsqueeze(1)
        intensities = F.avg_pool2d(intensities, kernel_size=downsample_ratio)
        intensities = intensities.squeeze(1)  # [B, n, n]

    return intensities


def generate_synthetic_study(
    dir_path: str | Path,
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
    downsample_ratio: int,
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
        object_tensor, pupil_tensor, kx_batch, ky_batch, downsample_ratio
    )

    # Save to captures/ directory
    captures_dir = dir_path / "captures"
    captures_dir.mkdir(exist_ok=True)

    for i, cap in enumerate(valid_captures):
        img = captures[i].detach().cpu().numpy()
        np.save(captures_dir / cap.filename, img)

    print(f"Generated {len(valid_captures)} synthetic captures in {captures_dir}")
