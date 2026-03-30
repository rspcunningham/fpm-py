from pathlib import Path

import numpy as np
import torch
from jaxtyping import Complex

from ptych.core.synthetic import synthesize_captures
from ptych.data.parse import write_manifest
from ptych.data.types import StudyManifest
from ptych.data.utils import prepare_captures


def generate_synthetic_study(
    manifest: StudyManifest,
    output_dir: str | Path,
    object_tensor: Complex[torch.Tensor, "N N"],
    pupil_tensor: Complex[torch.Tensor, "N N"],
) -> None:
    """
    Generate a synthetic study dataset from an in-memory manifest.

    Args:
        manifest: Manifest describing the target synthetic dataset
        output_dir: Directory where info.json and captures/ will be written
        object_tensor: Complex object tensor [N, N]
        pupil_tensor: Complex pupil tensor [N, N]
        (downsample ratio is derived from object_tensor size vs manifest capture_dimensions)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, output_dir / "info.json")

    valid_captures, _, kx_batch, ky_batch = prepare_captures(manifest)

    # Get the downsample ratio from the ideal object and target capture size
    width, height = manifest.capture_dimensions.width, manifest.capture_dimensions.height
    ratio_x = object_tensor.shape[1] / width
    ratio_y = object_tensor.shape[0] / height
    if ratio_x != ratio_y:
        raise ValueError(f"Downsample ratios in x and y do not match: {ratio_x} vs {ratio_y}. Please ensure the desired capture size is the same aspect ratio as the object tensor.")
    if ratio_x != int(ratio_x):
        raise ValueError(f"Downsample ratio is not an integer: {ratio_x}. Please ensure the desired capture size is an integer fraction of the object tensor size.")

    # Generate synthetic captures
    captures = synthesize_captures(
        object_tensor, pupil_tensor, kx_batch, ky_batch, int(ratio_x)
    )

    # Save to captures/ directory
    captures_dir = output_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    for i, cap in enumerate(valid_captures):
        img = captures[i].detach().cpu().numpy()
        np.save(captures_dir / cap.filename, img)

    print(f"Generated {len(valid_captures)} synthetic captures in {captures_dir}")
