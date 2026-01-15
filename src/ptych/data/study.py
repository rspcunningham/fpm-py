from pathlib import Path
import json
import warnings
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.types import Capture, StudyManifest
from ptych.data.parse import parse_manifest
from ptych.utils import compute_k_camera


class PtychStudy:
    manifest: StudyManifest
    captures: Float[torch.Tensor, "B n n"] # [B, n, n] float on (0, 1)
    kx_batch: Float[torch.Tensor, "B"] # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ky_batch: Float[torch.Tensor, "B"] # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)

    def __init__(
        self,
        manifest: StudyManifest,
        captures: Float[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
        kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
        ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ):
        self.manifest = manifest
        self.captures = captures
        self.kx_batch = kx_batch
        self.ky_batch = ky_batch

    @classmethod
    def from_disk(cls, dir_path: str | Path) -> 'PtychStudy':
        dir_path = Path(dir_path)

        # Load and parse manifest
        manifest_path = dir_path / "info.json"
        with open(manifest_path) as f:
            manifest = parse_manifest(cast(dict[str, object], json.load(f)))

        # Filter out darkfield captures (empty led_positions)
        valid_captures: list[Capture] = []
        for i, cap in enumerate(manifest.captures):
            if not cap.led_positions:
                warnings.warn(
                    f"Capture {i} ({cap.filename}) is a darkfield image. Darkfield processing is not supported yet so this image will be ignored."
                )
            else:
                valid_captures.append(cap)

        # === Temporary assertions (remove when edge cases are supported) ===
        # Assert all captures have the same wavelength
        wavelengths = [cap.wavelength for cap in valid_captures]
        assert len(set(wavelengths)) == 1, (
            f"All captures must have the same wavelength. Found: {set(wavelengths)}"
        )
        wavelength = wavelengths[0]
        # Assert all captures have exactly one LED position
        for i, cap in enumerate(valid_captures):
            assert len(cap.led_positions) == 1, (
                f"Multi-LED captures are not supported yet. "
                f"Capture {i} ({cap.filename}) has {len(cap.led_positions)} LED positions."
            )
        # === End temporary assertions ===

        # Load capture images from captures/ subdirectory
        images: list[npt.NDArray[np.float64]] = []
        for cap in valid_captures:
            img_path = dir_path / "captures" / cap.filename
            img = cast(npt.NDArray[np.float64], np.load(img_path))

            # === Temporary assertions (remove when edge cases are supported) ===
            assert img.ndim == 2, (
                f"Image must be 2D, got {img.ndim}D for {cap.filename}"
            )
            assert img.shape[0] == img.shape[1], (
                f"Image must be square (n x n), got {img.shape} for {cap.filename}"
            )
            if images:
                assert img.shape == images[0].shape, (
                    f"All images must have same dimensions. "
                    f"Expected {images[0].shape}, got {img.shape} for {cap.filename}"
                )
            # === End temporary assertions ===

            images.append(img)

        # Stack into tensor [B, n, n]
        captures_tensor = torch.from_numpy(np.stack(images, axis=0)).float()

        # Compute k-vectors (using first LED position from each capture)
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

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )
