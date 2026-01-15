from pathlib import Path
import json
import torch
import numpy as np
from jaxtyping import Float

from ptych.data.types import StudyManifest
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
            manifest = parse_manifest(json.load(f))

        # === Temporary assertions (remove when edge cases are supported) ===
        # Assert all captures have the same wavelength
        wavelengths = [cap.wavelength for cap in manifest.captures]
        assert len(set(wavelengths)) == 1, (
            f"All captures must have the same wavelength. Found: {set(wavelengths)}"
        )
        wavelength = wavelengths[0]
        # === End temporary assertions ===

        # Load capture images
        images: list[np.ndarray] = []
        for cap in manifest.captures:
            img_path = dir_path / cap.filename
            img = np.load(img_path)

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

        # Extract LED positions (using first LED position from each capture)
        led_positions = np.array([
            [cap.led_positions[0].x, cap.led_positions[0].y, cap.led_positions[0].z]
            for cap in manifest.captures
        ])

        # Compute k-vectors
        kx, ky = compute_k_camera(
            led_positions,
            wavelength,
            manifest.sensor_pixel_size,
            manifest.magnification,
        )
        kx_batch = torch.from_numpy(kx).float()
        ky_batch = torch.from_numpy(ky).float()

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )
