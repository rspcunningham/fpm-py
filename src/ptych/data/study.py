from pathlib import Path
import json
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.bayer import demosaic
from ptych.data.types import StudyManifest
from ptych.data.parse import parse_manifest
from ptych.data.utils import prepare_captures


_RGB_REFERENCE_WAVELENGTHS_M = (
    625e-9,  # red
    525e-9,  # green
    470e-9,  # blue
)


def _channel_index_for_wavelength(wavelength_m: float) -> int:
    return min(
        range(len(_RGB_REFERENCE_WAVELENGTHS_M)),
        key=lambda idx: abs(wavelength_m - _RGB_REFERENCE_WAVELENGTHS_M[idx]),
    )


def _capture_exposure_ms(capture_index: int, exposure_ms: float | None) -> float:
    if exposure_ms is None:
        raise ValueError(f"Capture {capture_index} is missing exposure metadata")
    if exposure_ms <= 0:
        raise ValueError(f"Capture {capture_index} has non-positive exposure {exposure_ms}")
    return exposure_ms


class PtychStudy:
    manifest: StudyManifest
    captures: Float[torch.Tensor, "B n n"] # [B, n, n] demosaiced, exposure-corrected single-channel float intensities normalized to max 1
    kx_batch: Float[torch.Tensor, "B"] # [B] normalized to camera grid (cycles per sample pixel)
    ky_batch: Float[torch.Tensor, "B"] # [B] normalized to camera grid (cycles per sample pixel)

    def __init__(
        self,
        manifest: StudyManifest,
        captures: Float[torch.Tensor, "B n n"],
        kx_batch: Float[torch.Tensor, "B"],
        ky_batch: Float[torch.Tensor, "B"],
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

        valid_captures, _, kx_batch, ky_batch = prepare_captures(manifest)

        # Load capture images from captures/ subdirectory
        images: list[npt.NDArray[np.float64]] = []
        for cap in valid_captures:
            img_path = dir_path / "captures" / cap.filename
            img = cast(npt.NDArray[np.float64], np.load(img_path))

            # Validate image dimensions against manifest
            assert img.ndim == 2, (
                f"Image must be 2D, got {img.ndim}D for {cap.filename}"
            )
            expected_shape = (manifest.capture_dimensions.height, manifest.capture_dimensions.width)
            assert img.shape == expected_shape, (
                f"Image dimensions don't match manifest. "
                f"Expected {expected_shape} (height, width), got {img.shape} for {cap.filename}"
            )
            images.append(img)

        raw_captures = torch.from_numpy(np.stack(images, axis=0)).float()
        demosaiced_captures = demosaic(raw_captures)
        channel_indices = torch.tensor(
            [_channel_index_for_wavelength(cap.wavelength) for cap in valid_captures],
            device=demosaiced_captures.device,
        )
        capture_indices = torch.arange(len(valid_captures), device=demosaiced_captures.device)
        captures_tensor = demosaiced_captures[capture_indices, channel_indices]
        exposure_ms = torch.tensor(
            [
                _capture_exposure_ms(i, cap.exposure)
                for i, cap in enumerate(valid_captures)
            ],
            dtype=captures_tensor.dtype,
            device=captures_tensor.device,
        )
        captures_tensor = captures_tensor / exposure_ms[:, None, None]

        max_value = torch.max(captures_tensor)
        if max_value <= 0:
            raise ValueError("Prepared captures must contain at least one positive intensity value")
        captures_tensor = captures_tensor / max_value

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )

    @classmethod
    def load(cls, dataset: str | Path) -> 'PtychStudy':
        candidate_path = Path(dataset)
        if candidate_path.exists():
            return cls.from_disk(candidate_path)

        from ptych.data.download.dataset_cache import NextcloudDatasetCache

        cache = NextcloudDatasetCache()
        return cls.from_disk(cache.fetch_dataset(str(dataset)))
