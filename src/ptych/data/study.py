from pathlib import Path
import json
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.types import StudyManifest
from ptych.data.parse import parse_manifest
from ptych.data.utils import prepare_captures


class PtychStudy:
    manifest: StudyManifest
    captures: Float[torch.Tensor, "B n n"] # [B, n, n] float intensities
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

        # Stack into tensor [B, n, n]
        captures_tensor = torch.from_numpy(np.stack(images, axis=0)).float()

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )

    @classmethod
    def load(cls, dataset_id: str) -> 'PtychStudy':
        from ptych.data.download.dataset_cache import NextcloudDatasetCache

        cache = NextcloudDatasetCache()
        return cls.from_disk(cache.fetch_dataset(dataset_id))
