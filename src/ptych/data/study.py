import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.parse import parse_manifest
from ptych.data.preprocess import preprocess_study_data
from ptych.data.types import StudyManifest


class PtychStudy:
    manifest: StudyManifest
    captures: Float[
        torch.Tensor, "B n n"
    ]  # [B, n, n] demosaiced, exposure-corrected single-channel float intensities normalized to max 1
    kx_batch: Float[
        torch.Tensor, "B"
    ]  # [B] normalized to camera grid (cycles per sample pixel)
    ky_batch: Float[
        torch.Tensor, "B"
    ]  # [B] normalized to camera grid (cycles per sample pixel)

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
    def load(
        cls,
        dataset: str | Path,
        *,
        crop_size: int | None = None,
    ) -> "PtychStudy":
        candidate_path = Path(dataset)
        if candidate_path.exists():
            dir_path = candidate_path
        else:
            from ptych.data.download.dataset_cache import NextcloudDatasetCache

            cache = NextcloudDatasetCache()
            dir_path = cache.fetch_dataset(str(dataset))

        manifest_path = dir_path / "info.json"
        with open(manifest_path) as f:
            manifest = parse_manifest(cast(dict[str, object], json.load(f)))
        expected_shape = (
            manifest.capture_dimensions.height,
            manifest.capture_dimensions.width,
        )
        raw_images: list[Float[torch.Tensor, "H W"]] = []
        for cap in manifest.captures:
            img_path = dir_path / "captures" / cap.filename
            img = cast(npt.NDArray[np.float64], np.load(img_path, mmap_mode="r"))

            assert img.ndim == 2, (
                f"Image must be 2D, got {img.ndim}D for {cap.filename}"
            )
            assert img.shape == expected_shape, (
                f"Image dimensions don't match manifest. "
                f"Expected {expected_shape} (height, width), got {img.shape} for {cap.filename}"
            )
            raw_images.append(torch.from_numpy(np.array(img, dtype=np.float32)))

        captures_tensor, kx_batch, ky_batch = preprocess_study_data(
            manifest,
            raw_images,
            crop_size=crop_size,
        )

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )
