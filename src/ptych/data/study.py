import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.parse import parse_manifest
from ptych.data.preprocess import (
    ImageCrop,
    PreprocessedStudyData,
    preprocess_study_data_by_channel,
    preprocess_study_data,
)
from ptych.data.types import Capture, ColorChannel, StudyManifest


class PtychStudy:
    manifest: StudyManifest
    capture_metadata: list[Capture]
    captures: Float[
        torch.Tensor, "illumination height width"
    ]  # Demosaiced, dark-subtracted, exposure-corrected scalar intensities normalized to max 1.
    illumination_kx: Float[
        torch.Tensor, "illumination"
    ]  # Normalized to camera grid (cycles per sample pixel).
    illumination_ky: Float[
        torch.Tensor, "illumination"
    ]  # Normalized to camera grid (cycles per sample pixel).

    def __init__(
        self,
        manifest: StudyManifest,
        capture_metadata: list[Capture],
        captures: Float[torch.Tensor, "illumination height width"],
        illumination_kx: Float[torch.Tensor, "illumination"],
        illumination_ky: Float[torch.Tensor, "illumination"],
    ):
        self.manifest = manifest
        self.capture_metadata = capture_metadata
        self.captures = captures
        self.illumination_kx = illumination_kx
        self.illumination_ky = illumination_ky

    @classmethod
    def load(
        cls,
        dataset: str | Path,
        *,
        crop: ImageCrop | None = None,
    ) -> "PtychStudy":
        manifest, raw_images = _load_manifest_and_raw_images(dataset)

        (
            capture_metadata,
            captures_tensor,
            illumination_kx,
            illumination_ky,
        ) = preprocess_study_data(
            manifest,
            raw_images,
            crop=crop,
        )

        return cls(
            manifest=manifest,
            capture_metadata=capture_metadata,
            captures=captures_tensor,
            illumination_kx=illumination_kx,
            illumination_ky=illumination_ky,
        )

    @classmethod
    def load_by_channel(
        cls,
        dataset: str | Path,
        *,
        crop: ImageCrop | None = None,
    ) -> dict[ColorChannel, "PtychStudy"]:
        manifest, raw_images = _load_manifest_and_raw_images(dataset)
        preprocessed: dict[ColorChannel, PreprocessedStudyData] = (
            preprocess_study_data_by_channel(
                manifest,
                raw_images,
                crop=crop,
            )
        )

        return {
            channel: cls(
                manifest=data.manifest,
                capture_metadata=data.capture_metadata,
                captures=data.captures,
                illumination_kx=data.illumination_kx,
                illumination_ky=data.illumination_ky,
            )
            for channel, data in preprocessed.items()
        }


def _resolve_dataset_path(dataset: str | Path) -> Path:
    candidate_path = Path(dataset)
    if candidate_path.exists():
        return candidate_path

    from ptych.data.download.dataset_cache import NextcloudDatasetCache

    cache = NextcloudDatasetCache()
    return cache.fetch_dataset(str(dataset))


def _load_manifest_and_raw_images(
    dataset: str | Path,
) -> tuple[StudyManifest, list[Float[torch.Tensor, "height width"]]]:
    dir_path = _resolve_dataset_path(dataset)
    manifest_path = dir_path / "info.json"
    with open(manifest_path) as f:
        manifest = parse_manifest(cast(dict[str, object], json.load(f)))
    expected_shape = (
        manifest.capture_dimensions.height,
        manifest.capture_dimensions.width,
    )
    raw_images: list[Float[torch.Tensor, "height width"]] = []
    for cap in manifest.captures:
        img_path = dir_path / "captures" / cap.filename
        img = cast(npt.NDArray[np.float64], np.load(img_path, mmap_mode="r"))

        if img.ndim != 2:
            raise ValueError(f"Image must be 2D, got {img.ndim}D for {cap.filename}")
        if img.shape != expected_shape:
            raise ValueError(
                f"Image dimensions don't match manifest. "
                f"Expected {expected_shape} (height, width), got {img.shape} for {cap.filename}"
            )
        raw_images.append(torch.from_numpy(np.array(img, dtype=np.float32)))

    return manifest, raw_images
