from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.preprocess import ImageCrop, preprocess_study_data
from ptych.data.types import Capture, DarkSubtraction, StudyManifest
from ptych.data.validate import DatasetValidationError, validate_dataset


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
        dark_subtraction: DarkSubtraction = "average_all",
    ) -> "PtychStudy":
        candidate_path = Path(dataset)
        if candidate_path.exists():
            dir_path = candidate_path
        else:
            from ptych.data.download.dataset_cache import NextcloudDatasetCache

            cache = NextcloudDatasetCache()
            dir_path = cache.fetch_dataset(str(dataset))

        dataset_validation = validate_dataset(dir_path)
        manifest = dataset_validation.manifest
        expected_shape = (
            manifest.capture_dimensions.height,
            manifest.capture_dimensions.width,
        )
        if crop is None:
            y_top, y_bottom, x_left, x_right = (
                0,
                expected_shape[0],
                0,
                expected_shape[1],
            )
        else:
            y_top, y_bottom, x_left, x_right = crop.bounds(
                image_height=expected_shape[0],
                image_width=expected_shape[1],
            )

        raw_images: list[Float[torch.Tensor, "height width"]] = []
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
            cropped_image = img[y_top:y_bottom, x_left:x_right]
            if not np.isfinite(cropped_image).all():
                raise DatasetValidationError(
                    f"Capture contains non-finite intensities: {cap.filename}"
                )
            if np.any(cropped_image < 0):
                raise DatasetValidationError(
                    f"Capture contains negative intensities: {cap.filename}"
                )
            raw_images.append(
                torch.from_numpy(np.array(cropped_image, dtype=np.float32))
            )

        (
            capture_metadata,
            captures_tensor,
            illumination_kx,
            illumination_ky,
        ) = preprocess_study_data(
            manifest,
            raw_images,
            crop_offset=(y_top, x_left),
            dark_subtraction=dark_subtraction,
        )

        return cls(
            manifest=manifest,
            capture_metadata=capture_metadata,
            captures=captures_tensor,
            illumination_kx=illumination_kx,
            illumination_ky=illumination_ky,
        )
