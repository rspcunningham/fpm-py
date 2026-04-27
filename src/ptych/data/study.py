from pathlib import Path
import json
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from ptych.data.bayer import demosaic
from ptych.data.types import StudyManifest, is_illuminated_capture
from ptych.data.parse import parse_manifest
from ptych.data.utils import prepare_captures


_RGB_REFERENCE_WAVELENGTHS_M = (
    625e-9,  # red
    525e-9,  # green
    470e-9,  # blue
)


def _shift_bayer_pattern(pattern: str, y_offset: int, x_offset: int) -> str:
    tile = ((pattern[0], pattern[1]), (pattern[2], pattern[3]))
    return "".join(
        tile[(y + y_offset) % 2][(x + x_offset) % 2] for y in range(2) for x in range(2)
    )


def _centered_square_bounds(
    *,
    height: int,
    width: int,
    size: int,
) -> tuple[int, int, int, int]:
    if size > width or size > height:
        raise ValueError(
            f"Centered square size ({size}) exceeds capture dimensions ({height}x{width})"
        )

    y_top = (height - size) // 2
    x_left = (width - size) // 2
    return y_top, y_top + size, x_left, x_left + size


def _channel_index_for_wavelength(wavelength_m: float) -> int:
    return min(
        range(len(_RGB_REFERENCE_WAVELENGTHS_M)),
        key=lambda idx: abs(wavelength_m - _RGB_REFERENCE_WAVELENGTHS_M[idx]),
    )


def _capture_exposure_ms(capture_index: int, exposure_ms: float | None) -> float:
    if exposure_ms is None:
        raise ValueError(f"Capture {capture_index} is missing exposure metadata")
    if exposure_ms <= 0:
        raise ValueError(
            f"Capture {capture_index} has non-positive exposure {exposure_ms}"
        )
    return exposure_ms


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

    @staticmethod
    def read_manifest(dir_path: str | Path) -> StudyManifest:
        dir_path = Path(dir_path)
        manifest_path = dir_path / "info.json"
        with open(manifest_path) as f:
            return parse_manifest(cast(dict[str, object], json.load(f)))

    @classmethod
    def from_disk(
        cls,
        dir_path: str | Path,
        *,
        crop_size: int | None = None,
    ) -> "PtychStudy":
        dir_path = Path(dir_path)

        # Load and parse manifest
        manifest = cls.read_manifest(dir_path)

        valid_captures, _, kx_batch, ky_batch = prepare_captures(manifest)
        dark_captures = [
            cap for cap in manifest.captures if not is_illuminated_capture(cap)
        ]

        expected_shape = (
            manifest.capture_dimensions.height,
            manifest.capture_dimensions.width,
        )
        if crop_size is None:
            y_top, y_bottom = 0, expected_shape[0]
            x_left, x_right = 0, expected_shape[1]
        else:
            y_top, y_bottom, x_left, x_right = _centered_square_bounds(
                height=expected_shape[0],
                width=expected_shape[1],
                size=crop_size,
            )

        image_slice = (slice(y_top, y_bottom), slice(x_left, x_right))
        bayer_pattern = _shift_bayer_pattern("RGGB", y_top, x_left)

        # Load capture images from captures/ subdirectory
        images: list[npt.NDArray[np.float32]] = []
        for cap in valid_captures:
            img_path = dir_path / "captures" / cap.filename
            img = cast(npt.NDArray[np.float64], np.load(img_path, mmap_mode="r"))

            assert img.ndim == 2, (
                f"Image must be 2D, got {img.ndim}D for {cap.filename}"
            )
            assert img.shape == expected_shape, (
                f"Image dimensions don't match manifest. "
                f"Expected {expected_shape} (height, width), got {img.shape} for {cap.filename}"
            )
            images.append(np.asarray(img[image_slice], dtype=np.float32))

        raw_captures = torch.from_numpy(np.stack(images, axis=0))
        demosaiced_captures = demosaic(raw_captures, pattern=bayer_pattern)

        # Dark-frame subtraction
        if dark_captures:
            dark_images: list[npt.NDArray[np.float32]] = []
            for cap in dark_captures:
                img_path = dir_path / "captures" / cap.filename
                img = cast(npt.NDArray[np.float64], np.load(img_path, mmap_mode="r"))
                assert img.ndim == 2, (
                    f"Image must be 2D, got {img.ndim}D for {cap.filename}"
                )
                assert img.shape == expected_shape, (
                    f"Image dimensions don't match manifest. "
                    f"Expected {expected_shape} (height, width), got {img.shape} for {cap.filename}"
                )
                dark_images.append(np.asarray(img[image_slice], dtype=np.float32))
            raw_darks = torch.from_numpy(np.stack(dark_images, axis=0))
            dark_avg = demosaic(raw_darks, pattern=bayer_pattern).mean(
                dim=0
            )  # [3, H, W]
            demosaiced_captures = (demosaiced_captures - dark_avg.unsqueeze(0)).clamp(
                min=0
            )
        channel_indices = torch.tensor(
            [_channel_index_for_wavelength(cap.wavelength) for cap in valid_captures],
            device=demosaiced_captures.device,
        )
        capture_indices = torch.arange(
            len(valid_captures), device=demosaiced_captures.device
        )
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
            raise ValueError(
                "Prepared captures must contain at least one positive intensity value"
            )
        captures_tensor = captures_tensor / max_value

        return cls(
            manifest=manifest,
            captures=captures_tensor,
            kx_batch=kx_batch,
            ky_batch=ky_batch,
        )

    @classmethod
    def load(
        cls,
        dataset: str | Path,
        *,
        crop_size: int | None = None,
    ) -> "PtychStudy":
        candidate_path = Path(dataset)
        if candidate_path.exists():
            return cls.from_disk(candidate_path, crop_size=crop_size)

        from ptych.data.download.dataset_cache import NextcloudDatasetCache

        cache = NextcloudDatasetCache()
        return cls.from_disk(cache.fetch_dataset(str(dataset)), crop_size=crop_size)
