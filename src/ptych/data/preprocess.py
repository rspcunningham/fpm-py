from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from jaxtyping import Float

from ptych.data.bayer import demosaic
from ptych.data.types import (
    BayerPattern,
    Capture,
    ColorChannel,
    StudyManifest,
    VALID_COLOR_CHANNELS,
    is_illuminated_capture,
)
from ptych.data.utils import prepare_captures

_CHANNEL_INDICES: dict[ColorChannel, int] = {"R": 0, "G": 1, "B": 2}


@dataclass(frozen=True, kw_only=True)
class ImageCrop:
    top: int
    left: int
    width: int
    height: int

    def bounds(
        self, *, image_height: int, image_width: int
    ) -> tuple[int, int, int, int]:
        if self.top < 0 or self.left < 0:
            raise ValueError(f"Crop top/left must be non-negative, got {self}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Crop width/height must be positive, got {self}")

        bottom = self.top + self.height
        right = self.left + self.width
        if bottom > image_height or right > image_width:
            raise ValueError(
                f"Crop {self} exceeds capture dimensions ({image_height}x{image_width})"
            )

        return self.top, bottom, self.left, right


@dataclass(frozen=True)
class PreprocessedStudyData:
    manifest: StudyManifest
    capture_metadata: list[Capture]
    captures: Float[torch.Tensor, "illumination height width"]
    illumination_kx: Float[torch.Tensor, "illumination"]
    illumination_ky: Float[torch.Tensor, "illumination"]


def _shift_bayer_pattern(
    pattern: BayerPattern, y_offset: int, x_offset: int
) -> BayerPattern:
    tile = ((pattern[0], pattern[1]), (pattern[2], pattern[3]))
    shifted = "".join(
        tile[(y + y_offset) % 2][(x + x_offset) % 2] for y in range(2) for x in range(2)
    )
    return cast(BayerPattern, shifted)


def _bayer_pattern_from_manifest(manifest: StudyManifest) -> BayerPattern:
    if manifest.bayer_pattern is None:
        raise NotImplementedError("Non-Bayer capture preprocessing is not implemented")
    return manifest.bayer_pattern


def centered_square_crop(
    *,
    height: int,
    width: int,
    crop_size: int,
) -> ImageCrop:
    if crop_size <= 0:
        raise ValueError(f"Centered square size must be positive, got {crop_size}")
    if crop_size > width or crop_size > height:
        raise ValueError(
            f"Centered square size ({crop_size}) exceeds capture dimensions ({height}x{width})"
        )

    return ImageCrop(
        top=(height - crop_size) // 2,
        left=(width - crop_size) // 2,
        width=crop_size,
        height=crop_size,
    )


def _crop_bounds(
    crop: ImageCrop | None,
    *,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, image_height, 0, image_width

    return crop.bounds(image_height=image_height, image_width=image_width)


def _capture_channel(capture_index: int, channel: ColorChannel | None) -> ColorChannel:
    if channel is None:
        raise NotImplementedError(
            f"Capture {capture_index} has no channel; non-Bayer preprocessing is not implemented"
        )
    return channel


def _capture_exposure_ms(capture_index: int, exposure_ms: float | None) -> float:
    if exposure_ms is None:
        raise ValueError(f"Capture {capture_index} is missing exposure metadata")
    if exposure_ms <= 0:
        raise ValueError(
            f"Capture {capture_index} has non-positive exposure {exposure_ms}"
        )
    return exposure_ms


def preprocess_study_data(
    manifest: StudyManifest,
    raw_images: Sequence[Float[torch.Tensor, "height width"]],
    *,
    crop: ImageCrop | None = None,
) -> tuple[
    list[Capture],
    Float[torch.Tensor, "illumination height width"],
    Float[torch.Tensor, "illumination"],
    Float[torch.Tensor, "illumination"],
]:
    manifest_bayer_pattern = _bayer_pattern_from_manifest(manifest)
    valid_captures, _, illumination_kx, illumination_ky = prepare_captures(manifest)
    dark_indices = [
        idx
        for idx, capture in enumerate(manifest.captures)
        if not is_illuminated_capture(capture)
    ]

    expected_shape = (
        manifest.capture_dimensions.height,
        manifest.capture_dimensions.width,
    )
    y_top, y_bottom, x_left, x_right = _crop_bounds(
        crop,
        image_height=expected_shape[0],
        image_width=expected_shape[1],
    )

    image_slice = (slice(y_top, y_bottom), slice(x_left, x_right))
    bayer_pattern = _shift_bayer_pattern(manifest_bayer_pattern, y_top, x_left)

    valid_images = [
        raw_images[idx][image_slice]
        for idx, capture in enumerate(manifest.captures)
        if is_illuminated_capture(capture)
    ]
    raw_captures = torch.stack(valid_images)
    demosaiced_captures = demosaic(raw_captures, pattern=bayer_pattern)

    channel_indices = torch.tensor(
        [
            _CHANNEL_INDICES[_capture_channel(i, cap.channel)]
            for i, cap in enumerate(valid_captures)
        ],
        device=demosaiced_captures.device,
    )
    capture_indices = torch.arange(
        len(valid_captures), device=demosaiced_captures.device
    )
    captures_tensor = demosaiced_captures[capture_indices, channel_indices]

    if dark_indices:
        raw_darks = torch.stack([raw_images[idx][image_slice] for idx in dark_indices])
        demosaiced_darks = demosaic(raw_darks, pattern=bayer_pattern)
        dark_captures = [manifest.captures[idx] for idx in dark_indices]
        dark_channels = [
            _capture_channel(idx, capture.channel)
            for idx, capture in zip(dark_indices, dark_captures, strict=True)
        ]
        dark_avg_by_channel: dict[
            ColorChannel, Float[torch.Tensor, "height width"]
        ] = {}
        for channel in VALID_COLOR_CHANNELS:
            dark_rows = [
                row
                for row, dark_channel in enumerate(dark_channels)
                if dark_channel == channel
            ]
            if dark_rows:
                dark_avg_by_channel[channel] = demosaiced_darks[
                    torch.tensor(dark_rows, device=demosaiced_darks.device),
                    _CHANNEL_INDICES[channel],
                ].mean(dim=0)

        if dark_avg_by_channel:
            capture_channels = [
                _capture_channel(i, cap.channel) for i, cap in enumerate(valid_captures)
            ]
            dark_backgrounds = torch.stack(
                [
                    dark_avg_by_channel.get(
                        channel,
                        torch.zeros_like(captures_tensor[0]),
                    )
                    for channel in capture_channels
                ]
            )
            captures_tensor = (captures_tensor - dark_backgrounds).clamp(min=0)

    exposure_ms = torch.tensor(
        [_capture_exposure_ms(i, cap.exposure) for i, cap in enumerate(valid_captures)],
        dtype=captures_tensor.dtype,
        device=captures_tensor.device,
    )
    captures_tensor = captures_tensor / exposure_ms[:, None, None]

    max_value = torch.max(captures_tensor)
    if max_value <= 0:
        raise ValueError(
            "Prepared captures must contain at least one positive intensity value"
        )

    return (
        valid_captures,
        captures_tensor / max_value,
        illumination_kx,
        illumination_ky,
    )


def preprocess_study_data_by_channel(
    manifest: StudyManifest,
    raw_images: Sequence[Float[torch.Tensor, "height width"]],
    *,
    crop: ImageCrop | None = None,
) -> dict[ColorChannel, PreprocessedStudyData]:
    if manifest.bayer_pattern is None:
        raise NotImplementedError("Non-Bayer capture preprocessing is not implemented")
    if len(raw_images) != len(manifest.captures):
        raise ValueError(
            f"Expected {len(manifest.captures)} raw images, got {len(raw_images)}"
        )

    illuminated_indices_by_channel: dict[ColorChannel, list[int]] = {}
    dark_indices_by_channel: dict[ColorChannel, list[int]] = {}
    for idx, capture in enumerate(manifest.captures):
        channel = _capture_channel(idx, capture.channel)
        if is_illuminated_capture(capture):
            illuminated_indices_by_channel.setdefault(channel, []).append(idx)
        else:
            dark_indices_by_channel.setdefault(channel, []).append(idx)

    if not illuminated_indices_by_channel:
        raise ValueError("Study manifest has no illuminated captures")

    preprocessed: dict[ColorChannel, PreprocessedStudyData] = {}
    for channel in VALID_COLOR_CHANNELS:
        if channel not in illuminated_indices_by_channel:
            continue
        selected_indices = [
            *illuminated_indices_by_channel[channel],
            *dark_indices_by_channel.get(channel, []),
        ]
        group_manifest = replace(
            manifest,
            captures=[manifest.captures[idx] for idx in selected_indices],
        )
        group_raw_images = [raw_images[idx] for idx in selected_indices]
        (
            capture_metadata,
            captures_tensor,
            illumination_kx,
            illumination_ky,
        ) = preprocess_study_data(
            group_manifest,
            group_raw_images,
            crop=crop,
        )
        preprocessed[channel] = PreprocessedStudyData(
            manifest=group_manifest,
            capture_metadata=capture_metadata,
            captures=captures_tensor,
            illumination_kx=illumination_kx,
            illumination_ky=illumination_ky,
        )

    return preprocessed
