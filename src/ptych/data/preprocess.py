from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import torch
from jaxtyping import Float

from ptych.data.bayer import demosaic
from ptych.data.types import (
    Channel,
    Capture,
    DarkSubtraction,
    StudyManifest,
    is_illuminated_capture,
)
from ptych.data.utils import prepare_captures

_CHANNEL_INDICES: dict[Channel, int] = {"R": 0, "G": 1, "B": 2}


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


def _shift_bayer_pattern(pattern: str, y_offset: int, x_offset: int) -> str:
    tile = ((pattern[0], pattern[1]), (pattern[2], pattern[3]))
    return "".join(
        tile[(y + y_offset) % 2][(x + x_offset) % 2] for y in range(2) for x in range(2)
    )


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


def preprocess_study_data(
    manifest: StudyManifest,
    raw_images: Sequence[Float[torch.Tensor, "height width"]],
    *,
    crop_offset: tuple[int, int] = (0, 0),
    dark_subtraction: DarkSubtraction = "average_all",
) -> tuple[
    list[Capture],
    Float[torch.Tensor, "illumination height width"],
    Float[torch.Tensor, "illumination"],
    Float[torch.Tensor, "illumination"],
]:
    valid_captures, _, illumination_kx, illumination_ky = prepare_captures(manifest)
    y_offset, x_offset = crop_offset
    bayer_pattern = _shift_bayer_pattern(
        manifest.bayer_format,
        y_offset,
        x_offset,
    )

    demosaiced_images = demosaic(torch.stack(list(raw_images)), pattern=bayer_pattern)
    channel_indices = torch.tensor(
        [_CHANNEL_INDICES[capture.channel] for capture in manifest.captures],
        device=demosaiced_images.device,
    )
    image_indices = torch.arange(
        len(manifest.captures),
        device=demosaiced_images.device,
    )
    selected_images = demosaiced_images[image_indices, channel_indices]

    # Dark captures grouped by (channel, exposure); each entry is (captured_at, index).
    dark_frames: dict[tuple[Channel, float], list[tuple[datetime, int]]] = {}
    for index, capture in enumerate(manifest.captures):
        if is_illuminated_capture(capture):
            continue
        key = (capture.channel, capture.exposure)
        dark_frames.setdefault(key, []).append((capture.captured_at, index))

    if dark_subtraction == "average_all":
        dark_averages = {
            key: selected_images[[index for _, index in frames]].mean(dim=0)
            for key, frames in dark_frames.items()
        }

    corrected_images = []
    for index, capture in enumerate(manifest.captures):
        if not is_illuminated_capture(capture):
            continue
        image = selected_images[index]
        if dark_frames:
            key = (capture.channel, capture.exposure)
            if dark_subtraction == "average_all":
                dark = dark_averages[key]
            else:
                # Nearest dark by |captured_at delta|; ties resolve to the earlier one.
                _, nearest_index = min(
                    dark_frames[key],
                    key=lambda frame: (
                        abs(frame[0] - capture.captured_at),
                        frame[0],
                    ),
                )
                dark = selected_images[nearest_index]
            image = (image - dark).clamp(min=0)
        corrected_images.append(image)

    captures_tensor = torch.stack(corrected_images)
    exposure_s = torch.tensor(
        [capture.exposure for capture in valid_captures],
        dtype=captures_tensor.dtype,
        device=captures_tensor.device,
    )
    captures_tensor = captures_tensor / exposure_s[:, None, None]

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
