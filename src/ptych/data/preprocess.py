from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Float

from ptych.data.bayer import demosaic
from ptych.data.types import Capture, StudyManifest, is_illuminated_capture
from ptych.data.utils import prepare_captures

_RGB_REFERENCE_WAVELENGTHS_M = (
    625e-9,  # red
    525e-9,  # green
    470e-9,  # blue
)


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


def _crop_bounds(
    crop: ImageCrop | None,
    *,
    image_height: int,
    image_width: int,
) -> tuple[int, int, int, int]:
    if crop is None:
        return 0, image_height, 0, image_width

    return crop.bounds(image_height=image_height, image_width=image_width)


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
    bayer_pattern = _shift_bayer_pattern(manifest.bayer_format, y_top, x_left)

    valid_images = [
        raw_images[idx][image_slice]
        for idx, capture in enumerate(manifest.captures)
        if is_illuminated_capture(capture)
    ]
    raw_captures = torch.stack(valid_images)
    demosaiced_captures = demosaic(raw_captures, pattern=bayer_pattern)

    if dark_indices:
        raw_darks = torch.stack([raw_images[idx][image_slice] for idx in dark_indices])
        dark_avg = demosaic(raw_darks, pattern=bayer_pattern).mean(dim=0)
        demosaiced_captures = (demosaiced_captures - dark_avg.unsqueeze(0)).clamp(min=0)

    channel_indices = torch.tensor(
        [_channel_index_for_wavelength(cap.wavelength) for cap in valid_captures],
        device=demosaiced_captures.device,
    )
    capture_indices = torch.arange(
        len(valid_captures), device=demosaiced_captures.device
    )
    captures_tensor = demosaiced_captures[capture_indices, channel_indices]
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
