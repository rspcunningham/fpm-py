"""Study manifest schema for Fourier ptychography reconstruction."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeGuard
from uuid import UUID

type BayerPattern = Literal["RGGB", "GRBG", "GBRG", "BGGR"]
type ColorChannel = Literal["R", "G", "B"]

VALID_BAYER_PATTERNS: tuple[BayerPattern, ...] = ("RGGB", "GRBG", "GBRG", "BGGR")
VALID_COLOR_CHANNELS: tuple[ColorChannel, ...] = ("R", "G", "B")


@dataclass
class CaptureDimensions:
    """Pixel dimensions of capture images.

    Attributes:
        width: Image width in pixels (number of columns).
        height: Image height in pixels (number of rows).
    """

    width: int
    height: int


@dataclass
class LedPosition:
    """3D position of an LED in the illumination array.

    Attributes:
        x: X coordinate in meters.
        y: Y coordinate in meters.
        z: Z coordinate (height above sample) in meters.
    """

    x: float
    y: float
    z: float


@dataclass
class Capture:
    """A single illuminated image with its illumination parameters.

    Attributes:
        filename: Path to the image file relative to the manifest.
        wavelength: Illumination wavelength in meters.
        led_positions: LED positions active during this capture.
        channel: Bayer color channel represented by this capture.
        captured_at: Timestamp when the image was captured.
        exposure: Exposure time in milliseconds.
    """

    filename: str
    wavelength: float
    led_positions: list[LedPosition]
    channel: ColorChannel | None
    captured_at: datetime | None = None
    exposure: float | None = None

    def __post_init__(self) -> None:
        if not self.led_positions:
            raise ValueError("Illuminated captures require at least one LED position")


@dataclass
class DarkfieldCapture:
    """A dark capture with no active LED illumination."""

    filename: str
    channel: ColorChannel | None
    led_positions: list[LedPosition] = field(default_factory=list)
    captured_at: datetime | None = None
    exposure: float | None = None

    def __post_init__(self) -> None:
        if self.led_positions:
            raise ValueError("Darkfield captures must not include LED positions")


type ManifestCapture = Capture | DarkfieldCapture


def is_illuminated_capture(capture: ManifestCapture) -> TypeGuard[Capture]:
    return isinstance(capture, Capture)


@dataclass
class StudyManifest:
    """Complete manifest for a Fourier ptychography study.

    Attributes:
        study_id: Unique identifier for this study.
        created_at: Timestamp when the study was created.
        magnification: Objective magnification factor (e.g., 4.0 for 4x).
        numerical_aperture: Objective numerical aperture.
        sensor_pixel_size: Physical size of sensor pixels in meters.
        capture_dimensions: Pixel dimensions of all capture images.
        bayer_pattern: Full-frame Bayer color filter array pattern, or None for
            future non-Bayer loading.
        captures: List of captured images. Illuminated captures carry wavelength;
            darkfield captures have no active LED positions and no wavelength.
        version: Manifest schema version.
    """

    study_id: UUID
    created_at: datetime
    magnification: float
    numerical_aperture: float
    sensor_pixel_size: float
    capture_dimensions: CaptureDimensions
    bayer_pattern: BayerPattern | None
    captures: list[ManifestCapture]
    version: str = "1.0"
