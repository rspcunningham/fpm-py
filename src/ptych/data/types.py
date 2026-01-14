"""Study manifest schema for Fourier ptychography reconstruction."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


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
    """A single captured image with its illumination parameters.

    Attributes:
        filename: Path to the image file relative to the manifest.
        wavelength: Illumination wavelength in meters.
        led_positions: LED positions active during this capture.
        captured_at: Timestamp when the image was captured.
        exposure: Exposure time in milliseconds.
    """

    filename: str
    wavelength: float
    led_positions: list[LedPosition]
    captured_at: datetime | None = None
    exposure: float | None = None


@dataclass
class StudyManifest:
    """Complete manifest for a Fourier ptychography study.

    Attributes:
        study_id: Unique identifier for this study.
        created_at: Timestamp when the study was created.
        magnification: Objective magnification factor (e.g., 4.0 for 4x).
        sensor_pixel_size: Physical size of sensor pixels in meters.
        captures: List of captured images with illumination data.
        version: Manifest schema version.
        metadata: Arbitrary user-defined metadata.
    """

    study_id: UUID
    created_at: datetime
    magnification: float
    sensor_pixel_size: float
    captures: list[Capture]
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]
