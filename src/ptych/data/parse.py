"""Parsing and serialization utilities for study manifest data."""

import json
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from .types import (
    BAYER_FORMATS,
    CHANNELS,
    BayerFormat,
    Capture,
    CaptureDimensions,
    Channel,
    DarkfieldCapture,
    LedPosition,
    ManifestCapture,
    StudyManifest,
    is_illuminated_capture,
)


class ManifestParseError(Exception):
    """Raised when manifest parsing fails due to missing or invalid data."""


_ROOT_KEYS = {
    "study_id",
    "created_at",
    "magnification",
    "numerical_aperture",
    "sensor_pixel_size",
    "bayer_format",
    "capture_dimensions",
    "captures",
    "metadata",
}
_ILLUMINATED_CAPTURE_KEYS = {
    "filename",
    "wavelength",
    "channel",
    "exposure",
    "led_positions",
    "captured_at",
}
_DARK_CAPTURE_KEYS = {
    "filename",
    "channel",
    "exposure",
    "led_positions",
    "captured_at",
}
_LED_POSITION_KEYS = {"x", "y", "z"}
_CAPTURE_DIMENSION_KEYS = {"width", "height"}


def _reject_unknown_keys(
    data: dict[str, object],
    allowed_keys: set[str],
    context: str,
) -> None:
    unknown_keys = sorted(data.keys() - allowed_keys)
    if unknown_keys:
        formatted = ", ".join(repr(key) for key in unknown_keys)
        raise ManifestParseError(f"{context}: Unknown field(s): {formatted}")


def _require_str(data: dict[str, object], key: str, context: str = "") -> str:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if not isinstance(value, str):
        raise ManifestParseError(
            f"{prefix}Expected str for '{key}', got {type(value).__name__}"
        )
    return value


def _require_num(data: dict[str, object], key: str, context: str = "") -> float:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestParseError(
            f"{prefix}Expected number for '{key}', got {type(value).__name__}"
        )
    return float(value)


def _require_int(data: dict[str, object], key: str, context: str = "") -> int:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestParseError(
            f"{prefix}Expected int for '{key}', got {type(value).__name__}"
        )
    return value


def _require_bayer_format(data: dict[str, object]) -> BayerFormat:
    value = _require_str(data, "bayer_format")
    if value not in BAYER_FORMATS:
        supported = ", ".join(BAYER_FORMATS)
        raise ManifestParseError(
            f"Expected 'bayer_format' to be one of {supported}, got {value!r}"
        )
    return cast(BayerFormat, value)


def _require_channel(data: dict[str, object], context: str) -> Channel:
    value = _require_str(data, "channel", context)
    if value not in CHANNELS:
        supported = ", ".join(CHANNELS)
        raise ManifestParseError(
            f"{context}: Expected 'channel' to be one of {supported}, got {value!r}"
        )
    return cast(Channel, value)


def _require_list(data: dict[str, object], key: str, context: str = "") -> list[object]:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if not isinstance(value, list):
        raise ManifestParseError(
            f"{prefix}Expected list for '{key}', got {type(value).__name__}"
        )
    return cast(list[object], value)


def _require_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestParseError(
            f"{context}: Expected dict, got {type(value).__name__}"
        )
    return cast(dict[str, object], value)


def _optional_str(
    data: dict[str, object],
    key: str,
    context: str = "",
) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if not isinstance(value, str):
        prefix = f"{context}: " if context else ""
        raise ManifestParseError(
            f"{prefix}Expected str for '{key}', got {type(value).__name__}"
        )
    return value


def _parse_datetime(value: str, context: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManifestParseError(
            f"{context}: Expected an ISO 8601 timestamp, got {value!r}"
        ) from exc


def _parse_uuid(value: str, context: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise ManifestParseError(
            f"{context}: Expected a valid UUID, got {value!r}"
        ) from exc


def manifest_to_dict(manifest: StudyManifest) -> dict[str, object]:
    """Serialize a StudyManifest into the JSON-compatible manifest shape."""
    captures: list[dict[str, object]] = []
    for capture in manifest.captures:
        capture_data: dict[str, object] = {
            "filename": capture.filename,
            "channel": capture.channel,
            "exposure": capture.exposure,
            "led_positions": [
                {
                    "x": position.x,
                    "y": position.y,
                    "z": position.z,
                }
                for position in capture.led_positions
            ],
        }
        if is_illuminated_capture(capture):
            capture_data["wavelength"] = capture.wavelength
        if capture.captured_at is not None:
            capture_data["captured_at"] = capture.captured_at.isoformat()
        captures.append(capture_data)

    manifest_data: dict[str, object] = {
        "study_id": str(manifest.study_id),
        "created_at": manifest.created_at.isoformat(),
        "magnification": manifest.magnification,
        "numerical_aperture": manifest.numerical_aperture,
        "sensor_pixel_size": manifest.sensor_pixel_size,
        "bayer_format": manifest.bayer_format,
        "capture_dimensions": {
            "width": manifest.capture_dimensions.width,
            "height": manifest.capture_dimensions.height,
        },
        "captures": captures,
    }
    if manifest.metadata:
        manifest_data["metadata"] = manifest.metadata
    return manifest_data


def write_manifest(manifest: StudyManifest, path: str | Path) -> None:
    """Write a StudyManifest to disk as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest_to_dict(manifest), fh, indent=4)
        fh.write("\n")


def parse_manifest(data: dict[str, object]) -> StudyManifest:
    """
    Parse a dictionary (from JSON) into a StudyManifest.

    Raises:
        ManifestParseError: If required keys are missing or values have incorrect types.
    """
    _reject_unknown_keys(data, _ROOT_KEYS, "root")
    captures_raw = _require_list(data, "captures")
    captures: list[ManifestCapture] = []

    for i, cap_raw in enumerate(captures_raw):
        capture_context = f"captures[{i}]"
        cap = _require_dict(cap_raw, capture_context)

        led_raw = _require_list(cap, "led_positions", capture_context)
        led_positions: list[LedPosition] = []
        for j, pos_raw in enumerate(led_raw):
            position_context = f"{capture_context}.led_positions[{j}]"
            pos = _require_dict(pos_raw, position_context)
            _reject_unknown_keys(pos, _LED_POSITION_KEYS, position_context)
            led_positions.append(
                LedPosition(
                    x=_require_num(pos, "x", position_context),
                    y=_require_num(pos, "y", position_context),
                    z=_require_num(pos, "z", position_context),
                )
            )

        captured_at_str = _optional_str(cap, "captured_at", capture_context)
        filename = _require_str(cap, "filename", capture_context)
        channel = _require_channel(cap, capture_context)
        exposure = _require_num(cap, "exposure", capture_context)
        captured_at = (
            _parse_datetime(captured_at_str, f"{capture_context}.captured_at")
            if captured_at_str is not None
            else None
        )
        if led_positions:
            _reject_unknown_keys(
                cap,
                _ILLUMINATED_CAPTURE_KEYS,
                capture_context,
            )
            captures.append(
                Capture(
                    filename=filename,
                    wavelength=_require_num(cap, "wavelength", capture_context),
                    channel=channel,
                    exposure=exposure,
                    led_positions=led_positions,
                    captured_at=captured_at,
                )
            )
        else:
            _reject_unknown_keys(cap, _DARK_CAPTURE_KEYS, capture_context)
            captures.append(
                DarkfieldCapture(
                    filename=filename,
                    channel=channel,
                    exposure=exposure,
                    led_positions=led_positions,
                    captured_at=captured_at,
                )
            )

    metadata: dict[str, object] = {}
    if "metadata" in data:
        metadata_raw = data["metadata"]
        if not isinstance(metadata_raw, dict):
            raise ManifestParseError(
                f"Expected dict for 'metadata', got {type(metadata_raw).__name__}"
            )
        metadata = cast(dict[str, object], metadata_raw)

    dims_raw = data.get("capture_dimensions")
    if dims_raw is None:
        raise ManifestParseError("Missing required key 'capture_dimensions'")
    dims = _require_dict(dims_raw, "capture_dimensions")
    _reject_unknown_keys(dims, _CAPTURE_DIMENSION_KEYS, "capture_dimensions")
    capture_dimensions = CaptureDimensions(
        width=_require_int(dims, "width", "capture_dimensions"),
        height=_require_int(dims, "height", "capture_dimensions"),
    )

    return StudyManifest(
        study_id=_parse_uuid(_require_str(data, "study_id"), "study_id"),
        created_at=_parse_datetime(_require_str(data, "created_at"), "created_at"),
        magnification=_require_num(data, "magnification"),
        numerical_aperture=_require_num(data, "numerical_aperture"),
        sensor_pixel_size=_require_num(data, "sensor_pixel_size"),
        bayer_format=_require_bayer_format(data),
        capture_dimensions=capture_dimensions,
        captures=captures,
        metadata=metadata,
    )
