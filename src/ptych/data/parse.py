"""Parsing and serialization utilities for study manifest data."""

import json
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from .types import (
    BayerPattern,
    Capture,
    CaptureDimensions,
    ColorChannel,
    DarkfieldCapture,
    LedPosition,
    ManifestCapture,
    StudyManifest,
    VALID_BAYER_PATTERNS,
    VALID_COLOR_CHANNELS,
    is_illuminated_capture,
)


class ManifestParseError(Exception):
    """Raised when manifest parsing fails due to missing or invalid data."""


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


def _require_nullable_str(
    data: dict[str, object], key: str, context: str = ""
) -> str | None:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestParseError(
            f"{prefix}Expected str or null for '{key}', got {type(value).__name__}"
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


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestParseError(
            f"Expected str for '{key}', got {type(value).__name__}"
        )
    return value


def _optional_num(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestParseError(
            f"Expected number for '{key}', got {type(value).__name__}"
        )
    return float(value)


def _require_bayer_pattern(data: dict[str, object]) -> BayerPattern | None:
    raw_pattern = _require_nullable_str(data, "bayer_pattern")
    if raw_pattern is None:
        return None
    if raw_pattern not in VALID_BAYER_PATTERNS:
        raise ManifestParseError(
            "'bayer_pattern' must be one of "
            f"{list(VALID_BAYER_PATTERNS)} or null, got {raw_pattern!r}"
        )
    return cast(BayerPattern, raw_pattern)


def _require_channel(data: dict[str, object], context: str) -> ColorChannel | None:
    raw_channel = _require_nullable_str(data, "channel", context)
    if raw_channel is None:
        return None
    if raw_channel not in VALID_COLOR_CHANNELS:
        raise ManifestParseError(
            f"{context}: 'channel' must be one of "
            f"{list(VALID_COLOR_CHANNELS)} or null, got {raw_channel!r}"
        )
    return cast(ColorChannel, raw_channel)


def _validate_bayer_channel_consistency(
    bayer_pattern: BayerPattern | None,
    captures: list[ManifestCapture],
) -> None:
    if bayer_pattern is None:
        invalid = [
            capture.filename for capture in captures if capture.channel is not None
        ]
        if invalid:
            raise ManifestParseError(
                "'bayer_pattern' is null, so all capture channels must also be null. "
                f"First invalid capture: {invalid[0]}"
            )
        return

    invalid = [capture.filename for capture in captures if capture.channel is None]
    if invalid:
        raise ManifestParseError(
            "'bayer_pattern' is set, so every capture must define a channel. "
            f"First invalid capture: {invalid[0]}"
        )


def manifest_to_dict(manifest: StudyManifest) -> dict[str, object]:
    """Serialize a StudyManifest into the JSON-compatible manifest shape."""
    captures: list[dict[str, object]] = []
    for capture in manifest.captures:
        capture_data: dict[str, object] = {
            "filename": capture.filename,
            "channel": capture.channel,
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
        if capture.exposure is not None:
            capture_data["exposure"] = capture.exposure
        captures.append(capture_data)

    manifest_data: dict[str, object] = {
        "study_id": str(manifest.study_id),
        "created_at": manifest.created_at.isoformat(),
        "version": manifest.version,
        "magnification": manifest.magnification,
        "numerical_aperture": manifest.numerical_aperture,
        "sensor_pixel_size": manifest.sensor_pixel_size,
        "capture_dimensions": {
            "width": manifest.capture_dimensions.width,
            "height": manifest.capture_dimensions.height,
        },
        "bayer_pattern": manifest.bayer_pattern,
        "captures": captures,
    }
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
    captures_raw = _require_list(data, "captures")
    captures: list[ManifestCapture] = []

    for i, cap_raw in enumerate(captures_raw):
        cap = _require_dict(cap_raw, f"captures[{i}]")

        led_raw = _require_list(cap, "led_positions", f"captures[{i}]")
        led_positions: list[LedPosition] = []
        for j, pos_raw in enumerate(led_raw):
            pos = _require_dict(pos_raw, f"captures[{i}].led_positions[{j}]")
            led_positions.append(
                LedPosition(
                    x=_require_num(pos, "x"),
                    y=_require_num(pos, "y"),
                    z=_require_num(pos, "z"),
                )
            )

        captured_at_str = _optional_str(cap, "captured_at")

        filename = _require_str(cap, "filename", f"captures[{i}]")
        channel = _require_channel(cap, f"captures[{i}]")
        captured_at = (
            datetime.fromisoformat(captured_at_str) if captured_at_str else None
        )
        exposure = _optional_num(cap, "exposure")
        if led_positions:
            captures.append(
                Capture(
                    filename=filename,
                    wavelength=_require_num(cap, "wavelength", f"captures[{i}]"),
                    led_positions=led_positions,
                    channel=channel,
                    captured_at=captured_at,
                    exposure=exposure,
                )
            )
        else:
            if "wavelength" in cap:
                raise ManifestParseError(
                    f"captures[{i}]: Darkfield captures must not define 'wavelength'"
                )
            captures.append(
                DarkfieldCapture(
                    filename=filename,
                    channel=channel,
                    led_positions=led_positions,
                    captured_at=captured_at,
                    exposure=exposure,
                )
            )

    version = _optional_str(data, "version")
    bayer_pattern = _require_bayer_pattern(data)
    _validate_bayer_channel_consistency(bayer_pattern, captures)

    dims_raw = data.get("capture_dimensions")
    if dims_raw is None:
        raise ManifestParseError("Missing required key 'capture_dimensions'")
    dims = _require_dict(dims_raw, "capture_dimensions")
    capture_dimensions = CaptureDimensions(
        width=_require_int(dims, "width", "capture_dimensions"),
        height=_require_int(dims, "height", "capture_dimensions"),
    )

    return StudyManifest(
        study_id=UUID(_require_str(data, "study_id")),
        created_at=datetime.fromisoformat(_require_str(data, "created_at")),
        magnification=_require_num(data, "magnification"),
        numerical_aperture=_require_num(data, "numerical_aperture"),
        sensor_pixel_size=_require_num(data, "sensor_pixel_size"),
        capture_dimensions=capture_dimensions,
        bayer_pattern=bayer_pattern,
        captures=captures,
        version=version if version else "1.0",
    )
