"""Parsing and serialization utilities for study manifest data."""

import json
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from .types import (
    Capture,
    CaptureDimensions,
    DarkfieldCapture,
    LedPosition,
    ManifestCapture,
    StudyManifest,
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


def manifest_to_dict(manifest: StudyManifest) -> dict[str, object]:
    """Serialize a StudyManifest into the JSON-compatible manifest shape."""
    captures: list[dict[str, object]] = []
    for capture in manifest.captures:
        capture_data: dict[str, object] = {
            "filename": capture.filename,
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
                    captured_at=captured_at,
                    exposure=exposure,
                )
            )
        else:
            captures.append(
                DarkfieldCapture(
                    filename=filename,
                    led_positions=led_positions,
                    captured_at=captured_at,
                    exposure=exposure,
                )
            )

    version = _optional_str(data, "version")
    metadata_raw = data.get("metadata")
    metadata: dict[str, object] = {}
    if metadata_raw is not None:
        if not isinstance(metadata_raw, dict):
            raise ManifestParseError(
                f"Expected dict for 'metadata', got {type(metadata_raw).__name__}"
            )
        metadata = cast(dict[str, object], metadata_raw)

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
        captures=captures,
        version=version if version else "1.0",
        metadata=metadata,
    )
