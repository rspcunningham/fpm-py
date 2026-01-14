"""Parsing utilities for study manifest data."""
from datetime import datetime
from typing import cast
from uuid import UUID

from .types import Capture, LedPosition, StudyManifest


class ManifestParseError(Exception):
    """Raised when manifest parsing fails due to missing or invalid data."""


def _require_str(data: dict[str, object], key: str, context: str = "") -> str:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if not isinstance(value, str):
        raise ManifestParseError(f"{prefix}Expected str for '{key}', got {type(value).__name__}")
    return value


def _require_num(data: dict[str, object], key: str, context: str = "") -> float:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestParseError(f"{prefix}Expected number for '{key}', got {type(value).__name__}")
    return float(value)


def _require_list(data: dict[str, object], key: str, context: str = "") -> list[object]:
    prefix = f"{context}: " if context else ""
    if key not in data:
        raise ManifestParseError(f"{prefix}Missing required key '{key}'")
    value = data[key]
    if not isinstance(value, list):
        raise ManifestParseError(f"{prefix}Expected list for '{key}', got {type(value).__name__}")
    return cast(list[object], value)


def _require_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestParseError(f"{context}: Expected dict, got {type(value).__name__}")
    return cast(dict[str, object], value)


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestParseError(f"Expected str for '{key}', got {type(value).__name__}")
    return value


def _optional_num(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestParseError(f"Expected number for '{key}', got {type(value).__name__}")
    return float(value)


def parse_manifest(data: dict[str, object]) -> StudyManifest:
    """
    Parse a dictionary (from JSON) into a StudyManifest.

    Raises:
        ManifestParseError: If required keys are missing or values have incorrect types.
    """
    captures_raw = _require_list(data, "captures")
    captures: list[Capture] = []

    for i, cap_raw in enumerate(captures_raw):
        cap = _require_dict(cap_raw, f"captures[{i}]")

        led_raw = _require_list(cap, "led_positions", f"captures[{i}]")
        led_positions: list[LedPosition] = []
        for j, pos_raw in enumerate(led_raw):
            pos = _require_dict(pos_raw, f"captures[{i}].led_positions[{j}]")
            led_positions.append(LedPosition(
                x=_require_num(pos, "x"),
                y=_require_num(pos, "y"),
                z=_require_num(pos, "z"),
            ))

        captured_at_str = _optional_str(cap, "captured_at")

        captures.append(Capture(
            filename=_require_str(cap, "filename", f"captures[{i}]"),
            wavelength=_require_num(cap, "wavelength", f"captures[{i}]"),
            led_positions=led_positions,
            captured_at=datetime.fromisoformat(captured_at_str) if captured_at_str else None,
            exposure=_optional_num(cap, "exposure"),
        ))

    version = _optional_str(data, "version")
    metadata_raw = data.get("metadata")
    metadata: dict[str, object] = {}
    if metadata_raw is not None:
        if not isinstance(metadata_raw, dict):
            raise ManifestParseError(f"Expected dict for 'metadata', got {type(metadata_raw).__name__}")
        metadata = cast(dict[str, object], metadata_raw)

    return StudyManifest(
        study_id=UUID(_require_str(data, "study_id")),
        created_at=datetime.fromisoformat(_require_str(data, "created_at")),
        magnification=_require_num(data, "magnification"),
        sensor_pixel_size=_require_num(data, "sensor_pixel_size"),
        captures=captures,
        version=version if version else "1.0",
        metadata=metadata,
    )
