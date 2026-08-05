from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from ptych.data.parse import ManifestParseError, parse_manifest
from ptych.data.types import StudyManifest, is_illuminated_capture


class DatasetValidationError(Exception):
    pass


@dataclass(frozen=True)
class ValidatedDataset:
    root: Path
    manifest: StudyManifest
    capture_paths: list[Path]
    total_bytes: int


def normalize_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip()
    if not normalized:
        raise DatasetValidationError("Dataset name must not be empty")
    if "/" in normalized or "\\" in normalized:
        raise DatasetValidationError("Dataset name must be a flat directory name")
    if normalized in {".", "..", ".staging"}:
        raise DatasetValidationError("Dataset name must be a normal directory name")
    return normalized


def load_manifest(path: str | Path) -> StudyManifest:
    manifest_path = Path(path)
    try:
        with manifest_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"info.json is invalid JSON: {exc}") from exc

    return parse_manifest_data(data)


def parse_manifest_data(data: object) -> StudyManifest:
    if not isinstance(data, dict):
        raise DatasetValidationError("info.json root must be an object")
    try:
        manifest = parse_manifest(cast(dict[str, object], data))
    except (ManifestParseError, ValueError) as exc:
        raise DatasetValidationError(f"info.json is invalid: {exc}") from exc

    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: StudyManifest) -> None:
    _require_positive_finite("magnification", manifest.magnification)
    _require_positive_finite("numerical_aperture", manifest.numerical_aperture)
    _require_positive_finite("sensor_pixel_size", manifest.sensor_pixel_size)

    if (
        manifest.capture_dimensions.width <= 0
        or manifest.capture_dimensions.height <= 0
    ):
        raise DatasetValidationError("capture_dimensions must be positive")
    if not manifest.captures:
        raise DatasetValidationError("Dataset has no captures")

    filenames = [capture.filename for capture in manifest.captures]
    if len(set(filenames)) != len(filenames):
        raise DatasetValidationError("Dataset contains duplicate capture filenames")

    illuminated = [
        capture for capture in manifest.captures if is_illuminated_capture(capture)
    ]
    darkfields = [
        capture for capture in manifest.captures if not is_illuminated_capture(capture)
    ]
    if not illuminated:
        raise DatasetValidationError("Dataset has no illuminated captures")

    for capture_index, capture in enumerate(manifest.captures):
        _validate_capture_filename(capture.filename)
        _require_positive_finite(
            f"captures[{capture_index}].exposure",
            capture.exposure,
        )

        if not is_illuminated_capture(capture):
            continue

        _require_positive_finite(
            f"captures[{capture_index}].wavelength",
            capture.wavelength,
        )
        if len(capture.led_positions) != 1:
            raise DatasetValidationError(
                f"captures[{capture_index}] must contain exactly one LED position"
            )
        led_position = capture.led_positions[0]
        for axis, value in (
            ("x", led_position.x),
            ("y", led_position.y),
            ("z", led_position.z),
        ):
            if not math.isfinite(value):
                raise DatasetValidationError(
                    f"captures[{capture_index}].led_positions[0].{axis} must be finite"
                )
        if led_position.z <= 0:
            raise DatasetValidationError(
                f"captures[{capture_index}].led_positions[0].z must be positive"
            )

    wavelengths = {capture.wavelength for capture in illuminated}
    if len(wavelengths) != 1:
        raise DatasetValidationError(
            "All illuminated captures must use one wavelength. "
            f"Found: {sorted(wavelengths)}"
        )

    if darkfields:
        illuminated_dark_keys = {
            (capture.channel, capture.exposure) for capture in illuminated
        }
        dark_keys = {(capture.channel, capture.exposure) for capture in darkfields}
        missing_dark_keys = sorted(illuminated_dark_keys - dark_keys)
        if missing_dark_keys:
            raise DatasetValidationError(
                "Missing dark captures for channel/exposure pair(s): "
                f"{missing_dark_keys}"
            )
        unused_dark_keys = sorted(dark_keys - illuminated_dark_keys)
        if unused_dark_keys:
            raise DatasetValidationError(
                "Dark captures have unused channel/exposure pair(s): "
                f"{unused_dark_keys}"
            )


def validate_dataset(dataset_dir: str | Path) -> ValidatedDataset:
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset path is not a directory: {root}")

    manifest_path = root / "info.json"
    if not manifest_path.is_file():
        raise DatasetValidationError("Dataset is missing info.json")

    captures_dir = root / "captures"
    if not captures_dir.is_dir():
        raise DatasetValidationError("Dataset is missing captures/")

    manifest = load_manifest(manifest_path)
    expected_names = {capture.filename for capture in manifest.captures}
    actual_names = {entry.name for entry in captures_dir.iterdir()}
    missing_names = sorted(expected_names - actual_names)
    if missing_names:
        raise DatasetValidationError(
            f"Dataset is missing capture file(s): {missing_names}"
        )
    extra_names = sorted(actual_names - expected_names)
    if extra_names:
        raise DatasetValidationError(
            f"Dataset contains unreferenced capture entries: {extra_names}"
        )

    expected_shape = (
        manifest.capture_dimensions.height,
        manifest.capture_dimensions.width,
    )
    capture_paths: list[Path] = []
    total_bytes = manifest_path.stat().st_size
    for capture in manifest.captures:
        capture_path = captures_dir / capture.filename
        if not capture_path.is_file():
            raise DatasetValidationError(
                f"Capture path is not a file: {capture.filename}"
            )
        _validate_capture_array(capture_path, expected_shape)
        capture_paths.append(capture_path)
        total_bytes += capture_path.stat().st_size

    return ValidatedDataset(
        root=root,
        manifest=manifest,
        capture_paths=capture_paths,
        total_bytes=total_bytes,
    )


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise DatasetValidationError(f"{name} must be positive and finite; got {value}")


def _validate_capture_filename(filename: str) -> None:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise DatasetValidationError(
            f"Capture filename must be a flat basename: {filename!r}"
        )
    if Path(filename).suffix != ".npy":
        raise DatasetValidationError(f"Capture file must be .npy: {filename}")


def _validate_capture_array(path: Path, expected_shape: tuple[int, int]) -> None:
    try:
        array = cast(npt.NDArray[np.generic], np.load(path, mmap_mode="r"))
    except Exception as exc:
        raise DatasetValidationError(
            f"Capture file is not a readable .npy: {path.name}"
        ) from exc

    try:
        if not isinstance(array, np.ndarray):
            raise DatasetValidationError(
                f"Capture file must contain one NumPy array: {path.name}"
            )
        if array.ndim != 2:
            raise DatasetValidationError(
                f"Capture file must be 2D, got {array.ndim}D: {path.name}"
            )
        if array.shape != expected_shape:
            raise DatasetValidationError(
                f"Capture shape mismatch for {path.name}: "
                f"expected {expected_shape}, got {array.shape}"
            )
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        ):
            raise DatasetValidationError(
                f"Capture array must have a real numeric dtype: "
                f"{path.name} has {array.dtype}"
            )
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise DatasetValidationError(
                f"Capture array must contain only finite values: {path.name}"
            )
        if np.issubdtype(array.dtype, np.signedinteger) or np.issubdtype(
            array.dtype, np.floating
        ):
            if np.any(np.less(array, 0)):
                raise DatasetValidationError(
                    f"Capture array must contain only non-negative values: {path.name}"
                )
    finally:
        del array
