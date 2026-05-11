from __future__ import annotations

import json
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
        raise DatasetValidationError("dataset name must not be empty")
    if "/" in normalized or "\\" in normalized:
        raise DatasetValidationError("dataset name must be a flat directory name")
    if normalized in {".", "..", ".staging"}:
        raise DatasetValidationError("dataset name must be a normal directory name")
    return normalized


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

    try:
        with manifest_path.open(encoding="utf-8") as fh:
            manifest_data = json.load(fh)
        if not isinstance(manifest_data, dict):
            raise DatasetValidationError("info.json root must be an object")
        manifest = parse_manifest(cast(dict[str, object], manifest_data))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"info.json is invalid JSON: {exc}") from exc
    except (ManifestParseError, ValueError) as exc:
        raise DatasetValidationError(f"info.json is invalid: {exc}") from exc

    if not manifest.captures:
        raise DatasetValidationError("Dataset has no captures")

    expected_shape = (
        manifest.capture_dimensions.height,
        manifest.capture_dimensions.width,
    )
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise DatasetValidationError("capture_dimensions must be positive")

    filenames = [capture.filename for capture in manifest.captures]
    if len(set(filenames)) != len(filenames):
        raise DatasetValidationError("Dataset contains duplicate capture filenames")

    capture_paths: list[Path] = []
    total_bytes = manifest_path.stat().st_size
    for filename in filenames:
        _validate_capture_filename(filename)
        capture_path = captures_dir / filename
        if not capture_path.is_file():
            raise DatasetValidationError(f"Dataset is missing capture file {filename}")
        _validate_capture_array(capture_path, expected_shape)
        capture_paths.append(capture_path)
        total_bytes += capture_path.stat().st_size

    illuminated = [
        capture for capture in manifest.captures if is_illuminated_capture(capture)
    ]
    if not illuminated:
        raise DatasetValidationError("Dataset has no illuminated captures")

    multi_led = [
        capture.filename for capture in illuminated if len(capture.led_positions) != 1
    ]
    if multi_led:
        first = multi_led[0]
        raise DatasetValidationError(
            f"Multi-LED captures are not supported. First invalid capture: {first}"
        )

    return ValidatedDataset(
        root=root,
        manifest=manifest,
        capture_paths=capture_paths,
        total_bytes=total_bytes,
    )


def _validate_capture_filename(filename: str) -> None:
    if not filename or filename in {".", ".."}:
        raise DatasetValidationError("Capture filename must be a basename")
    if "/" in filename or "\\" in filename:
        raise DatasetValidationError(f"Capture filename must be flat: {filename}")
    if Path(filename).name != filename:
        raise DatasetValidationError(f"Capture filename must be a basename: {filename}")
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
                f"Capture shape mismatch for {path.name}: expected {expected_shape}, got {array.shape}"
            )
    finally:
        del array
