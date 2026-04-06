from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from jaxtyping import Complex

from ptych.core.metrics import (
    BatchCompleteCallback,
    BatchMetricsRecord,
    TileCompleteCallback,
)
from ptych.core.pupil import ZernikeParams
from ptych.core.tiled import solve_tiled_from_inputs
from ptych.data.study import PtychStudy
from ptych.data.types import Capture


# A capture range is a tuple of (start,) or (start, stop) defining a half-open range.
# A single-element tuple (start,) means "from start to the end".
CaptureRange = tuple[int] | tuple[int, int]

# A capture selector is either:
#   - None                         → all captures
#   - a list of CaptureRange       → e.g. [(0, 27), (61,)]
CaptureSelector = None | list[CaptureRange]


def _resolve_capture_indices(selector: CaptureSelector, total: int) -> list[int]:
    """Convert a capture selector into a deduplicated list of integer indices."""
    if selector is None:
        return list(range(total))

    indices: list[int] = []
    for r in selector:
        if len(r) == 1:
            indices.extend(range(r[0], total))
        elif len(r) == 2:
            indices.extend(range(r[0], r[1]))
        else:
            raise ValueError(f"Expected 1- or 2-element tuple, got {r}")

    seen: set[int] = set()
    unique: list[int] = []
    for idx in indices:
        if not 0 <= idx < total:
            raise IndexError(f"Capture index {idx} out of range for {total} captures")
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


@dataclass
class StudySolveResult:
    reconstruction_history: Complex[torch.Tensor, "C N N"]
    checkpoint_epochs: list[int]
    tile_pupils: dict[tuple[int, int], Complex[torch.Tensor, "N N"]]
    batch_metrics: list[BatchMetricsRecord]


@dataclass(frozen=True)
class CaptureRegion:
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int

    @classmethod
    def centered_square(cls, *, width: int, height: int, size: int) -> CaptureRegion:
        if size > width or size > height:
            raise ValueError(
                f"Centered square size ({size}) exceeds capture dimensions ({height}x{width})"
            )
        x_left = (width - size) // 2
        y_top = (height - size) // 2
        return cls(
            x_left=x_left,
            x_right=x_left + size,
            y_top=y_top,
            y_bottom=y_top + size,
        )


def _valid_study_captures(study: PtychStudy) -> list[Capture]:
    valid_captures = [capture for capture in study.manifest.captures if capture.led_positions]
    if len(valid_captures) != study.captures.shape[0]:
        raise ValueError(
            "Study manifest captures do not align with loaded study tensors. "
            + "Expected one loaded tensor per valid manifest capture."
        )
    return valid_captures


def _crop_captures(
    captures: torch.Tensor,
    region: CaptureRegion,
) -> torch.Tensor:
    height = captures.shape[-2]
    width = captures.shape[-1]

    if region.x_left < 0 or region.y_top < 0:
        raise ValueError(
            f"Capture region has negative bounds: {region}"
        )
    if region.x_right > width or region.y_bottom > height:
        raise ValueError(
            f"Capture region {region} exceeds capture dimensions ({height}x{width})"
        )
    if region.x_left >= region.x_right or region.y_top >= region.y_bottom:
        raise ValueError(f"Capture region must have positive width and height: {region}")

    return captures[..., region.y_top:region.y_bottom, region.x_left:region.x_right]


def _prepare_study_inputs(
    study: PtychStudy,
    *,
    captures: CaptureSelector = None,
    capture_region: CaptureRegion,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_captures = _valid_study_captures(study)
    total = study.captures.shape[0]

    indices = _resolve_capture_indices(captures, total)
    if not indices:
        raise ValueError("Capture selection must select at least one capture")

    idx_tensor = torch.tensor(indices)
    selected_captures = study.captures[idx_tensor]
    reconstruction_captures = _crop_captures(selected_captures, capture_region)

    return (
        reconstruction_captures,
        study.kx_batch[idx_tensor],
        study.ky_batch[idx_tensor],
    )


def solve_study(
    study: PtychStudy,
    pupil: ZernikeParams,
    *,
    capture_selector: CaptureSelector = None,
    capture_region: CaptureRegion,
    tile_size: int,
    object_to_capture_ratio: int = 4,
    epochs: int = 1000,
    torch_device: str | torch.device = "cpu",
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
    checkpoint_interval: int = 50,
    on_tile_complete: TileCompleteCallback | None = None,
    on_batch_complete: BatchCompleteCallback | None = None,
    tile_batch_size: int = 1,
) -> StudySolveResult:
    captures, kx_batch, ky_batch = _prepare_study_inputs(
        study,
        captures=capture_selector,
        capture_region=capture_region,
    )
    reconstruction_history, checkpoint_epochs, tile_pupils, batch_metrics = solve_tiled_from_inputs(
        captures,
        kx_batch,
        ky_batch,
        pupil,
        tile_size=tile_size,
        object_to_capture_ratio=object_to_capture_ratio,
        epochs=epochs,
        torch_device=torch_device,
        learn_pupil=learn_pupil,
        learn_k_vectors=learn_k_vectors,
        checkpoint_interval=checkpoint_interval,
        on_tile_complete=on_tile_complete,
        on_batch_complete=on_batch_complete,
        tile_batch_size=tile_batch_size,
    )
    return StudySolveResult(
        reconstruction_history=reconstruction_history,
        checkpoint_epochs=checkpoint_epochs,
        tile_pupils=tile_pupils,
        batch_metrics=batch_metrics,
    )
