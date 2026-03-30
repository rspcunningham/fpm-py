from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Complex

from ptych.core.metrics import (
    BatchCompleteCallback,
    BatchMetricsRecord,
    CheckpointCallback,
    TileCompleteCallback,
)
from ptych.core.pupil import ZernikeParams
from ptych.core.tiled import solve_tiled_from_inputs
from ptych.data.bayer import demosaic
from ptych.data.study import PtychStudy
from ptych.data.types import Capture


_RGB_REFERENCE_WAVELENGTHS_M = (
    625e-9,  # red
    525e-9,  # green
    470e-9,  # blue
)


@dataclass
class StudySolveResult:
    stitched_object: Complex[torch.Tensor, "N N"]
    tile_pupils: dict[tuple[int, int], Complex[torch.Tensor, "N N"]]
    batch_metrics: list[BatchMetricsRecord]


def _valid_study_captures(study: PtychStudy) -> list[Capture]:
    valid_captures = [capture for capture in study.manifest.captures if capture.led_positions]
    if len(valid_captures) != study.captures.shape[0]:
        raise ValueError(
            "Study manifest captures do not align with loaded study tensors. "
            + "Expected one loaded tensor per valid manifest capture."
        )
    return valid_captures


def _channel_index_for_wavelength(wavelength_m: float) -> int:
    return min(
        range(len(_RGB_REFERENCE_WAVELENGTHS_M)),
        key=lambda idx: abs(wavelength_m - _RGB_REFERENCE_WAVELENGTHS_M[idx]),
    )


def _prepare_study_inputs(
    study: PtychStudy,
    *,
    n_captures: int = -1,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_captures = _valid_study_captures(study)

    if n_captures == -1:
        selected_count = study.captures.shape[0]
    elif n_captures <= 0:
        raise ValueError("n_captures must be -1 (all captures) or a positive integer")
    else:
        selected_count = min(n_captures, study.captures.shape[0])

    selected_captures = study.captures[:selected_count]
    selected_count = selected_captures.shape[0]

    if selected_count == 0:
        raise ValueError("n_captures must select at least one capture")

    _, height, width = selected_captures.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size ({crop_size}) exceeds capture dimensions ({height}x{width})"
        )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    cropped_captures = selected_captures[:, top:top + crop_size, left:left + crop_size]
    demosaiced_captures = demosaic(cropped_captures)

    selected_metadata = valid_captures[:selected_count]
    channel_indices = torch.tensor(
        [_channel_index_for_wavelength(capture.wavelength) for capture in selected_metadata],
        device=demosaiced_captures.device,
    )
    capture_indices = torch.arange(selected_count, device=demosaiced_captures.device)
    reconstruction_captures = demosaiced_captures[capture_indices, channel_indices]

    return (
        reconstruction_captures / reconstruction_captures.max(),
        study.kx_batch[:selected_count],
        study.ky_batch[:selected_count],
    )


def solve_study(
    study: PtychStudy,
    pupil: ZernikeParams,
    *,
    n_captures: int = -1,
    crop_size: int,
    roi_size: int,
    object_to_capture_ratio: int = 4,
    epochs: int = 1000,
    torch_device: str | torch.device = "cpu",
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
    on_checkpoint: CheckpointCallback | None = None,
    checkpoint_interval: int = 50,
    on_tile_complete: TileCompleteCallback | None = None,
    on_batch_complete: BatchCompleteCallback | None = None,
    tile_batch_size: int = 1,
) -> StudySolveResult:
    captures, kx_batch, ky_batch = _prepare_study_inputs(
        study,
        n_captures=n_captures,
        crop_size=crop_size,
    )
    stitched_object, tile_pupils, batch_metrics = solve_tiled_from_inputs(
        captures,
        kx_batch,
        ky_batch,
        pupil,
        roi_size=roi_size,
        object_to_capture_ratio=object_to_capture_ratio,
        epochs=epochs,
        torch_device=torch_device,
        learn_pupil=learn_pupil,
        learn_k_vectors=learn_k_vectors,
        on_checkpoint=on_checkpoint,
        checkpoint_interval=checkpoint_interval,
        on_tile_complete=on_tile_complete,
        on_batch_complete=on_batch_complete,
        tile_batch_size=tile_batch_size,
    )
    return StudySolveResult(
        stitched_object=stitched_object,
        tile_pupils=tile_pupils,
        batch_metrics=batch_metrics,
    )
