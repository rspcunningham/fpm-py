from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Complex, Float
from torch import Tensor
from tqdm import tqdm

from ptych.core.inverse import InversePtychographyModel
from ptych.core.metrics import BatchMetricsRecord, InverseMetrics
from ptych.data.study import PtychStudy


@dataclass
class StudySolveResult:
    object: Complex[Tensor, "N N"]
    pupils: dict[tuple[int, int], Complex[Tensor, "N N"]]
    metrics: list[BatchMetricsRecord]


@dataclass(frozen=True)
class _AxisPatch:
    start: int
    output_start: int
    trim: int
    length: int


@dataclass(frozen=True)
class _Patch:
    y: _AxisPatch
    x: _AxisPatch


@dataclass(frozen=True)
class _PatchCrop:
    source_y: slice
    source_x: slice
    output_y: slice
    output_x: slice


@dataclass(frozen=True)
class _SolvedBatch:
    patches: list[_Patch]
    reconstruction: Float[Tensor, "P N N"]
    object: Complex[Tensor, "P N N"]


def _axis_patches(length: int, patch_size: int) -> list[_AxisPatch]:
    if patch_size > length:
        raise ValueError(
            f"patch_size ({patch_size}) exceeds capture dimension ({length})"
        )

    starts = list(range(0, length - patch_size + 1, patch_size))
    if not starts or starts[-1] != length - patch_size:
        starts.append(length - patch_size)

    patches: list[_AxisPatch] = []
    for idx, start in enumerate(starts):
        trim = max(0, starts[idx - 1] + patch_size - start) if idx else 0
        output_start = start + trim
        owned_length = (
            length - output_start if idx == len(starts) - 1 else patch_size - trim
        )
        patches.append(
            _AxisPatch(
                start=start,
                output_start=output_start,
                trim=trim,
                length=owned_length,
            )
        )
    return patches


def _patch_crop(patch: _Patch, object_to_capture_ratio: int) -> _PatchCrop:
    crop_top = patch.y.trim * object_to_capture_ratio
    crop_left = patch.x.trim * object_to_capture_ratio
    crop_height = patch.y.length * object_to_capture_ratio
    crop_width = patch.x.length * object_to_capture_ratio
    output_top = patch.y.output_start * object_to_capture_ratio
    output_left = patch.x.output_start * object_to_capture_ratio

    return _PatchCrop(
        source_y=slice(crop_top, crop_top + crop_height),
        source_x=slice(crop_left, crop_left + crop_width),
        output_y=slice(output_top, output_top + crop_height),
        output_x=slice(output_left, output_left + crop_width),
    )


def _stitch_patch_field(
    patch_tensors: list[tuple[list[_Patch], Tensor]],
    *,
    height: int,
    width: int,
    object_to_capture_ratio: int,
) -> Tensor:
    first_tensor = next(
        (tensor for _, tensors in patch_tensors for tensor in tensors), None
    )
    if first_tensor is None:
        raise ValueError("Cannot stitch an empty set of patch tensors")

    stitched = torch.zeros(
        height * object_to_capture_ratio,
        width * object_to_capture_ratio,
        dtype=first_tensor.dtype,
        device=first_tensor.device,
    )

    for patches, tensors in patch_tensors:
        for patch, tensor in zip(patches, tensors, strict=True):
            crop = _patch_crop(patch, object_to_capture_ratio)
            stitched[crop.output_y, crop.output_x] = tensor[
                crop.source_y, crop.source_x
            ]

    return stitched


def _train_batch(
    captures: Float[Tensor, "P B n n"],
    kx: Float[Tensor, "B"],
    ky: Float[Tensor, "B"],
    *,
    object_to_capture_ratio: int,
    pupil_radius_fraction: Tensor | float,
    pupil_num_phase_terms: int,
    pupil_num_amp_terms: int,
    epochs: int,
    device: str | torch.device,
) -> tuple[
    Complex[Tensor, "P N N"],
    Complex[Tensor, "P N N"],
    InverseMetrics,
]:
    captures = captures.to(device)
    model = InversePtychographyModel(
        captures,
        kx.to(device),
        ky.to(device),
        object_to_capture_ratio=object_to_capture_ratio,
        pupil_radius_fraction_init=pupil_radius_fraction,
        pupil_num_phase_terms=pupil_num_phase_terms,
        pupil_num_amp_terms=pupil_num_amp_terms,
    ).to(device)
    model.train()

    print("Model parameters:")
    for name, parameter in model.named_parameters():
        print(f"  {name}: {parameter.numel():,} {tuple(parameter.shape)}")
    print(f"  total: {sum(parameter.numel() for parameter in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": [model.object.amplitude], "lr": 1e-2},
            {"params": [model.object.phase], "lr": 1e-1},
            {"params": model.pupil.parameters(), "lr": 1e-3},
        ]
    )

    num_patches, num_captures, _, _ = captures.shape
    loss_history = captures.new_zeros(epochs)
    patch_loss_history = captures.new_zeros(epochs, num_patches)
    capture_loss_history = captures.new_zeros(epochs, num_captures)

    for epoch in tqdm(range(epochs), desc="Solving inverse model..."):
        predicted = model()
        residual = torch.sqrt(predicted + 1e-8) - torch.sqrt(captures + 1e-8)
        squared_residual = residual.square()
        patch_loss = squared_residual.mean(dim=(1, 2, 3))
        loss = patch_loss.sum()
        capture_loss = squared_residual.mean(dim=(0, 2, 3))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history[epoch] = loss.detach()
        patch_loss_history[epoch] = patch_loss.detach()
        capture_loss_history[epoch] = capture_loss.detach()

    metrics: InverseMetrics = {
        "loss": loss_history.cpu().tolist(),
        "patch_loss": patch_loss_history.cpu().tolist(),
        "capture_loss": capture_loss_history.cpu().tolist(),
    }
    with torch.no_grad():
        return (
            model.object().detach().cpu(),
            model.pupil().detach().cpu(),
            metrics,
        )


def solve_study(
    study: PtychStudy,
    *,
    patch_size: int,
    pupil_radius_fraction: Tensor | float,
    object_to_capture_ratio: int = 4,
    pupil_num_phase_terms: int = 5,
    pupil_num_amp_terms: int = 1,
    epochs: int = 1000,
    device: str | torch.device = "cpu",
    batch_size: int = 1,
) -> StudySolveResult:
    captures = study.captures
    kx = study.kx_batch
    ky = study.ky_batch

    _, height, width = captures.shape
    patches = [
        _Patch(y=y_patch, x=x_patch)
        for y_patch in _axis_patches(height, patch_size)
        for x_patch in _axis_patches(width, patch_size)
    ]

    solved_batches: list[_SolvedBatch] = []
    pupils: dict[tuple[int, int], Complex[Tensor, "N N"]] = {}
    metrics: list[BatchMetricsRecord] = []

    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size]
        batch_captures = torch.stack(
            [
                captures[
                    :,
                    patch.y.start : patch.y.start + patch_size,
                    patch.x.start : patch.x.start + patch_size,
                ]
                for patch in batch
            ]
        )

        (
            objects,
            batch_pupils,
            batch_metrics,
        ) = _train_batch(
            batch_captures,
            kx,
            ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_radius_fraction=pupil_radius_fraction,
            pupil_num_phase_terms=pupil_num_phase_terms,
            pupil_num_amp_terms=pupil_num_amp_terms,
            epochs=epochs,
            device=device,
        )
        solved_batches.append(
            _SolvedBatch(
                patches=batch,
                reconstruction=objects.abs().square(),
                object=objects,
            )
        )
        metrics.append(
            {
                "patches": [(patch.y.start, patch.x.start) for patch in batch],
                "metrics": batch_metrics,
            }
        )

        for idx, patch in enumerate(batch):
            pupils[(patch.y.start, patch.x.start)] = batch_pupils[idx]

    stitch_kwargs = {
        "height": height,
        "width": width,
        "object_to_capture_ratio": object_to_capture_ratio,
    }

    object_tensor = _stitch_patch_field(
        [(batch.patches, batch.object) for batch in solved_batches],
        **stitch_kwargs,
    )

    return StudySolveResult(
        object=object_tensor,
        pupils=pupils,
        metrics=metrics,
    )
