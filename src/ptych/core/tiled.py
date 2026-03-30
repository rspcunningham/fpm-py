from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex

from ptych.core.inverse import solve_inverse
from ptych.core.metrics import (
    BatchCompleteCallback,
    BatchMetricsRecord,
    CheckpointCallback,
    TileCompleteCallback,
)
from ptych.core.pupil import ZernikeParams, make_zernike_pupil, precompute_zernike_basis


@dataclass(frozen=True)
class _AxisTilePlan:
    start: int
    output_start: int
    trim: int
    length: int


@dataclass(frozen=True)
class _TilePlan:
    y: _AxisTilePlan
    x: _AxisTilePlan


def _axis_tile_plans(length: int, tile_size: int) -> list[_AxisTilePlan]:
    if tile_size > length:
        raise ValueError(
            f"tile_size ({tile_size}) exceeds capture dimension ({length})"
        )

    starts = list(range(0, length - tile_size + 1, tile_size))
    last_start = length - tile_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    plans: list[_AxisTilePlan] = []
    for idx, start in enumerate(starts):
        trim = 0
        if idx > 0:
            trim = max(0, starts[idx - 1] + tile_size - start)

        output_start = start + trim
        length_owned = length - output_start if idx == len(starts) - 1 else tile_size - trim
        plans.append(
            _AxisTilePlan(
                start=start,
                output_start=output_start,
                trim=trim,
                length=length_owned,
            )
        )

    return plans


def solve_tiled_from_inputs(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: ZernikeParams,
    tile_size: int,
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
) -> tuple[
    Complex[torch.Tensor, "N N"],
    dict[tuple[int, int], Complex[torch.Tensor, "N N"]],
    list[BatchMetricsRecord],
]:
    """Tile a full capture, reconstruct each tile independently, and stitch results.

    Args:
        captures: Capture intensities [B, H, W].
        kx_batch: Normalized k-vectors in x [B].
        ky_batch: Normalized k-vectors in y [B].
        pupil: Initial ZernikeParams (re-initialized per tile batch from these values).
        tile_size: Size of each square capture tile (pixels).
        object_to_capture_ratio: Linear ratio between object and capture grids.
        epochs: Optimization steps per tile batch.
        torch_device: Device for solve_inverse.
        on_tile_complete: Callback(y_start, x_start, object, pupil, metrics) after each tile.
        on_batch_complete: Callback(batch_tiles, pupils, metrics) after each batch solve.
        tile_batch_size: Number of tiles to solve simultaneously.
        learn_pupil: Whether to optimize pupil coefficients.
        learn_k_vectors: Whether to optimize illumination k-vectors.
        on_checkpoint: Optional inverse-solver checkpoint callback.
        checkpoint_interval: Epoch spacing for inverse-solver checkpoints.

    Returns:
        Stitched object, one pupil tensor per tile, and per-batch metrics.
    """
    kx = kx_batch
    ky = ky_batch

    # 4. Compute grid
    _, H, W = captures.shape

    y_plans = _axis_tile_plans(H, tile_size)
    x_plans = _axis_tile_plans(W, tile_size)

    # 5. Precompute zernike basis at tile resolution (shared across all tiles)
    upsampled_size = tile_size * object_to_capture_ratio
    basis = precompute_zernike_basis(
        upsampled_size,
        num_phase_terms=pupil.basis.num_phase_terms,
        num_amp_terms=pupil.basis.num_amp_terms,
    )

    # 6. Allocate output
    out_H = H * object_to_capture_ratio
    out_W = W * object_to_capture_ratio
    output = torch.zeros(out_H, out_W, dtype=torch.complex64)
    tile_pupils: dict[tuple[int, int], Complex[torch.Tensor, "N N"]] = {}
    batch_metrics: list[BatchMetricsRecord] = []

    # 7. Build flat list of tile coordinates and iterate in batches
    tiles = [_TilePlan(y=y_plan, x=x_plan) for y_plan in y_plans for x_plan in x_plans]

    for batch_start in range(0, len(tiles), tile_batch_size):
        batch = tiles[batch_start:batch_start + tile_batch_size]
        T = len(batch)

        # a. Stack tile captures: [T, B, n, n]
        tile_captures: list[Float[torch.Tensor, "B H W"]] = [
            captures[
                :,
                tile.y.start:tile.y.start + tile_size,
                tile.x.start:tile.x.start + tile_size,
            ]
            for tile in batch
        ]
        batch_captures = torch.stack(tile_captures)

        # b. Init objects from first capture of each tile: [T, N, N]
        batch_objects: list[Complex[torch.Tensor, "N N"]] = []
        for i in range(T):
            init_amp = F.interpolate(
                batch_captures[i, 0:1].unsqueeze(1),
                scale_factor=object_to_capture_ratio,
                mode="nearest",
            ).squeeze()
            init_amp = torch.sqrt(init_amp + 1e-8)
            init_phase = torch.zeros_like(init_amp)
            batch_objects.append(init_amp * torch.exp(1j * init_phase))
        batch_objects_tensor = torch.stack(batch_objects)  # [T, N, N]

        # c. Fresh initial ZernikeParams copied per tile inside solve_inverse
        tile_pupil = ZernikeParams(
            pupil.phase_coeffs.clone().detach(),
            pupil.amp_coeffs.clone().detach(),
            basis,
            pupil.rad_fraction.clone().detach(),
        )

        # d. Solve batch
        result_objects, solved_pupils, metrics = solve_inverse(
            batch_captures,
            batch_objects_tensor,
            tile_pupil,
            kx,
            ky,
            epochs=epochs,
            learn_pupil=learn_pupil,
            learn_k_vectors=learn_k_vectors,
            torch_device=torch_device,
            on_checkpoint=on_checkpoint,
            checkpoint_interval=checkpoint_interval,
        )

        batch_metrics.append({
            "tiles": [(tile.y.start, tile.x.start) for tile in batch],
            "metrics": metrics,
        })

        if on_batch_complete is not None:
            on_batch_complete(
                [(tile.y.start, tile.x.start) for tile in batch],
                solved_pupils,
                metrics,
            )

        # e. Unpack into output grid
        for i, tile in enumerate(batch):
            obj_cpu = result_objects[i].cpu()
            solved_pupil = solved_pupils[i]
            solved_pupil_tensor = make_zernike_pupil(
                solved_pupil.phase_coeffs,
                solved_pupil.amp_coeffs,
                solved_pupil.basis,
                solved_pupil.rad_fraction,
            ).cpu()
            crop_top = tile.y.trim * object_to_capture_ratio
            crop_left = tile.x.trim * object_to_capture_ratio
            crop_height = tile.y.length * object_to_capture_ratio
            crop_width = tile.x.length * object_to_capture_ratio
            obj_owned = obj_cpu[
                crop_top:crop_top + crop_height,
                crop_left:crop_left + crop_width,
            ]

            out_row = tile.y.output_start * object_to_capture_ratio
            out_col = tile.x.output_start * object_to_capture_ratio
            output[out_row:out_row + crop_height, out_col:out_col + crop_width] = obj_owned
            tile_pupils[(tile.y.start, tile.x.start)] = solved_pupil_tensor

            # f. Callback
            if on_tile_complete is not None:
                on_tile_complete(tile.y.start, tile.x.start, obj_cpu, solved_pupil, metrics)

    return output, tile_pupils, batch_metrics


def solve_tiled(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: ZernikeParams,
    tile_size: int,
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
) -> Complex[torch.Tensor, "N N"]:
    """Compatibility wrapper for the prepared-input tiled solver."""
    output, _, _ = solve_tiled_from_inputs(
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
        on_checkpoint=on_checkpoint,
        checkpoint_interval=checkpoint_interval,
        on_tile_complete=on_tile_complete,
        on_batch_complete=on_batch_complete,
        tile_batch_size=tile_batch_size,
    )
    return output
