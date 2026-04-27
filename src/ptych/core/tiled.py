from dataclasses import dataclass

import torch
from jaxtyping import Complex, Float

from ptych.core.inverse import solve_inverse
from ptych.core.metrics import BatchMetricsRecord
from ptych.core.pupil import Pupil


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
        length_owned = (
            length - output_start if idx == len(starts) - 1 else tile_size - trim
        )
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
    pupil: Pupil,
    tile_size: int,
    object_to_capture_ratio: int = 4,
    epochs: int = 1000,
    torch_device: str | torch.device = "cpu",
    learn_k_vectors: bool = False,
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
        pupil: Initial pupil parameters, re-initialized per tile batch.
        tile_size: Size of each square capture tile (pixels).
        object_to_capture_ratio: Linear ratio between object and capture grids.
        epochs: Optimization steps per tile batch.
        torch_device: Device for solve_inverse.
        tile_batch_size: Number of tiles to solve simultaneously.
        learn_k_vectors: Whether to optimize illumination k-vectors.

    Returns:
        Final stitched reconstruction, one pupil tensor per tile, and per-batch metrics.
    """
    kx = kx_batch
    ky = ky_batch

    # 4. Compute grid
    _, H, W = captures.shape

    y_plans = _axis_tile_plans(H, tile_size)
    x_plans = _axis_tile_plans(W, tile_size)

    # 6. Allocate output
    out_H = H * object_to_capture_ratio
    out_W = W * object_to_capture_ratio
    tile_pupils: dict[tuple[int, int], Complex[torch.Tensor, "N N"]] = {}
    batch_metrics: list[BatchMetricsRecord] = []
    batch_results: list[tuple[list[_TilePlan], Complex[torch.Tensor, "T N N"]]] = []

    # 7. Build flat list of tile coordinates and iterate in batches
    tiles = [_TilePlan(y=y_plan, x=x_plan) for y_plan in y_plans for x_plan in x_plans]

    for batch_start in range(0, len(tiles), tile_batch_size):
        batch = tiles[batch_start : batch_start + tile_batch_size]

        # a. Stack tile captures: [T, B, n, n]
        tile_captures: list[Float[torch.Tensor, "B H W"]] = [
            captures[
                :,
                tile.y.start : tile.y.start + tile_size,
                tile.x.start : tile.x.start + tile_size,
            ]
            for tile in batch
        ]
        batch_captures = torch.stack(tile_captures)

        # b. Solve batch
        result_objects, result_pupils, metrics = solve_inverse(
            batch_captures,
            kx,
            ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_num_phase_terms=pupil.num_phase_terms,
            pupil_num_amp_terms=pupil.num_amp_terms,
            pupil_radius_fraction_init=pupil.radius_fraction,
            pupil_edge_width_px=pupil.edge_width_px,
            pupil_use_softplus=pupil.use_softplus,
            epochs=epochs,
            learn_k_vectors=learn_k_vectors,
            torch_device=torch_device,
        )

        batch_results.append((batch, result_objects))

        batch_metrics.append(
            {
                "tiles": [(tile.y.start, tile.x.start) for tile in batch],
                "metrics": metrics,
            }
        )

        # e. Record tile pupils.
        for i, tile in enumerate(batch):
            solved_pupil_tensor = result_pupils[i].cpu()
            tile_pupils[(tile.y.start, tile.x.start)] = solved_pupil_tensor

    # Merge final reconstructed tile objects across all batches.
    reconstruction = torch.zeros(out_H, out_W, dtype=torch.complex64)
    for tiles_in_batch, object_tensor in batch_results:
        for tile_idx, tile in enumerate(tiles_in_batch):
            crop_top = tile.y.trim * object_to_capture_ratio
            crop_left = tile.x.trim * object_to_capture_ratio
            crop_height = tile.y.length * object_to_capture_ratio
            crop_width = tile.x.length * object_to_capture_ratio
            obj_owned = object_tensor[tile_idx][
                crop_top : crop_top + crop_height,
                crop_left : crop_left + crop_width,
            ]
            out_row = tile.y.output_start * object_to_capture_ratio
            out_col = tile.x.output_start * object_to_capture_ratio
            reconstruction[
                out_row : out_row + crop_height,
                out_col : out_col + crop_width,
            ] = obj_owned

    return reconstruction, tile_pupils, batch_metrics


def solve_tiled(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: Pupil,
    tile_size: int,
    object_to_capture_ratio: int = 4,
    epochs: int = 1000,
    torch_device: str | torch.device = "cpu",
    learn_k_vectors: bool = False,
    tile_batch_size: int = 1,
) -> Complex[torch.Tensor, "N N"]:
    """Compatibility wrapper for the prepared-input tiled solver."""
    reconstruction, _, _ = solve_tiled_from_inputs(
        captures,
        kx_batch,
        ky_batch,
        pupil,
        tile_size=tile_size,
        object_to_capture_ratio=object_to_capture_ratio,
        epochs=epochs,
        torch_device=torch_device,
        learn_k_vectors=learn_k_vectors,
        tile_batch_size=tile_batch_size,
    )
    return reconstruction
