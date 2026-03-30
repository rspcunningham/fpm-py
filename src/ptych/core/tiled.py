import warnings

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


def solve_tiled_from_inputs(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: ZernikeParams,
    roi_size: int,
    upsample_ratio: int = 4,
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
        roi_size: Size of each square tile (pixels).
        upsample_ratio: Super-resolution factor per tile.
        epochs: Optimization steps per tile batch.
        torch_device: Device for solve_inverse.
        on_tile_complete: Callback(row, col, object, pupil, metrics) after each tile.
        on_batch_complete: Callback(batch_tiles, pupil, metrics) after each batch solve.
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

    if roi_size > H or roi_size > W:
        raise ValueError(
            f"roi_size ({roi_size}) exceeds capture dimensions ({H}x{W})"
        )

    n_rows = H // roi_size
    n_cols = W // roi_size

    remainder_h = H % roi_size
    remainder_w = W % roi_size
    if remainder_h or remainder_w:
        warnings.warn(
            f"Captures ({H}x{W}) not evenly divisible by roi_size ({roi_size}). "
            + f"Discarding {remainder_h}px bottom, {remainder_w}px right.",
            stacklevel=2,
        )

    # 5. Precompute zernike basis at tile resolution (shared across all tiles)
    upsampled_size = roi_size * upsample_ratio
    basis = precompute_zernike_basis(
        upsampled_size,
        num_phase_terms=pupil.basis.num_phase_terms,
        num_amp_terms=pupil.basis.num_amp_terms,
    )

    # 6. Allocate output
    out_H = n_rows * upsampled_size
    out_W = n_cols * upsampled_size
    output = torch.zeros(out_H, out_W, dtype=torch.complex64)
    tile_pupils: dict[tuple[int, int], Complex[torch.Tensor, "N N"]] = {}
    batch_metrics: list[BatchMetricsRecord] = []

    # 7. Build flat list of tile coordinates and iterate in batches
    tiles = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    for batch_start in range(0, len(tiles), tile_batch_size):
        batch = tiles[batch_start:batch_start + tile_batch_size]
        T = len(batch)

        # a. Stack tile captures: [T, B, n, n]
        tile_captures: list[Float[torch.Tensor, "B H W"]] = [
            captures[:, r * roi_size:(r + 1) * roi_size, c * roi_size:(c + 1) * roi_size]
            for r, c in batch
        ]
        batch_captures = torch.stack(tile_captures)

        # b. Init objects from first capture of each tile: [T, N, N]
        batch_objects: list[Complex[torch.Tensor, "N N"]] = []
        for i in range(T):
            init_amp = F.interpolate(
                batch_captures[i, 0:1].unsqueeze(1),
                scale_factor=upsample_ratio,
                mode="nearest",
            ).squeeze()
            init_amp = torch.sqrt(init_amp + 1e-8)
            init_phase = torch.zeros_like(init_amp)
            batch_objects.append(init_amp * torch.exp(1j * init_phase))
        batch_objects_tensor = torch.stack(batch_objects)  # [T, N, N]

        # c. Fresh ZernikeParams from initial values (shared across batch)
        tile_pupil = ZernikeParams(
            pupil.phase_coeffs.clone().detach(),
            pupil.amp_coeffs.clone().detach(),
            basis,
            pupil.rad_fraction.clone().detach(),
        )

        # d. Solve batch
        result_objects, solved_pupil, metrics = solve_inverse(
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
            "tiles": list(batch),
            "metrics": metrics,
        })

        if on_batch_complete is not None:
            on_batch_complete(batch, solved_pupil, metrics)

        solved_pupil_tensor = make_zernike_pupil(
            solved_pupil.phase_coeffs,
            solved_pupil.amp_coeffs,
            solved_pupil.basis,
            solved_pupil.rad_fraction,
        ).cpu()

        # e. Unpack into output grid
        for i, (r, c) in enumerate(batch):
            obj_cpu = result_objects[i].cpu()
            out_row = r * upsampled_size
            out_col = c * upsampled_size
            output[out_row:out_row + upsampled_size, out_col:out_col + upsampled_size] = obj_cpu
            tile_pupils[(r, c)] = solved_pupil_tensor.clone()

            # f. Callback
            if on_tile_complete is not None:
                on_tile_complete(r, c, obj_cpu, solved_pupil, metrics)

    return output, tile_pupils, batch_metrics


def solve_tiled(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: ZernikeParams,
    roi_size: int,
    upsample_ratio: int = 4,
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
        roi_size=roi_size,
        upsample_ratio=upsample_ratio,
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
