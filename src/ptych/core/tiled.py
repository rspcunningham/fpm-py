import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex

from ptych.core.inverse import solve_inverse
from ptych.core.pupil import ZernikeParams, precompute_zernike_basis



def solve_tiled(
    captures: Float[torch.Tensor, "B H W"],
    kx_batch: Float[torch.Tensor, "B"],
    ky_batch: Float[torch.Tensor, "B"],
    pupil: ZernikeParams,
    roi_size: int,
    upsample_ratio: int = 4,
    torch_device: str | torch.device = "cpu",
    on_tile_complete: Callable[[int, int, torch.Tensor, ZernikeParams, dict[str, list[float]]], None] | None = None,
    tile_batch_size: int = 1,
    **kwargs: Any,
) -> Complex[torch.Tensor, "N N"]:
    """Tile a full capture, reconstruct each tile independently, and stitch results.

    Args:
        captures: Capture intensities [B, H, W].
        kx_batch: Normalized k-vectors in x [B].
        ky_batch: Normalized k-vectors in y [B].
        pupil: Initial ZernikeParams (re-initialized per tile batch from these values).
        roi_size: Size of each square tile (pixels).
        upsample_ratio: Super-resolution factor per tile.
        torch_device: Device for solve_inverse.
        on_tile_complete: Callback(row, col, object, pupil, metrics) after each tile.
        tile_batch_size: Number of tiles to solve simultaneously.
        **kwargs: Forwarded to solve_inverse (learn_pupil, checkpoint_interval, etc.).

    Returns:
        Stitched complex object at upsampled resolution.
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
            f"Discarding {remainder_h}px bottom, {remainder_w}px right.",
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

    # 7. Build flat list of tile coordinates and iterate in batches
    tiles = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    for batch_start in range(0, len(tiles), tile_batch_size):
        batch = tiles[batch_start:batch_start + tile_batch_size]
        T = len(batch)

        # a. Stack tile captures: [T, B, n, n]
        batch_captures = torch.stack([
            captures[:, r * roi_size:(r + 1) * roi_size, c * roi_size:(c + 1) * roi_size]
            for r, c in batch
        ])

        # b. Init objects from first capture of each tile: [T, N, N]
        batch_objects = []
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
            torch_device=torch_device,
            **kwargs,
        )
        assert isinstance(solved_pupil, ZernikeParams)

        # e. Unpack into output grid
        for i, (r, c) in enumerate(batch):
            obj_cpu = result_objects[i].cpu()
            out_row = r * upsampled_size
            out_col = c * upsampled_size
            output[out_row:out_row + upsampled_size, out_col:out_col + upsampled_size] = obj_cpu

            # f. Callback
            if on_tile_complete is not None:
                on_tile_complete(r, c, obj_cpu, solved_pupil, metrics)

    return output
