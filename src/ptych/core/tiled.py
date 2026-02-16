import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex

from ptych.core.inverse import solve_inverse
from ptych.core.zernike import ZernikeParams, precompute_zernike_basis
from ptych.data.study import PtychStudy

eps = 1e-8


def solve_tiled(
    study: PtychStudy,
    roi_size: int,
    upsample_ratio: int = 4,
    n_captures: int | None = None,
    preprocess: Callable[[Float[torch.Tensor, "B n n"]], Float[torch.Tensor, "B n n"]] | None = None,
    num_phase_terms: int = 10,
    num_amp_terms: int = 10,
    rad_fraction: float = 0.047,
    torch_device: str | torch.device = "cpu",
    on_tile_complete: Callable[[int, int, torch.Tensor, ZernikeParams, dict[str, list[float]]], None] | None = None,
    tile_batch_size: int = 1,
    **kwargs: Any,
) -> Complex[torch.Tensor, "N N"]:
    """Tile a full capture, reconstruct each tile independently, and stitch results.

    Args:
        study: PtychStudy containing captures and k-vectors.
        roi_size: Size of each square tile (pixels).
        upsample_ratio: Super-resolution factor per tile.
        n_captures: Number of captures to use (None = all).
        preprocess: Optional transform applied to captures before tiling
            (e.g. Bayer demosaic). Expected to return a new tensor.
        num_phase_terms: Zernike phase terms per tile.
        num_amp_terms: Zernike amplitude terms per tile.
        rad_fraction: Initial pupil radius fraction.
        torch_device: Device for solve_inverse.
        on_tile_complete: Callback(row, col, object, pupil, metrics) after each tile.
        tile_batch_size: Number of tiles to solve simultaneously.
        **kwargs: Forwarded to solve_inverse (learn_pupil, checkpoint_interval, etc.).

    Returns:
        Stitched complex object at upsampled resolution.
    """
    # 1. Preprocess or clone captures
    if preprocess is not None:
        captures = preprocess(study.captures)
    else:
        captures = study.captures.clone()

    # 2. Slice to n_captures
    kx = study.kx_batch
    ky = study.ky_batch
    if n_captures is not None:
        captures = captures[:n_captures]
        kx = kx[:n_captures]
        ky = ky[:n_captures]

    # 3. Normalize
    captures = captures / captures.max()

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

    # 5. Precompute zernike basis once (shared across all tiles)
    upsampled_size = roi_size * upsample_ratio
    basis = precompute_zernike_basis(
        upsampled_size,
        num_phase_terms=num_phase_terms,
        num_amp_terms=num_amp_terms,
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
            init_amp = torch.sqrt(init_amp + eps)
            init_phase = torch.zeros_like(init_amp)
            batch_objects.append(init_amp * torch.exp(1j * init_phase))
        batch_objects_tensor = torch.stack(batch_objects)  # [T, N, N]

        # c. Fresh ZernikeParams (shared across batch)
        phase_coeffs = torch.zeros(basis.num_phase_terms)
        amp_coeffs = torch.zeros(basis.num_amp_terms)
        amp_coeffs[0] = 1.0
        pupil = ZernikeParams(phase_coeffs, amp_coeffs, basis, torch.tensor(rad_fraction))

        # d. Solve batch
        result_objects, solved_pupil, metrics = solve_inverse(
            batch_captures,
            batch_objects_tensor,
            pupil,
            kx,
            ky,
            torch_device=torch_device,
            **kwargs,
        )
        assert isinstance(solved_pupil, ZernikeParams)

        # e. Unpack into output grid
        for i, (r, c) in enumerate(batch):
            obj_cpu = result_objects[i].cpu()
            or0 = r * upsampled_size
            oc0 = c * upsampled_size
            output[or0:or0 + upsampled_size, oc0:oc0 + upsampled_size] = obj_cpu

            # f. Callback
            if on_tile_complete is not None:
                on_tile_complete(r, c, obj_cpu, solved_pupil, metrics)

    return output
