from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from jaxtyping import Float, Complex

from ptych.core.forward import forward_model
from ptych.core.pupil import ZernikeParams, make_zernike_pupil


def solve_inverse(
    captures: Float[torch.Tensor, "T B n n"], # [T, B, n, n] float on (0, 1)
    object: Complex[torch.Tensor, "T N N"], # [T, N, N] complex on (0, 1)
    pupil: ZernikeParams,
    kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    epochs: int = 1000,
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
    torch_device: str | torch.device = "cpu",
    on_checkpoint: Callable[[int, Complex[torch.Tensor, "T N N"]], None] | None = None,
    checkpoint_interval: int = 50,
) -> tuple[Complex[torch.Tensor, "T N N"], ZernikeParams, dict[str, Any]]:

    # Move all tensors to the specified device
    captures = captures.to(torch_device)
    object = object.to(torch_device)
    kx_batch = kx_batch.to(torch_device)
    ky_batch = ky_batch.to(torch_device)

    T, B, n, _ = captures.shape

    if object.shape[1] % captures.shape[2] != 0:
            raise ValueError(
                f"Object size ({object.shape[1]}) must be integer multiple of capture size ({captures.shape[2]})"
            )

    upsample_ratio = object.shape[1] // captures.shape[2]
    N = object.shape[1]

    # renormalize k vectors
    kx_batch = kx_batch / upsample_ratio
    ky_batch = ky_batch / upsample_ratio

    learned_tensors: list[dict[str, torch.Tensor | float]] = []
    object_amp = torch.abs(object).clone().detach().requires_grad_(True)      # [T, N, N]
    object_phase = torch.angle(object).clone().detach().requires_grad_(True)  # [T, N, N]
    learned_tensors.append({'params': object_amp, 'lr': 1e-2})
    learned_tensors.append({'params': object_phase, 'lr': 1e-2})

    intensity_scale = torch.ones(B, device=torch_device).requires_grad_(True)  # [B]
    learned_tensors.append({'params': intensity_scale, 'lr': 1e-2})

    # Clone/detach coefficients and rad_fraction (basis is fixed, not cloned)
    phase_coeffs = pupil.phase_coeffs.clone().detach().to(torch_device)
    amp_coeffs = pupil.amp_coeffs.clone().detach().to(torch_device)
    rad_fraction = pupil.rad_fraction.clone().detach().to(torch_device)
    basis = pupil.basis.to(torch_device)

    if learn_pupil:
        phase_coeffs = phase_coeffs.requires_grad_(True)
        amp_coeffs = amp_coeffs.requires_grad_(True)
        rad_fraction = rad_fraction.requires_grad_(True)
        learned_tensors.append({'params': phase_coeffs, 'lr': 1e-3})
        learned_tensors.append({'params': amp_coeffs, 'lr': 1e-3})
        learned_tensors.append({'params': rad_fraction, 'lr': 1e-3})

    working_zernike = ZernikeParams(phase_coeffs, amp_coeffs, basis, rad_fraction)

    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': kx_batch, 'lr': 1e-3})
        learned_tensors.append({'params': ky_batch, 'lr': 1e-3})
    else:
        kx_batch = kx_batch.detach()
        ky_batch = ky_batch.detach()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(learned_tensors)

    # Telemetry — accumulate loss on GPU, transfer to CPU only when needed
    loss_accumulator = torch.zeros(epochs, device=torch_device)
    tile_loss_accumulator = torch.zeros(epochs, T, device=torch_device)
    capture_loss_accumulator = torch.zeros(epochs, B, device=torch_device)

    # Training loop
    for epoch in tqdm(range(epochs), desc="Solving inverse model..."):
        pupil_tensor = make_zernike_pupil(
            working_zernike.phase_coeffs,
            working_zernike.amp_coeffs,
            working_zernike.basis,
            working_zernike.rad_fraction,
        )

        # Reconstruct complex object from amplitude and phase
        object_complex = object_amp * torch.exp(1j * object_phase)  # [T, N, N]

        # Batched forward pass
        predicted_intensities = forward_model(object_complex, pupil_tensor, kx_batch, ky_batch)  # [T, B, N, N]
        N2 = predicted_intensities.shape[-1]
        downsampled = F.avg_pool2d(
            predicted_intensities.reshape(T * B, 1, N2, N2),
            kernel_size=upsample_ratio,
            stride=upsample_ratio
        ).reshape(T, B, n, n)  # [T, B, n, n]

        # Compute loss across all captures
        scaled_pred = intensity_scale[None, :, None, None] * downsampled  # [T, B, n, n]
        residual = torch.sqrt(scaled_pred + 1e-8) - torch.sqrt(captures + 1e-8)
        squared_residual = residual.square()
        total_loss = squared_residual.mean()
        tile_loss = squared_residual.mean(dim=(1, 2, 3))
        capture_loss = squared_residual.mean(dim=(0, 2, 3))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record loss on GPU (no sync)
        loss_accumulator[epoch] = total_loss.detach()
        tile_loss_accumulator[epoch] = tile_loss.detach()
        capture_loss_accumulator[epoch] = capture_loss.detach()

        # Checkpoint callback
        if on_checkpoint is not None and epoch % checkpoint_interval == 0:
            object_checkpoint = (object_amp * torch.exp(1j * object_phase)).detach().clone()
            on_checkpoint(epoch, object_checkpoint)

    # Transfer loss history to CPU in one bulk operation
    metrics: dict[str, Any] = {
        'loss': loss_accumulator.cpu().tolist(),
        'tile_loss': tile_loss_accumulator.cpu().tolist(),
        'capture_loss': capture_loss_accumulator.cpu().tolist(),
    }

    # Reconstruct final complex object from optimized amplitude and phase
    object_final = object_amp.detach() * torch.exp(1j * object_phase.detach())  # [T, N, N]

    return (
        object_final,
        ZernikeParams(
            working_zernike.phase_coeffs.detach(),
            working_zernike.amp_coeffs.detach(),
            working_zernike.basis,
            working_zernike.rad_fraction.detach(),
        ),
        metrics
    )
