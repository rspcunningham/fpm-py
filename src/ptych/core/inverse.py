import torch
import torch.nn.functional as F
from tqdm import tqdm
from jaxtyping import Float, Complex

from ptych.core.forward import forward_model
from ptych.core.metrics import InverseMetrics
from ptych.core.pupil import ZernikeBasis, ZernikeParams, make_zernike_pupil_unmasked


def _repeat_initial_pupil(
    pupil: ZernikeParams,
    num_tiles: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ZernikeBasis]:
    phase_coeffs = (
        pupil.phase_coeffs.clone().detach().to(device).unsqueeze(0).repeat(num_tiles, 1)
    )
    amp_coeffs = (
        pupil.amp_coeffs.clone().detach().to(device).unsqueeze(0).repeat(num_tiles, 1)
    )
    rad_fraction = (
        pupil.rad_fraction.clone().detach().to(device).reshape(1).repeat(num_tiles)
    )
    basis = pupil.basis.to(device)
    return phase_coeffs, amp_coeffs, rad_fraction, basis


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
    checkpoint_interval: int = 50,
) -> tuple[list[ZernikeParams], InverseMetrics, list[Complex[torch.Tensor, "T N N"]], list[int]]:

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

    object_to_capture_ratio = object.shape[1] // captures.shape[2]
    # Renormalize k-vectors from the capture grid to the object grid.
    kx_batch = kx_batch / object_to_capture_ratio
    ky_batch = ky_batch / object_to_capture_ratio

    learned_tensors: list[dict[str, torch.Tensor | float]] = []
    object_amp = torch.abs(object).clone().detach().requires_grad_(True)      # [T, N, N]
    object_phase = torch.angle(object).clone().detach().requires_grad_(True)  # [T, N, N]
    learned_tensors.append({'params': object_amp, 'lr': 1e-2})
    learned_tensors.append({'params': object_phase, 'lr': 1e-1})

    #intensity_scale = torch.ones(B, device=torch_device).requires_grad_(True)  # [B]
    #learned_tensors.append({'params': intensity_scale, 'lr': 1e-2})

    # Repeat the initial pupil parameters so each tile has an independent pupil.
    phase_coeffs, amp_coeffs, rad_fraction, basis = _repeat_initial_pupil(
        pupil,
        T,
        torch_device,
    )

    if learn_pupil:
        phase_coeffs = phase_coeffs.requires_grad_(True)
        amp_coeffs = amp_coeffs.requires_grad_(True)
        rad_fraction = rad_fraction.requires_grad_(True)
        learned_tensors.append({'params': phase_coeffs, 'lr': 1e-3})
        learned_tensors.append({'params': amp_coeffs, 'lr': 1e-3})
        learned_tensors.append({'params': rad_fraction, 'lr': 1e-3})

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

    # Reconstruction history accumulation
    history_frames: list[Complex[torch.Tensor, "T N N"]] = []
    history_epochs: list[int] = []

    # Training loop
    for epoch in tqdm(range(epochs), desc="Solving inverse model..."):
        pupil_tensor = torch.stack([
            make_zernike_pupil_unmasked(
                phase_coeffs[tile_idx],
                amp_coeffs[tile_idx],
                basis,
                rad_fraction[tile_idx],
            )
            for tile_idx in range(T)
        ])

        # Reconstruct complex object from amplitude and phase
        object_complex = object_amp * torch.exp(1j * object_phase)  # [T, N, N]

        # Batched forward pass
        predicted_intensities = forward_model(object_complex, pupil_tensor, kx_batch, ky_batch)  # [T, B, N, N]
        N2 = predicted_intensities.shape[-1]
        downsampled = F.avg_pool2d(
            predicted_intensities.reshape(T * B, 1, N2, N2),
            kernel_size=object_to_capture_ratio,
            stride=object_to_capture_ratio
        ).reshape(T, B, n, n)  # [T, B, n, n]

        # Compute loss across all captures
        # scaled_pred = intensity_scale[None, :, None, None] * downsampled  # [T, B, n, n]
        # residual = torch.sqrt(scaled_pred + 1e-8) - torch.sqrt(captures + 1e-8)
        residual = torch.sqrt(downsampled + 1e-8) - torch.sqrt(captures + 1e-8)
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

        # Record reconstruction frame
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            history_frames.append((object_amp * torch.exp(1j * object_phase)).detach().cpu().clone())
            history_epochs.append(epoch)

    # Transfer loss history to CPU in one bulk operation
    metrics: InverseMetrics = {
        "loss": loss_accumulator.cpu().tolist(),
        "tile_loss": tile_loss_accumulator.cpu().tolist(),
        "capture_loss": capture_loss_accumulator.cpu().tolist(),
    }

    # Always include the final epoch
    last_epoch = epochs - 1
    if not history_epochs or history_epochs[-1] != last_epoch:
        history_frames.append((object_amp * torch.exp(1j * object_phase)).detach().cpu().clone())
        history_epochs.append(last_epoch)

    return (
        [
            ZernikeParams(
                phase_coeffs[tile_idx].detach(),
                amp_coeffs[tile_idx].detach(),
                basis,
                rad_fraction[tile_idx].detach(),
            )
            for tile_idx in range(T)
        ],
        metrics,
        history_frames,
        history_epochs,
    )
