import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float
from tqdm import tqdm

from ptych.core.forward import PtychographicForward
from ptych.core.metrics import InverseMetrics
from ptych.core.pupil import Pupil


def _radius_limits(radius_fraction: torch.Tensor | float) -> tuple[float, float]:
    radius = torch.as_tensor(radius_fraction).detach().flatten().cpu()
    if torch.any(radius <= 0):
        raise ValueError(f"radius_fraction must be positive; got {radius.tolist()}")

    return 0.8 * float(radius.min()), 1.2 * float(radius.max())


def solve_inverse(
    captures: Float[torch.Tensor, "T B n n"],  # [T, B, n, n] float on (0, 1)
    kx_batch: Float[
        torch.Tensor, "B"
    ],  # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ky_batch: Float[
        torch.Tensor, "B"
    ],  # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    *,
    object_to_capture_ratio: int,
    pupil_radius_fraction_init: torch.Tensor | float,
    pupil_num_phase_terms: int = 5,
    pupil_num_amp_terms: int = 1,
    pupil_edge_width_px: float = 2.0,
    pupil_use_softplus: bool = True,
    epochs: int = 1000,
    learn_k_vectors: bool = False,
    torch_device: str | torch.device = "cpu",
) -> tuple[
    Complex[torch.Tensor, "T N N"],
    Complex[torch.Tensor, "T N N"],
    InverseMetrics,
]:

    # Move all tensors to the specified device
    captures = captures.to(torch_device)
    kx_batch = kx_batch.to(torch_device)
    ky_batch = ky_batch.to(torch_device)

    T, B, n, _ = captures.shape
    object_grid_size = n * object_to_capture_ratio

    # Renormalize k-vectors from the capture grid to the object grid.
    kx_batch = kx_batch / object_to_capture_ratio
    ky_batch = ky_batch / object_to_capture_ratio

    learned_tensors = []
    init_amp = F.interpolate(
        captures[:, 0].unsqueeze(1),
        scale_factor=object_to_capture_ratio,
        mode="nearest",
    ).squeeze(1)
    object_amp = torch.sqrt(init_amp + 1e-8).detach().requires_grad_(True)
    object_phase = torch.zeros_like(object_amp).requires_grad_(True)
    learned_tensors.append({"params": object_amp, "lr": 1e-2})
    learned_tensors.append({"params": object_phase, "lr": 1e-1})

    # intensity_scale = torch.ones(B, device=torch_device).requires_grad_(True)  # [B]
    # learned_tensors.append({'params': intensity_scale, 'lr': 1e-2})

    min_radius, max_radius = _radius_limits(pupil_radius_fraction_init)

    pupil_model = Pupil(
        object_grid_size,
        num_phase_terms=pupil_num_phase_terms,
        num_amp_terms=pupil_num_amp_terms,
        radius_fraction=pupil_radius_fraction_init,
        num_tiles=T,
        edge_width_px=pupil_edge_width_px,
        use_softplus=pupil_use_softplus,
        radius_bounds=(min_radius, max_radius),
    ).to(torch_device)
    learned_tensors.append({"params": list(pupil_model.parameters()), "lr": 1e-3})
    image_formation = PtychographicForward(object_grid_size, device=torch_device)

    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append({"params": kx_batch, "lr": 1e-3})
        learned_tensors.append({"params": ky_batch, "lr": 1e-3})
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
        pupil_tensor = pupil_model()

        # Reconstruct complex object from amplitude and phase
        object_complex = object_amp * torch.exp(1j * object_phase)  # [T, N, N]

        # Batched forward pass
        predicted_intensities = image_formation(
            object_complex, pupil_tensor, kx_batch, ky_batch
        )  # [T, B, N, N]
        N2 = predicted_intensities.shape[-1]
        downsampled = F.avg_pool2d(
            predicted_intensities.reshape(T * B, 1, N2, N2),
            kernel_size=object_to_capture_ratio,
            stride=object_to_capture_ratio,
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

    # Transfer loss metrics to CPU in one bulk operation.
    metrics: InverseMetrics = {
        "loss": loss_accumulator.cpu().tolist(),
        "tile_loss": tile_loss_accumulator.cpu().tolist(),
        "capture_loss": capture_loss_accumulator.cpu().tolist(),
    }

    final_object = (object_amp * torch.exp(1j * object_phase)).detach().cpu()
    final_pupil = pupil_model().detach().cpu()
    return final_object, final_pupil, metrics
