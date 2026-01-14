import torch
import torch.nn.functional as F
from tqdm import tqdm
from jaxtyping import Float, Complex

from ptych.forward import forward_model

def solve_inverse(
    captures: Float[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
    object: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    pupil: Complex[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
) -> tuple[Complex[torch.Tensor, "N N"], Complex[torch.Tensor, "N N"], dict[str, list[float]]]:

    epochs = 500

    if object.shape[0] % captures.shape[1] != 0:
            raise ValueError(
                f"Object size ({object.shape[0]}) must be integer multiple of capture size ({captures.shape[1]})"
            )

    upsample_ratio = object.shape[0] // captures.shape[1]

    # renormalize k vectors
    kx_batch = kx_batch / upsample_ratio
    ky_batch = ky_batch / upsample_ratio

    learned_tensors: list[dict[str, torch.Tensor | float]] = []
    object = object.clone().detach().requires_grad_(True)
    learned_tensors.append({'params': object, 'lr': 0.1})

    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': pupil, 'lr': 0.1})
    else:
        pupil = pupil.detach()

    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': kx_batch, 'lr': 0.1})
        learned_tensors.append({'params': ky_batch, 'lr': 0.1})
    else:
        kx_batch = kx_batch.detach()
        ky_batch = ky_batch.detach()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(learned_tensors)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.05,
        total_steps=epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1e4,
    )

    # Telemetry
    metrics: dict[str, list[float]] = {
        'loss': [],
        'lr': []
    }

    # Training loop
    for _ in tqdm(range(epochs), desc="Solving inverse model..."):
        # Batched forward pass
        predicted_intensities = forward_model(object, pupil, kx_batch, ky_batch)  # [B, N, N]
        downsampled = F.avg_pool2d(
            predicted_intensities.unsqueeze(1),  # [B, 1, N, N]
            kernel_size=upsample_ratio,
            stride=upsample_ratio
        ).squeeze(1)  # [B, n, n]

        # Compute loss across all captures
        total_loss = torch.nn.functional.l1_loss(downsampled, captures)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['loss'].append(total_loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

    return object.detach(), pupil.detach(), metrics
