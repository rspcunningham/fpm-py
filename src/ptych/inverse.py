from ptych.forward import forward_model
import torch
from tqdm import tqdm

from jaxtyping import Float32

def solve_inverse(
    captures: Float32[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
    object: Float32[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    pupil: Float32[torch.Tensor, "N N"], # [N, N] complex on (0, 1)
    kx_batch: Float32[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    ky_batch: Float32[torch.Tensor, "B"], # [B] float on (-0.5, 0.5)
    learn_pupil: bool = True,
    learn_k_vectors: bool = False,
) -> tuple[Float32[torch.Tensor, "N N"], Float32[torch.Tensor, "N N"], dict[str, list[float]]]:

    epochs = 500

    output_size = object.shape[0]
    downsample_factor = output_size // captures[0].shape[0]
    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)
    print("Downsample factor:", downsample_factor)

    learned_tensors: list[dict[str, torch.Tensor | float]] = []
    object = object.clone().detach().requires_grad_(True)
    learned_tensors.append({'params': object, 'lr': 0.1})

    if learn_pupil:
        pupil = pupil.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': pupil, 'lr': 0.1})
    if learn_k_vectors:
        kx_batch = kx_batch.clone().detach().requires_grad_(True)
        ky_batch = ky_batch.clone().detach().requires_grad_(True)
        learned_tensors.append({'params': kx_batch, 'lr': 0.1})
        learned_tensors.append({'params': ky_batch, 'lr': 0.1})

    print(f"Learning {len(learned_tensors)} tensors | pupil:{learn_pupil}, k_vectors:{learn_k_vectors}")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(learned_tensors)

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # total epochs
        eta_min=0.01  # minimum LR
    )

    # Telemetry
    metrics: dict[str, list[float]] = {
        'loss': [],
        'lr': []
    }

    # Training loop
    for _ in tqdm(range(epochs), desc="Solving"):
        # Batched forward pass
        predicted_intensities = forward_model(object, pupil, kx_batch, ky_batch, downsample_factor)  # [B, H, W]

        # Compute loss across all captures
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['loss'].append(total_loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

    return object.detach(), pupil.detach(), metrics
