from sys import meta_path
from ptych.forward import forward_model
import torch
from tqdm import tqdm

def training_loop(captures: list[torch.Tensor], k_vectors: list[tuple[int, int]], output_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:

    n_captures = len(captures)
    assert n_captures == len(k_vectors), "Captures and k-vectors must match"

    epochs = 100

    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)

    # Initiate the learnable parameters
    O = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)
    P = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)
    downsample_factor = output_size // captures[0].shape[0]

    # Initialize the optimizer
    optimizer = torch.optim.AdamW([O, P], lr=0.1)

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # total epochs
        eta_min=0.001  # minimum LR
    )

    # Prepare batched k-vectors and captures
    kx_batch = torch.tensor([k[0] for k in k_vectors])
    ky_batch = torch.tensor([k[1] for k in k_vectors])
    captures_batch = torch.stack(captures)  # [B, H, W]

    # Telemetry
    metrics = {
        'losses_per_epoch': [],
        'lr_per_epoch': []
    }

    # Training loop
    for _ in tqdm(range(epochs), desc="Solving"):
        # Batched forward pass
        predicted_intensities = forward_model(O, P, kx_batch, ky_batch, downsample_factor)  # [B, H, W]

        # Compute loss across all captures
        #total_loss = torch.nn.functional.mse_loss(predicted_intensities, captures_batch)
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures_batch)
        #total_loss = torch.nn.functional.smooth_l1_loss(predicted_intensities, captures_batch)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['losses_per_epoch'].append(total_loss.item())
        metrics['lr_per_epoch'].append(scheduler.get_last_lr()[0])

    return O.detach(), P.detach(), metrics
