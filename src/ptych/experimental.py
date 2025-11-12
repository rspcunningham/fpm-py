from ptych.forward import forward_model
import torch
from tqdm import tqdm

def solve_inverse(
    captures: list[torch.Tensor],
    output_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:

    epochs = 200

    n_captures = len(captures)
    downsample_factor = output_size // captures[0].shape[0]
    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)
    print("Downsample factor:", downsample_factor)

    # Initiate the learnable parameters
    O_real = (0.5 * torch.ones(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    O_imag = (torch.zeros(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    P_real = (0.5 * torch.ones(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    P_imag = (torch.zeros(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    Kx_batch = torch.zeros(n_captures, dtype=torch.float32).requires_grad_(True)
    Ky_batch = torch.zeros(n_captures, dtype=torch.float32).requires_grad_(True)

    O_full = O_real + 1j * O_imag
    P_full = P_real + 1j * P_imag


    # Initialize the optimizer
    optimizer = torch.optim.AdamW([O_real, O_imag, P_real, P_imag, Kx_batch, Ky_batch], lr=0.1)

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # total epochs
        eta_min=0.01  # minimum LR
    )

    captures_batch = torch.stack(captures)  # [B, H, W]

    # Telemetry
    metrics: dict[str, list[float]] = {
        'loss': [],
        'lr': []
    }

    # Training loop
    for _ in tqdm(range(epochs), desc="Solving"):
        # need to do this again so that gradients flow through the optimizer correctly.
        O_full = O_real + 1j * O_imag
        P_full = P_real + 1j * P_imag
        # Batched forward pass
        predicted_intensities = forward_model(O_full, P_full, Kx_batch, Ky_batch, downsample_factor)  # [B, H, W]

        # Compute loss across all captures
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures_batch)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss for this epoch
        metrics['loss'].append(total_loss.item())
        metrics['lr'].append(scheduler.get_last_lr()[0])

    return O_full.detach(), P_full.detach(), metrics
