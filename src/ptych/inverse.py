from ptych.forward import forward_model
import torch
from tqdm import tqdm

def solve_inverse(captures: list[torch.Tensor], k_vectors: list[tuple[int, int]], output_size: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:

    n_captures = len(captures)
    assert n_captures == len(k_vectors), "Captures and k-vectors must match"

    epochs = 100

    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)

    # Initiate the learnable parameters
    O_real = (0.5 * torch.ones(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    O_imag = (torch.zeros(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    P_real = (0.5 * torch.ones(output_size, output_size, dtype=torch.float32)).requires_grad_(True)
    P_imag = (torch.zeros(output_size, output_size, dtype=torch.float32)).requires_grad_(True)

    O_full = O_real + 1j * O_imag
    P_full = P_real + 1j * P_imag

    downsample_factor = output_size // captures[0].shape[0]

    # Initialize the optimizer
    optimizer = torch.optim.AdamW([O_real, O_imag, P_real, P_imag], lr=0.1)

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
        predicted_intensities = forward_model(O_full, P_full, kx_batch, ky_batch, downsample_factor)  # [B, H, W]

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
