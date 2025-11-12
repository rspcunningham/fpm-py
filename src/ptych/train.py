from ptych.forward import forward_model
import torch
from tqdm import tqdm

def training_loop(captures: list[torch.Tensor], k_vectors: list[tuple[int, int]], output_size: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

    n_captures = len(captures)
    assert n_captures == len(k_vectors), "Captures and k-vectors must match"

    print("Training loop started")
    print("Capture size:", captures[0].shape[0])
    print("Output size:", output_size)

    # Initiate the learnable parameters
    O = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)
    P = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)
    downsample_factor = output_size // captures[0].shape[0]

    # Initialize the optimizer
    optimizer = torch.optim.AdamW([O, P], lr=0.1)


    # Prepare batched k-vectors and captures
    kx_batch = torch.tensor([k[0] for k in k_vectors])
    ky_batch = torch.tensor([k[1] for k in k_vectors])
    captures_batch = torch.stack(captures)  # [B, H, W]

    # Training loop
    losses_per_epoch: list[float] = []
    for _ in tqdm(range(50), desc="Solving"):
        # Batched forward pass
        predicted_intensities = forward_model(O, P, kx_batch, ky_batch, downsample_factor)  # [B, H, W]

        # Compute loss across all captures
        #total_loss = torch.nn.functional.mse_loss(predicted_intensities, captures_batch)
        total_loss = torch.nn.functional.l1_loss(predicted_intensities, captures_batch)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record loss for this epoch
        losses_per_epoch.append(total_loss.item())

    return O.detach(), P.detach(), losses_per_epoch
