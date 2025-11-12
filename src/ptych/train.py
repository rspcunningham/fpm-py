from ptych.forward import forward_model
import torch

def training_loop(captures: list[torch.Tensor], k_vectors: list[tuple[int, int]], output_size: int) -> tuple[torch.Tensor, torch.Tensor, list[float]]:

    # Initiate the learnable parameters
    O = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)
    P = (0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)).requires_grad_(True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam([O, P], lr=0.1)

    n_captures = len(captures)
    assert n_captures == len(k_vectors), "Captures and k-vectors must match"

    # Prepare batched k-vectors and captures
    kx_batch = torch.tensor([k[0] for k in k_vectors])
    ky_batch = torch.tensor([k[1] for k in k_vectors])
    captures_batch = torch.stack(captures)  # [B, H, W]

    # Training loop
    losses_per_epoch: list[float] = []
    for epoch in range(50):
        print(f"Starting epoch {epoch}")

        # Batched forward pass - all captures at once!
        predicted_intensities = forward_model(O, P, kx_batch, ky_batch)  # [B, H, W]

        # Debug logging for first epoch
        if epoch == 0:
            print(f"  Capture range: [{captures_batch.min():.3f}, {captures_batch.max():.3f}]")
            print(f"  Predicted range: [{predicted_intensities.min():.3f}, {predicted_intensities.max():.3f}]")

        # Compute loss across all captures
        total_loss = torch.nn.functional.mse_loss(predicted_intensities, captures_batch)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        if O.grad is not None and P.grad is not None:
            print(f"  O grad norm: {torch.abs(O.grad).norm().item():.6f}, P grad norm: {torch.abs(P.grad).norm().item():.6f}")
        else:
            print("  WARNING: Gradients are None!")

        optimizer.step()

        # Record loss for this epoch
        losses_per_epoch.append(total_loss.item())
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

    return O.detach(), P.detach(), losses_per_epoch
