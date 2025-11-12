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

    # Training loop
    losses_per_epoch: list[float] = []
    for epoch in range(50):
        print(f"Starting epoch {epoch}")
        # Forward pass
        losses: list[torch.Tensor] = []
        for i in range(n_captures):
            target_intensity = captures[i]
            k_vector = k_vectors[i]

            predicted_intensity = forward_model(O, P, k_vector[0], k_vector[1])

            # Debug logging for first capture of first epoch
            if epoch == 0 and i == 0:
                print(f"  Capture range: [{target_intensity.min():.3f}, {target_intensity.max():.3f}]")
                print(f"  Predicted range: [{predicted_intensity.min():.3f}, {predicted_intensity.max():.3f}]")

            # Compute loss
            loss = torch.nn.functional.mse_loss(predicted_intensity, target_intensity)
            losses.append(loss)

        # Backward pass
        optimizer.zero_grad()
        total_loss = torch.stack(losses).sum()
        total_loss.backward()

        # Check gradients
        if O.grad is not None and P.grad is not None:
            print(f"  O grad norm: {O.grad.norm().item():.6f}, P grad norm: {P.grad.norm().item():.6f}")
        else:
            print("  WARNING: Gradients are None!")

        optimizer.step()

        # Record loss for this epoch
        losses_per_epoch.append(total_loss.item())
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

    return O.detach(), P.detach(), losses_per_epoch
