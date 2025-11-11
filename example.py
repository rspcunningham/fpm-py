from fpm_py import reconstruct, kvectors_to_image_series, image_to_tensor
from fpm_py.analysis import plot_comparison_with_histograms
import matplotlib.pyplot as plt
import torch


# Step 1: Load the target image

full_image = image_to_tensor("data/bars.png", to_complex=True)

# Step 2: Define the k-space

# Define optical parameters
wavelength = 550e-9  # 550 nm (green light)
pixel_size = 1e-6    # 1 µm
pupil_radius = 100    # pixels
magnification = 10.0  # 10x

# Create a 5x5 grid of k-vectors
# TODO: implement (x, y, z) --> (kx, ky)
grid_size = 5
spacing = 0.2

k_range = torch.linspace(-(grid_size // 2), grid_size // 2, grid_size) * spacing
kx_vals, ky_vals = torch.meshgrid(k_range, k_range, indexing="ij")
k_vectors = torch.stack((kx_vals.flatten(), ky_vals.flatten()), dim=1)

print(f"Generated {k_vectors.shape[0]} k-vectors for a {grid_size}x{grid_size} grid")
print(f"K-vector range: min={k_vectors.min().item():.3f}, max={k_vectors.max().item():.3f}")

# Step 3: Backwards pass (target + k-vectors --> captures)

dataset = kvectors_to_image_series(
    obj=full_image,
    k_vectors=k_vectors,
    pupil_radius=pupil_radius,
    wavelength=wavelength,
    pixel_size=pixel_size,
    magnification=magnification
)

print(f"Dataset generated with {len(dataset.captures)} captures using direct k-vectors.")

# Step 4: Forward pass (captures + k-vectors --> reconstruction)

target = full_image.abs().cpu().numpy()
output1 = reconstruct(dataset, output_scale_factor=4, max_iters=1).abs().cpu().numpy()
output2 = reconstruct(dataset, output_scale_factor=4, max_iters=10).abs().cpu().numpy()

# Step 5: Assess the outputs

fig, stats = plot_comparison_with_histograms(
    images=[target, output1, output2],
    titles=["Target", "1 Iteration", "10 Iterations"],
    reference_idx=0  # Use first image as reference
)
plt.show()
