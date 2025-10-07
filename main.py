from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

from fpm_py.core.forward import reconstruct
from fpm_py.core.backward import kvectors_to_image_series
from fpm_py.analysis.metrics import plot_comparison_with_histograms
from fpm_py.utils.data_utils import image_to_tensor
import matplotlib.pyplot as plt
import torch

# Path to test image
im_path = "data/bars.png"
full_image = image_to_tensor(im_path, to_complex=True)

# Define optical parameters
wavelength = 550e-9  # 550nm (green light)
pixel_size = 1e-6    # 1 micron pixel size
pupil_radius = 100    # Pupil radius in pixels
magnification = 10.0  # System magnification (10x objective)

# Create a 5x5 grid of k-vectors
grid_size = 5
spacing = 0.2

k_range = torch.linspace(-(grid_size // 2), grid_size // 2, grid_size) * spacing
kx_vals, ky_vals = torch.meshgrid(k_range, k_range, indexing="ij")
k_vectors = torch.stack((kx_vals.flatten(), ky_vals.flatten()), dim=1)

print(f"Generated {k_vectors.shape[0]} k-vectors for a {grid_size}x{grid_size} grid")
print(f"K-vector range: min={k_vectors.min().item():.3f}, max={k_vectors.max().item():.3f}")

# Generate dataset using direct k-vectors
# Note: we don't pass du, it will be calculated automatically
dataset = kvectors_to_image_series(
    obj=full_image,
    k_vectors=k_vectors,
    pupil_radius=pupil_radius,
    wavelength=wavelength,
    pixel_size=pixel_size,
    magnification=magnification
)

# Print basic info about the dataset
print(f"Dataset generated with {len(dataset.captures)} captures using direct k-vectors.")

# Visualize the captures
#fig = visualize_image_series(dataset)
#plt.show()

target = full_image.abs().cpu().numpy()
output1 = reconstruct(dataset, output_scale_factor=4, max_iters=1).abs().cpu().numpy()
output2 = reconstruct(dataset, output_scale_factor=4, max_iters=10).abs().cpu().numpy()

# Use the new function with your outputs - now accepting a list of images
fig, stats = plot_comparison_with_histograms(
    images=[target, output1, output2],  # Now pass as a list
    titles=["Target", "1 Iteration", "10 Iterations"],
    reference_idx=0  # Use first image as reference (optional)
)
plt.show()
