from fpm_py import reconstruct, kvectors_to_image_series, image_to_tensor
from fpm_py.analysis import plot_comparison_with_histograms
import matplotlib.pyplot as plt
import torch

from fpm_py.core.backward import xyz_to_kxky


def generate_grid_points(x_spacing: float, y_spacing: float, x_n: int, y_n: int, z: float) -> list[tuple[float, float, float]]:
    """
    Generate a grid of (x, y, z) points centered at origin.

    Args:
        x_spacing: Spacing between points in x direction (meters)
        y_spacing: Spacing between points in y direction (meters)
        x_n: Number of points in x direction
        y_n: Number of points in y direction
        z: Fixed z coordinate for all points (meters)

    Returns:
        List of (x, y, z) tuples representing grid points
    """
    points = []

    # Calculate centered offsets
    x_start = -(x_n - 1) / 2
    y_start = -(y_n - 1) / 2

    for i in range(x_n):
        for j in range(y_n):
            x = (x_start + i) * x_spacing
            y = (y_start + j) * y_spacing
            points.append((x, y, z))

    return points


# Step 1: Load the target image

full_image = image_to_tensor("data/bars.png", to_complex=True)

# Step 2: Define the k-space

# Define optical parameters
wavelength = 550e-9  # 550 nm (green light)
pixel_size = 1e-6    # 1 µm
pupil_radius = 100    # pixels
magnification = 1.0  # 10x

z = 100e-3  # 100 mm
x_spacing = 10e-3 # 10 mm
y_spacing = 10e-3 # 10 mm

# Generate 3x3 grid of points
points = generate_grid_points(x_spacing, y_spacing, x_n=10, y_n=10, z=z)

k_vectors = torch.stack([xyz_to_kxky(point, wavelength, pixel_size, 9.48e-3, 9.48e-3) for point in points])

print(f"Generated {len(k_vectors)} k-vectors")
print(f"K vectors: {k_vectors}")

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
output1 = reconstruct(dataset, output_scale_factor=20, max_iters=1).abs().cpu().numpy()
output2 = reconstruct(dataset, output_scale_factor=20, max_iters=10).abs().cpu().numpy()

# Step 5: Assess the outputs

fig, stats = plot_comparison_with_histograms(
    images=[target, output1, output2],
    titles=["Target", "1 Iteration", "10 Iterations"],
    reference_idx=0  # Use first image as reference
)
plt.show()
