import numpy as np
import cv2
import matplotlib.pyplot as plt

import fpm_py as fpm
from fpm_py.evaluation import rmse

object_output = cv2.imread('datasets/hq_object.png', cv2.IMREAD_GRAYSCALE)

# Define parameters for generating synthetic low-resolution images.
wavelength = 0.525  # Wavelength of the illumination light in micrometers (or relevant units).
optical_magnification = 1.7  # Optical magnification factor of the system.
sensor_pixel_size = 1.12  # Size of each pixel on the sensor in micrometers (or relevant units).
numerical_aperture = 0.15  # Numerical aperture of the objective lens.
desired_input_shape = (50, 50)  # Shape to which the input image will be resized for simulation.

def generate(led_positions) -> float:

    print(f'Using {len(led_positions)} LEDs')

    image_series = fpm.generate_data(
        object_output,
        None,  # Assuming output phase is zero.
        wavelength,
        optical_magnification,
        sensor_pixel_size,
        numerical_aperture,
        led_positions,
        desired_input_shape,
        z_value=50000
    )

    output = fpm.reconstruct(
        image_series, 
        output_image_size=(1000, 1000), 
        max_iters=1
    ).abs().cpu().numpy()

    # normalize to betyween 0, 255
    output = (output - np.min(output)) / (np.max(output) - np.min(output)) * 255

    return output

def plot_3d_points(points: np.ndarray):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo', markersize=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    plt.show()

def show_results(results):
    results = [(res, rmse(object_output, res)) for res in results]

    # Create subplots to show the results and corresponding RMSE scores
    fig, axs = plt.subplots(1, len(results), figsize=(15, 5))

    for i, res in enumerate(results):
        output_image, rmse_value = res  # Assuming the function returns a tuple of (reconstructed_image, RMSE_value)
        axs[i].imshow(output_image, cmap='gray', vmin=0, vmax=255)
        axs[i].set_title(f"RMSE score: {rmse_value:.4f}")

    plt.show()

"""
# Generate Fermat spiral points in 3D
num_points = 100
#scaling_factor = [10, 100, 1000, 10000, 100000]
scaling_factor = [100, 10000]
z_value = 50000  # All points will be on the plane z = 5.0
led_positions = [
    generate_fermat_spiral_3d(num_points, factor, z_value) for factor in scaling_factor
]

# Plot the generated points
for led_position in led_positions:
    plot_3d_points(led_position)"""

# a grid of 2D points
from experimental.values import led_positions

results = [generate(led_position) for led_position in led_positions]

show_results(results)