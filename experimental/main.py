import numpy as np
import cv2
import matplotlib.pyplot as plt

import fpm_py as fpm
from fpm_py.evaluation import sail, rmse

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

object_output = cv2.imread('datasets/hq_object.png', cv2.IMREAD_GRAYSCALE)

# Define parameters for generating synthetic low-resolution images.
wavelength = 0.525  # Wavelength of the illumination light in micrometers (or relevant units).
optical_magnification = 1.7  # Optical magnification factor of the system.
sensor_pixel_size = 1.12  # Size of each pixel on the sensor in micrometers (or relevant units).
numerical_aperture = 0.15  # Numerical aperture of the objective lens.
desired_input_shape = (50, 50)  # Shape to which the input image will be resized for simulation.

def calculate_sail(led_positions) -> float:
    image_series = fpm.generate_data(
        object_output,
        None,  # Assuming output phase is zero.
        wavelength,
        optical_magnification,
        sensor_pixel_size,
        numerical_aperture,
        led_positions,
        desired_input_shape
    )

    output = fpm.reconstruct(image_series, max_iters=20).abs().cpu().numpy()

    alex_value = loss_fn_alex(object_output, output)
    sail_value = sail(object_output, output)

    return output, sail_value, alex_value

def calculate_rmse(led_positions) -> float:
    image_series = fpm.generate_data(
        object_output,
        None,  # Assuming output phase is zero.
        wavelength,
        optical_magnification,
        sensor_pixel_size,
        numerical_aperture,
        led_positions,
        desired_input_shape
    )

    output = fpm.reconstruct(image_series, max_iters=20).abs().cpu().numpy()

    rmse_value = rmse(object_output, output)

    return output, rmse_value


def generate_fermat_spiral_3d(num_points: int, scaling_factor: float, z_value: float) -> np.ndarray:
    """
    Generates points on a Fermat spiral in 3D space with a constant z-coordinate.

    Args:
        num_points (int): The number of points to generate on the spiral.
        spacing (float): The spacing between successive turns of the spiral (scaling factor).
        z_value (float): The constant z-coordinate for all points.

    Returns:
        np.ndarray: An array of (x, y, z) points representing the Fermat spiral in 3D.
    """
   # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    # Generate the points
    theta = np.arange(num_points) * golden_angle
    r = scaling_factor * np.sqrt(np.arange(num_points))
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Create a constant z-coordinate array
    z = np.full_like(x, z_value)
    
    return np.column_stack((x, y, z))

def plot_3d_points(points: np.ndarray):
    """
    Plots the points of a Fermat spiral in 3D space.

    Args:
        points (np.ndarray): An array of (x, y, z) points representing the Fermat spiral in 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo', markersize=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    plt.show()


# Generate Fermat spiral points in 3D
num_points = 100
#scaling_factor = [10, 100, 1000, 10000, 100000]
scaling_factor = [100, 10000]
z_value = 50000  # All points will be on the plane z = 5.0
led_positions = [
    generate_fermat_spiral_3d(num_points, factor, z_value) for factor in scaling_factor
]

"""led_positions = np.array([[
    [10000, 0, 50000],
    [0, 10000, 50000],
    [10000, 10000, 50000],
]])
"""
for positions in led_positions:
    plot_3d_points(positions)

results = [calculate_sail(led_position) for led_position in led_positions]

fig, axs = plt.subplots(1, len(results), figsize=(15, 5))

for i, res in enumerate(results):
    output, sail_value, alex_value = res
    axs[i].imshow(output, cmap='gray')
    axs[i].set_title(f"SAIL score: {sail_value}\nLPIPS score: {alex_value}")

plt.show()

# Apply RMSE calculation to each LED position configuration
results = [calculate_rmse(led_position) for led_position in led_positions]

# Create subplots to show the results and corresponding RMSE scores
fig, axs = plt.subplots(1, len(results), figsize=(15, 5))

for i, res in enumerate(results):
    output_image, rmse_value = res  # Assuming the function returns a tuple of (reconstructed_image, RMSE_value)
    axs[i].imshow(output_image, cmap='gray')
    axs[i].set_title(f"RMSE score: {rmse_value:.4f}")

plt.show()