import numpy as np
import cv2
import matplotlib.pyplot as plt

import fpm_py as fpm
from fpm_py.evaluation import sail

# Load a single-channel (grayscale) image using OpenCV.
# The image is used as the high-quality object (target) for generating synthetic data.
object_output = cv2.imread('datasets/hq_object.png', cv2.IMREAD_GRAYSCALE)

# Define parameters for generating synthetic low-resolution images.
wavelength = 0.525  # Wavelength of the illumination light in micrometers (or relevant units).
optical_magnification = 1.7  # Optical magnification factor of the system.
sensor_pixel_size = 1.12  # Size of each pixel on the sensor in micrometers (or relevant units).
numerical_aperture = 0.15  # Numerical aperture of the objective lens.
desired_input_shape = (512, 512)  # Shape to which the input image will be resized for simulation.

# Define positions of the LEDs in a 3D space (X, Y, Z coordinates).
# These positions determine the direction of illumination for each capture.
led_positions = np.array([
    [10000, 0, 50000],
    [0, 10000, 50000],
    [10000, 10000, 50000],
])

# Generate a series of low-resolution images using the specified parameters and LED positions.
# The generated images simulate how the object would appear under different illumination angles.
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

# Loop through each image in the generated image series.
# Convert the image from a tensor to a NumPy array and display it using matplotlib.
for item in image_series.image_stack:
    im = item.image.cpu().numpy().astype(np.uint16)  # Convert tensor to NumPy array and ensure correct data type.
    plt.imshow(im, cmap='gray')  # Display the image in grayscale.
    plt.show()

# Reconstruct a high-resolution image from the series of low-resolution images.
# The `output_scale_factor` determines how much the resolution is increased.
output = fpm.reconstruct(image_series, output_image_size=object_output.shape).abs().cpu().numpy()

# Display the magnitude of the reconstructed high-resolution image.
plt.imshow(output, cmap='gray')  # Convert the tensor to NumPy array and display in grayscale.
plt.show()

# Evaluate the reconstruction using the SAIL metric.
sail_value = sail(object_output, output)

print(f"SAIL value: {sail_value}")