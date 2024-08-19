import numpy as np
import cv2
import fpm_py as fpm
import torch

from matplotlib import pyplot as plt

def k_vect_from_um(wavelength: float, x: float, y: float, z: float) -> np.ndarray:
    """Expects x, y, z positions in micrometers and returns the corresponding k-vector in the Fourier domain."""

    k = 1 / wavelength # wavenumber

    denom = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    k_x = k * x / denom
    k_y = k * y / denom

    return np.array([k_x, k_y])

def generate_data(
        object_output: np.ndarray,
        phase_output: np.ndarray | None,
        wavelength: float,
        optical_magnification: float,
        sensor_pixel_size: float,
        numerical_aperture: float,
        led_positions: np.ndarray,
        desired_input_shape: tuple
) -> fpm.ImageSeries:
    
    object_pixel_size = sensor_pixel_size / optical_magnification
    max_spatial_frequency = numerical_aperture / wavelength
    num_leds = led_positions.shape[0]

    if phase_output is None:
        phase_output = np.zeros_like(object_output)
    
    synthetic_object = object_output * np.exp(1j * phase_output)

    led_positions_x = led_positions[:, 0]
    led_positions_y = led_positions[:, 1]
    led_positions_z = led_positions[:, 2]

    led_distances = np.sqrt(led_positions_x ** 2 + led_positions_y ** 2 + led_positions_z ** 2)
    sin_theta_x = led_positions_x / led_distances
    sin_theta_y = led_positions_y / led_distances

    illumination_na = np.sqrt(sin_theta_x ** 2 + sin_theta_y ** 2)

    max_illumination_spatial_frequency = np.max(illumination_na) / wavelength + max_spatial_frequency

    downsample_factor = 1
    downsampled_width = synthetic_object.shape[1] + 1

    while downsampled_width > synthetic_object.shape[1]:
        downsample_factor += 1
        downsampled_width = int(np.round(synthetic_object.shape[1] / downsample_factor / 2) * 2)
        field_of_view_x = downsampled_width * object_pixel_size
        frequency_sampling_x = 1 / field_of_view_x
        downsampled_width = int(np.round(2 * max_illumination_spatial_frequency / frequency_sampling_x) * 2)

    downsampled_height = int(np.round(synthetic_object.shape[0] / downsample_factor / 2) * 2)
    downsampled_width = int(np.round(synthetic_object.shape[1] / downsample_factor / 2) * 2)

    field_of_view_x = downsampled_width * object_pixel_size
    field_of_view_y = downsampled_height * object_pixel_size

    frequency_sampling_x = 1 / field_of_view_x if downsampled_width % 2 == 0 else 1 / object_pixel_size / (desired_input_shape[1] - 1)
    frequency_sampling_y = 1 / field_of_view_y if downsampled_height % 2 == 0 else 1 / object_pixel_size / (desired_input_shape[0] - 1)

    mesh_x, mesh_y = np.meshgrid(
        np.arange(downsampled_width) - np.round(downsampled_width / 2),
        np.arange(downsampled_height) - np.round(downsampled_height / 2)
    )

    if downsampled_width > downsampled_height:
        mesh_y = mesh_y * np.max(mesh_x) / np.max(mesh_y)
    else:
        mesh_x = mesh_x * np.max(mesh_y) / np.max(mesh_x)

    # Calculate radial distance in the Fourier domain
    radial_distance = np.sqrt(mesh_x**2 + mesh_y**2)

    # Compute the cutoff spatial frequency index for the pupil function
    cutoff_spatial_frequency_index = max_spatial_frequency / np.min([frequency_sampling_x, frequency_sampling_y])

    # Generate the pupil function (a binary mask in the Fourier domain)
    pupil_function = (radial_distance < cutoff_spatial_frequency_index).astype("float64")

    # Calculate corresponding spatial frequencies for each LED
    led_spatial_frequencies_x = sin_theta_x / wavelength
    led_spatial_frequencies_y = sin_theta_y / wavelength

    # Calculate spatial frequency indices for each LED relative to the center
    led_frequency_indices_y = np.round(led_spatial_frequencies_y / frequency_sampling_y)
    led_frequency_indices_x = np.round(led_spatial_frequencies_x / frequency_sampling_x)

    # Compute the Fourier transform of the synthetic object
    object_fourier_transform = np.fft.fftshift(np.fft.fft2(synthetic_object))

    capture_list = []

    # Loop over each LED to simulate images
    for led_index in range(num_leds):
        # Calculate the Fourier domain coordinates for the current LED
        fourier_center_x = int(led_frequency_indices_x[led_index] + np.ceil(synthetic_object.shape[1] / 2))
        fourier_center_y = int(led_frequency_indices_y[led_index] + np.ceil(synthetic_object.shape[0] / 2))

        k_vect = [led_spatial_frequencies_x[led_index], led_spatial_frequencies_y[led_index]]


        y_low = int(np.round(fourier_center_y - downsampled_height / 2))
        y_high = int(np.round(fourier_center_y + downsampled_height / 2))
        x_low = int(np.round(fourier_center_x - downsampled_width / 2))
        x_high = int(np.round(fourier_center_x + downsampled_width / 2))

        # Crop the Fourier transform and apply the pupil function
        cropped_fourier_transform = ((1 / downsample_factor) ** 2) * object_fourier_transform[y_low:y_high, x_low:x_high] * pupil_function

        # Perform inverse Fourier transform to obtain the low-resolution image
        image = np.abs(np.fft.ifft2(np.fft.ifftshift(cropped_fourier_transform))) ** 2
        image = np.round(image).astype("uint16")

        capture = fpm.ImageCapture(torch.Tensor(image), torch.Tensor(k_vect))
        capture_list.append(capture)
    
    image_series = fpm.ImageSeries(capture_list, wavelength=wavelength, numerical_aperture=numerical_aperture, optical_magnification=optical_magnification, sensor_pixel_size=sensor_pixel_size)
    return image_series

if __name__ == '__main__':
    object_output = cv2.imread('fpm_py/experimental/object.tiff', cv2.IMREAD_GRAYSCALE)
    phase_output = cv2.imread('fpm_py/experimental/phase.tiff', cv2.IMREAD_GRAYSCALE)
    wavelength = 0.525
    optical_magnification = 1.7
    sensor_pixel_size = 1.12
    numerical_aperture = 0.15
    led_to_object_distance = 60500
    led_positions = np.array([
        [10000, 0],
        [0, 10000],
        [10000, 10000],
    ])
    desired_input_shape = (512, 512)

    image_series = generate_data(
        object_output,
        phase_output,
        wavelength,
        optical_magnification,
        sensor_pixel_size,
        numerical_aperture,
        led_to_object_distance,
        led_positions,
        desired_input_shape
    )

    for item in image_series.image_stack:
        im = item.image.cpu().numpy().astype(np.uint16)
        plt.imshow(im, cmap='gray')
        plt.show()

    output = fpm.reconstruct(image_series, output_scale_factor=4)

    plt.imshow(output.abs().cpu().numpy(), cmap='gray')
    plt.show()
