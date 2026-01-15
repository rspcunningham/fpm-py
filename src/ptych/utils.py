import math
import torch

from ptych.data.types import LedPosition

def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def compute_k_camera(
    led_position: LedPosition,
    wavelength: float,
    camera_pixel_size: float,
    magnification: float,
) -> tuple[float, float]:
    """
    Compute k-vector normalized to camera Nyquist for a single LED position.

    Args:
        led_position: LED position (x, y, z) in meters
        wavelength: in meters
        camera_pixel_size: sensor pixel pitch in meters
        magnification: objective magnification

    Returns:
        (kx_camera, ky_camera): normalized to camera grid, dimensionless
    """
    theta_x = math.atan2(led_position.x, led_position.z)
    theta_y = math.atan2(led_position.y, led_position.z)

    sample_pixel = camera_pixel_size / magnification

    kx_camera = math.sin(theta_x) * sample_pixel / wavelength
    ky_camera = math.sin(theta_y) * sample_pixel / wavelength

    return kx_camera, ky_camera
