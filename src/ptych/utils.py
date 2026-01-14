import torch
import numpy as np

def get_default_device():
   if torch.cuda.is_available():
       return torch.device("cuda")
   if torch.backends.mps.is_available() and torch.backends.mps.is_built():
       return torch.device("mps")
   return torch.device("cpu")

def compute_k_camera(
    led_positions: np.ndarray,
    wavelength: float,
    camera_pixel_size: float,
    magnification: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k-vectors normalized to camera Nyquist.

    Args:
        led_positions: (B, 3) array of (x, y, z) in meters
        wavelength: in meters
        camera_pixel_size: sensor pixel pitch in meters
        magnification: objective magnification

    Returns:
        kx_camera, ky_camera: normalized to camera grid, dimensionless
    """
    x, y, z = led_positions[:, 0], led_positions[:, 1], led_positions[:, 2]

    theta_x = np.arctan2(x, z)
    theta_y = np.arctan2(y, z)

    sample_pixel = camera_pixel_size / magnification

    kx_camera = np.sin(theta_x) * sample_pixel / wavelength
    ky_camera = np.sin(theta_y) * sample_pixel / wavelength

    return kx_camera, ky_camera
