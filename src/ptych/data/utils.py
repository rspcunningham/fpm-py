import math

import torch
from jaxtyping import Float

from ptych.data.types import Capture, LedPosition, StudyManifest

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
    r = math.sqrt(led_position.x**2 + led_position.y**2 + led_position.z**2)
    sin_x = led_position.x / r
    sin_y = led_position.y / r

    sample_pixel = camera_pixel_size / magnification

    kx_camera = sin_x * sample_pixel / wavelength
    ky_camera = sin_y * sample_pixel / wavelength

    return kx_camera, ky_camera


def prepare_captures(
    manifest: StudyManifest,
) -> tuple[list[Capture], float, Float[torch.Tensor, "B"], Float[torch.Tensor, "B"]]:
    """Filter, validate, and compute k-vectors for a manifest's captures.

    Filters out darkfield captures (no LED positions), asserts single wavelength
    and single LED per capture, then computes camera-normalized k-vectors.

    Returns:
        (valid_captures, wavelength, kx_batch, ky_batch)
    """
    # Filter out darkfield captures
    valid_captures: list[Capture] = [cap for cap in manifest.captures if cap.led_positions]

    # Assert single wavelength
    wavelengths = list({cap.wavelength for cap in valid_captures})
    assert len(wavelengths) == 1, (
        f"All captures must have the same wavelength. Found: {wavelengths}"
    )
    wavelength = wavelengths[0]

    # Assert single LED per capture
    for i, cap in enumerate(valid_captures):
        assert len(cap.led_positions) == 1, (
            f"Multi-LED captures not supported. "
            f"Capture {i} ({cap.filename}) has {len(cap.led_positions)} LEDs."
        )

    # Compute k-vectors
    k_vectors = [
        compute_k_camera(
            cap.led_positions[0],
            wavelength,
            manifest.sensor_pixel_size,
            manifest.magnification,
        )
        for cap in valid_captures
    ]
    kx_batch = torch.tensor([kx for kx, _ in k_vectors])
    ky_batch = torch.tensor([ky for _, ky in k_vectors])

    return valid_captures, wavelength, kx_batch, ky_batch
