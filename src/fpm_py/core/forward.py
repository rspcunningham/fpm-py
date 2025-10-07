from __future__ import annotations

from typing import Optional, Tuple
import torch

from ..utils.math_utils import overlap_matrices, circle_like, ft, ift, kvector_to_x_y
from .structs import ImageSeries, ImageCapture
from ..utils.data_utils import auto_place


"""
Forward Model
"""

__all__ = ["reconstruct"]



def _update_obj_pupil(
    obj_est: torch.Tensor,
    pupil: torch.Tensor,
    wave_fourier: torch.Tensor,
    wave_fourier_new: torch.Tensor,
    x: int,
    y: int,
    alpha: float = 1.0,
    beta: float = 1000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Light-weight quasi-Newton optimiser used in early FPM papers."""

    delta_wave = wave_fourier_new - wave_fourier

    if torch.all(delta_wave == 0):
        return obj_est, pupil

    # crop from obj_est once
    obj_region = obj_est[
        x : x + pupil.shape[0],
        y : y + pupil.shape[1],
    ]

    # object update
    obj_factor = (pupil.abs() ** 2).conj()
    obj_update = obj_factor * delta_wave / (pupil.abs() ** 2 + alpha)
    max_p = torch.max(pupil.abs())
    obj_update.div_(max_p)

    overlap_matrices(obj_est, obj_update, x, y)

    # pupil update
    pup_factor = (obj_region.abs() ** 2).conj()
    pup_update = pup_factor * delta_wave / (obj_region.abs() ** 2 + beta)
    max_o = torch.max(obj_region.abs())
    pup_update.div_(max_o)
    pupil.add_(pup_update)

    return obj_est, pupil



def _single_capture_update(
    obj_est: torch.Tensor,
    pupil: torch.Tensor,
    capture: ImageCapture,
    image_size: torch.Tensor,
    du: float,
    fourier_center: tuple[int, int],
    pupil_binary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process one capture and return updated (obj_est, pupil, wave_new)."""

    image = capture.image
    k_vector = capture.k_vector

    # coordinates of this crop inside the global spectrum
    obj_h, obj_w = obj_est.shape
    x, y = kvector_to_x_y(fourier_center, image_size, du, k_vector, obj_size=(obj_h, obj_w))

    # forward model
    wave_fourier = obj_est[x : x + image_size[0], y : y + image_size[1]] * pupil
    wave_spatial = ift(wave_fourier)
    image_est = wave_spatial.abs() ** 2

    # intensity correction - optimized for GPU performance using float32 ops
    # Extract phase of wave_spatial (preserve this)
    phase = torch.angle(wave_spatial)
    # Calculate amplitude correction factor (all in float32)
    amplitude = torch.abs(wave_spatial)
    amplitude_correction = torch.sqrt(torch.clamp(image, min=0.0)) / (torch.sqrt(image_est) + torch.finfo(torch.float32).eps)
    # Combine corrected amplitude with original phase
    corrected_real = amplitude_correction * amplitude * torch.cos(phase)
    corrected_imag = amplitude_correction * amplitude * torch.sin(phase)
    # Back to complex for FFT
    wave_spatial_corrected = torch.complex(corrected_real, corrected_imag)
    wave_fourier_new = ft(wave_spatial_corrected)

    # enforce binary pupil support before optimization
    pupil.mul_(pupil_binary)

    # optimiser step (in-place updates)
    obj_est, pupil = _update_obj_pupil(obj_est, pupil, wave_fourier, wave_fourier_new, x, y)
    
    # enforce binary pupil support after optimization as well
    pupil.mul_(pupil_binary)
    
    return obj_est, pupil, wave_fourier_new


def reconstruct(
    series: ImageSeries,
    output_scale_factor: int | None = None,
    output_image_size: tuple[int, int] | None = None,
    pupil_0: Optional[torch.Tensor] = None,
    max_iters: int = 10,
) -> torch.Tensor:
    """Fourier ptychographic reconstruction entry-point.
    
    Args:
        series: Image series containing captures and acquisition settings
        output_scale_factor: Optional scale factor for output size
        output_image_size: Optional explicit output dimensions (H, W)
        pupil_0: Optional initial pupil function (defaults to circle)
        iteration_terminator: Function determining when to stop iterating
        optimizer: Function to update object and pupil estimates
        check_for_nans: Whether to check for NaNs during iterations
        **kwargs: Additional arguments passed to terminator and optimizer
        
    Returns:
        torch.Tensor: Reconstructed complex-valued object in spatial domain
        
    Notes:
        Output size selection uses the following priority:
        1. User-provided output_image_size
        2. Size calculated from output_scale_factor
        3. Auto-calculated minimum size needed for reconstruction
        
        For optimal FFT processing, all output sizes are forced to even dimensions.    
    """

    # Disable autograd for entire reconstruction process
    with torch.no_grad():
        # ------------------------------------------------------------------
        #  Device placement & parameter sanity-checks
        # ------------------------------------------------------------------
        series = auto_place(series)
        device = series.device
        image_size = series.image_size.to(device)

        if pupil_0 is None:
            pupil_0 = circle_like(series.captures[0].image)
        pupil_0 = pupil_0.to(device)

        if not torch.equal(image_size, torch.tensor(pupil_0.shape, device=device)):
            raise ValueError("Pupil and image size mismatch.")

        if output_image_size is None:
            if output_scale_factor is None:
                # Compute minimum needed size for full k-space coverage
                k_span = image_size // 2
                total = k_span + series.max_k // series.settings.du
                min_size = (image_size * total // k_span).tolist()

                # Force output to be even-sized for FFT symmetry
                min_size_list = [2 * ((s + 1) // 2) for s in min_size]
                output_image_size = (min_size_list[0], min_size_list[1])
            else:
                # When using scale factor, ensure even dimensions
                scaled_size = (image_size * output_scale_factor).tolist()
                scaled_size_list = [2 * ((s + 1) // 2) for s in scaled_size]
                output_image_size = (scaled_size_list[0], scaled_size_list[1])

        # Convert to integers - handles both user-provided and computed sizes
        H_out, W_out = int(output_image_size[0]), int(output_image_size[1])

        # ------------------------------------------------------------------
        #  Allocations
        # ------------------------------------------------------------------
        obj_est = torch.zeros((H_out, W_out), dtype=torch.complex64, device=device)
        pupil = pupil_0.clone()  # do not mutate caller's tensor
        pupil_binary = pupil.abs() > 0
        fourier_center = (H_out // 2, W_out // 2)


        i = 0

        # object initialisation with first frame (same math as before)
        cap0 = series.captures[0]
        # We pass obj_size to validate crop boundaries
        x0, y0 = kvector_to_x_y(fourier_center, image_size, series.settings.du, cap0.k_vector, obj_size=(H_out, W_out))
        wave_new = ft(torch.sqrt(cap0.image))

        overlap_matrices(obj_est, wave_new * pupil, x0, y0)

        # ------------------------------------------------------------------
        #  Main loop
        # ------------------------------------------------------------------
        while i < max_iters:
            print(f"Starting iteration {i}")
            
            for cap in series.captures:
                obj_est, pupil, wave_new = _single_capture_update(
                    obj_est=obj_est,
                    pupil=pupil,
                    capture=cap,
                    image_size=image_size,
                    du=series.settings.du,
                    fourier_center=fourier_center,
                    pupil_binary=pupil_binary,
                )
            
            i += 1
            print(f"Completed iteration {i}")

            if torch.isnan(obj_est).any():
                raise RuntimeError("NaN values detected in object estimate - failed to converge")


        # Final transform to spatial domain
        return ift(obj_est)
