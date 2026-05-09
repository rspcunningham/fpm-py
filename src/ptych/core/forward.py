from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from jaxtyping import Complex, Float

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))


class FPMForwardModel(nn.Module):
    def __init__(
        self,
        object_grid_size: int,
        illumination_kx: Float[torch.Tensor, "illumination"] | None = None,
        illumination_ky: Float[torch.Tensor, "illumination"] | None = None,
    ) -> None:
        super().__init__()
        if (illumination_kx is None) != (illumination_ky is None):
            raise ValueError(
                "illumination_kx and illumination_ky must be provided together"
            )

        device = None
        if illumination_kx is not None:
            device = illumination_kx.device
        coords = torch.arange(
            object_grid_size,
            dtype=torch.get_default_dtype(),
            device=device,
        )
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        self.object_grid_size = object_grid_size
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("y_grid", y_grid)
        cached_phase_ramps = None
        if illumination_kx is not None:
            cached_phase_ramps = self.make_phase_ramps(
                illumination_kx,
                cast(torch.Tensor, illumination_ky),
            )
        self.register_buffer(
            "cached_phase_ramps",
            cached_phase_ramps,
            persistent=False,
        )

    def make_phase_ramps(
        self,
        illumination_kx: Float[torch.Tensor, "illumination"],
        illumination_ky: Float[torch.Tensor, "illumination"],
    ) -> Complex[torch.Tensor, "illumination object_height object_width"]:
        x_grid = cast(torch.Tensor, self.x_grid)
        y_grid = cast(torch.Tensor, self.y_grid)
        illumination_kx = illumination_kx.to(device=x_grid.device).view(-1, 1, 1)
        illumination_ky = illumination_ky.to(device=y_grid.device).view(-1, 1, 1)
        complex_dtype = (
            torch.complex128
            if torch.get_default_dtype() == torch.float64
            else torch.complex64
        )
        phase = (
            2
            * torch.pi
            * (illumination_kx * x_grid[None] + illumination_ky * y_grid[None])
        )
        return torch.exp(1j * phase.to(complex_dtype))

    def forward(
        self,
        object_tensor: Complex[torch.Tensor, "patch_batch object_height object_width"],
        pupil_tensor: Complex[torch.Tensor, "patch_batch object_height object_width"],
        illumination_kx: Float[torch.Tensor, "illumination"],
        illumination_ky: Float[torch.Tensor, "illumination"],
    ) -> Complex[torch.Tensor, "patch_batch illumination object_height object_width"]:
        """
        Return predicted full-resolution complex fields for each k-space location.
        """
        _, object_height, _ = object_tensor.shape
        if object_height != self.object_grid_size:
            raise ValueError(
                f"Expected object grid size {self.object_grid_size}, got {object_height}"
            )

        return self.forward_with_phase_ramps(
            object_tensor,
            pupil_tensor,
            self.make_phase_ramps(illumination_kx, illumination_ky),
        )

    def forward_with_phase_ramps(
        self,
        object_tensor: Complex[torch.Tensor, "patch_batch object_height object_width"],
        pupil_tensor: Complex[torch.Tensor, "patch_batch object_height object_width"],
        phase_ramps: Complex[torch.Tensor, "illumination object_height object_width"],
    ) -> Complex[torch.Tensor, "patch_batch illumination object_height object_width"]:
        _, object_height, _ = object_tensor.shape
        if object_height != self.object_grid_size:
            raise ValueError(
                f"Expected object grid size {self.object_grid_size}, got {object_height}"
            )

        phase_ramps = phase_ramps.to(
            device=object_tensor.device,
            dtype=object_tensor.dtype,
        )
        tilted_objects = object_tensor[:, None] * phase_ramps[None]
        objects_fourier = fft2(tilted_objects)
        filtered_fourier = pupil_tensor[:, None] * objects_fourier
        return ifft2(filtered_fourier)
