from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from jaxtyping import Complex, Float

# Use unitary Fourier transforms
fft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.fft2, norm="ortho"))
ifft2 = cast(Callable[..., torch.Tensor], partial(torch.fft.ifft2, norm="ortho"))


class PtychographicForward(nn.Module):
    def __init__(self, object_grid_size: int) -> None:
        super().__init__()
        coords = torch.arange(object_grid_size, dtype=torch.get_default_dtype())
        y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
        self.object_grid_size = object_grid_size
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("y_grid", y_grid)

    def forward(
        self,
        object_tensor: Complex[torch.Tensor, "T N N"],
        pupil_tensor: Complex[torch.Tensor, "T N N"],
        kx: Float[torch.Tensor, "B"],
        ky: Float[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "T B N N"]:
        """
        Return predicted full-resolution intensities for each k-space location.
        """
        _, n, _ = object_tensor.shape
        if n != self.object_grid_size:
            raise ValueError(
                f"Expected object grid size {self.object_grid_size}, got {n}"
            )

        x_grid = cast(torch.Tensor, self.x_grid)
        y_grid = cast(torch.Tensor, self.y_grid)
        kx = kx.view(-1, 1, 1)
        ky = ky.view(-1, 1, 1)

        phase = 2 * torch.pi * (kx * x_grid[None] + ky * y_grid[None])
        phase_ramps = torch.exp(1j * phase.to(object_tensor.dtype))

        tilted_objects = object_tensor[:, None] * phase_ramps[None]
        objects_fourier = fft2(tilted_objects)
        filtered_fourier = pupil_tensor[:, None] * objects_fourier
        complex_image_fields = ifft2(filtered_fourier)
        return torch.abs(complex_image_fields) ** 2
