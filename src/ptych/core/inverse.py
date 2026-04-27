import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ptych.core.forward import PtychographicForward
from ptych.core.object import Object
from ptych.core.pupil import Pupil


def _radius_limits(radius_fraction: Tensor | float) -> tuple[float, float]:
    radius = torch.as_tensor(radius_fraction).detach().flatten().cpu()
    if torch.any(radius <= 0):
        raise ValueError(f"radius_fraction must be positive; got {radius.tolist()}")

    return 0.8 * float(radius.min()), 1.2 * float(radius.max())


class InversePtychographyModel(nn.Module):
    def __init__(
        self,
        captures: Float[Tensor, "T B n n"],
        kx_batch: Float[Tensor, "B"],
        ky_batch: Float[Tensor, "B"],
        *,
        object_to_capture_ratio: int,
        pupil_radius_fraction_init: Tensor | float,
        pupil_num_phase_terms: int = 5,
        pupil_num_amp_terms: int = 1,
    ) -> None:
        super().__init__()
        num_tiles, _, capture_size, _ = captures.shape
        object_grid_size = capture_size * object_to_capture_ratio
        min_radius, max_radius = _radius_limits(pupil_radius_fraction_init)

        self.object_to_capture_ratio = object_to_capture_ratio
        self.object = Object(captures, object_to_capture_ratio)
        self.pupil = Pupil(
            object_grid_size,
            num_phase_terms=pupil_num_phase_terms,
            num_amp_terms=pupil_num_amp_terms,
            radius_fraction=pupil_radius_fraction_init,
            num_tiles=num_tiles,
            radius_bounds=(min_radius, max_radius),
        )
        self.image_formation = PtychographicForward(object_grid_size)
        self.register_buffer("kx", kx_batch / object_to_capture_ratio)
        self.register_buffer("ky", ky_batch / object_to_capture_ratio)

    def forward(self) -> Float[Tensor, "T B n n"]:
        object_tensor = self.object()
        pupil_tensor = self.pupil()
        predicted = self.image_formation(
            object_tensor,
            pupil_tensor,
            self.kx,
            self.ky,
        )
        num_tiles, num_captures, object_size, _ = predicted.shape
        return F.avg_pool2d(
            predicted.reshape(num_tiles * num_captures, 1, object_size, object_size),
            kernel_size=self.object_to_capture_ratio,
            stride=self.object_to_capture_ratio,
        ).reshape(
            num_tiles,
            num_captures,
            object_size // self.object_to_capture_ratio,
            object_size // self.object_to_capture_ratio,
        )
