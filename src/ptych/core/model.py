import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ptych.core.object import Object
from ptych.core.forward import FPMForwardModel
from ptych.core.pupil import Pupil


def _pupil_cutoff_limits(
    pupil_cutoff_cyc_per_px: Tensor | float,
) -> tuple[float, float]:
    pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().flatten().cpu()
    if torch.any(pupil_cutoff <= 0):
        raise ValueError(
            f"pupil_cutoff_cyc_per_px must be positive; got {pupil_cutoff.tolist()}"
        )

    return 0.8 * float(pupil_cutoff.min()), 1.2 * float(pupil_cutoff.max())


class PtychographyModel(nn.Module):
    def __init__(
        self,
        measured_intensities: Float[Tensor, "T B n n"],
        kx_batch: Float[Tensor, "B"],
        ky_batch: Float[Tensor, "B"],
        *,
        object_to_capture_ratio: int,
        pupil_cutoff_cyc_per_px_init: Tensor | float,
        pupil_num_phase_terms: int = 5,
        pupil_num_amplitude_terms: int = 1,
    ) -> None:
        super().__init__()
        num_tiles, _, capture_grid_size, _ = measured_intensities.shape
        object_grid_size = capture_grid_size * object_to_capture_ratio
        min_pupil_cutoff, max_pupil_cutoff = _pupil_cutoff_limits(
            pupil_cutoff_cyc_per_px_init
        )

        self.object_to_capture_ratio = object_to_capture_ratio
        self.object = Object(measured_intensities, object_to_capture_ratio)
        self.pupil = Pupil(
            object_grid_size,
            num_phase_terms=pupil_num_phase_terms,
            num_amplitude_terms=pupil_num_amplitude_terms,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
            num_tiles=num_tiles,
            pupil_cutoff_bounds=(min_pupil_cutoff, max_pupil_cutoff),
        )
        self.forward_model = FPMForwardModel(object_grid_size)
        self.register_buffer("kx", kx_batch / object_to_capture_ratio)
        self.register_buffer("ky", ky_batch / object_to_capture_ratio)

    def forward(self) -> Float[Tensor, "T B n n"]:
        object_tensor = self.object()
        pupil_tensor = self.pupil()
        predicted_intensities_full_res = self.forward_model(
            object_tensor,
            pupil_tensor,
            self.kx,
            self.ky,
        )
        num_tiles, num_illuminations, object_size, _ = (
            predicted_intensities_full_res.shape
        )
        return F.avg_pool2d(
            predicted_intensities_full_res.reshape(
                num_tiles * num_illuminations,
                1,
                object_size,
                object_size,
            ),
            kernel_size=self.object_to_capture_ratio,
            stride=self.object_to_capture_ratio,
        ).reshape(
            num_tiles,
            num_illuminations,
            object_size // self.object_to_capture_ratio,
            object_size // self.object_to_capture_ratio,
        )
