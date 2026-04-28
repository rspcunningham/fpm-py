import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ptych.core.forward import FPMForwardModel
from ptych.core.object import Object
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
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        illumination_kx: Float[Tensor, "illumination"],
        illumination_ky: Float[Tensor, "illumination"],
        *,
        object_to_capture_ratio: int,
        pupil_cutoff_cyc_per_px_init: Tensor | float,
        pupil_phase_radial_order: int = 2,
        pupil_amplitude_radial_order: int = 0,
    ) -> None:
        super().__init__()
        patch_batch_size, _, capture_height, _ = measured_intensity_batch.shape
        object_grid_size = capture_height * object_to_capture_ratio
        min_pupil_cutoff, max_pupil_cutoff = _pupil_cutoff_limits(
            pupil_cutoff_cyc_per_px_init
        )

        self.object_to_capture_ratio = object_to_capture_ratio
        self.object = Object(measured_intensity_batch, object_to_capture_ratio)
        self.pupil = Pupil(
            object_grid_size,
            phase_radial_order=pupil_phase_radial_order,
            amplitude_radial_order=pupil_amplitude_radial_order,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
            patch_batch_size=patch_batch_size,
            pupil_cutoff_bounds=(min_pupil_cutoff, max_pupil_cutoff),
        )
        self.forward_model = FPMForwardModel(object_grid_size)
        self.register_buffer(
            "illumination_kx", illumination_kx / object_to_capture_ratio
        )
        self.register_buffer(
            "illumination_ky", illumination_ky / object_to_capture_ratio
        )

    def forward(self) -> Float[Tensor, "patch_batch illumination height width"]:
        object_tensor = self.object()
        pupil_tensor = self.pupil()
        predicted_intensities_full_res = self.forward_model(
            object_tensor,
            pupil_tensor,
            self.illumination_kx,
            self.illumination_ky,
        )
        patch_batch_size, num_illuminations, object_size, _ = (
            predicted_intensities_full_res.shape
        )
        return F.avg_pool2d(
            predicted_intensities_full_res.reshape(
                patch_batch_size * num_illuminations,
                1,
                object_size,
                object_size,
            ),
            kernel_size=self.object_to_capture_ratio,
            stride=self.object_to_capture_ratio,
        ).reshape(
            patch_batch_size,
            num_illuminations,
            object_size // self.object_to_capture_ratio,
            object_size // self.object_to_capture_ratio,
        )
