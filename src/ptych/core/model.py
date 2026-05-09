from typing import cast

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ptych.core.darkfield import DarkfieldBackgrounds, DarkfieldScatter
from ptych.core.forward import FPMForwardModel
from ptych.core.intensity_downsample import intensity_downsample
from ptych.core.object import Object
from ptych.core.pupil import Pupil


class IlluminationGains(nn.Module):
    def __init__(self, num_illuminations: int) -> None:
        super().__init__()
        self.log_gains = nn.Parameter(torch.zeros(num_illuminations))

    def forward(self) -> Float[Tensor, "illumination"]:
        return torch.exp(self.log_gains)


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
        patch_batch_size, num_illuminations, capture_height, _ = (
            measured_intensity_batch.shape
        )
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
        self.illumination_gains = IlluminationGains(num_illuminations)
        self.darkfield_backgrounds = DarkfieldBackgrounds(
            measured_intensity_batch,
            illumination_kx,
            illumination_ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
        )
        self.darkfield_scatter = DarkfieldScatter(
            measured_intensity_batch,
            illumination_kx,
            illumination_ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px_init,
        )
        illumination_kx = illumination_kx / object_to_capture_ratio
        illumination_ky = illumination_ky / object_to_capture_ratio
        self.forward_model = FPMForwardModel(
            object_grid_size,
            illumination_kx,
            illumination_ky,
        )
        self.register_buffer("illumination_kx", illumination_kx)
        self.register_buffer("illumination_ky", illumination_ky)

    def forward(
        self,
        illumination_slice: slice | None = None,
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        if illumination_slice is None:
            illumination_slice = slice(None)

        object_tensor = self.object()
        phase_ramps = cast(Tensor, self.forward_model.cached_phase_ramps)[
            illumination_slice
        ]
        complex_image_fields = self.forward_model.forward_with_phase_ramps(
            object_tensor,
            self.pupil(),
            phase_ramps,
        )
        predicted_low_res = intensity_downsample(
            complex_image_fields,
            self.object_to_capture_ratio,
        )
        predicted_low_res = (
            predicted_low_res
            * self.illumination_gains()[illumination_slice][None, :, None, None]
        )
        return (
            predicted_low_res
            + self.darkfield_backgrounds.incoherent_intensity()[illumination_slice][
                None, :, None, None
            ]
            + self.darkfield_scatter.incoherent_intensity(illumination_slice)
        )
