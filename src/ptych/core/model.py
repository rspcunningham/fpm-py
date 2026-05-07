from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ptych.core.forward import FPMForwardModel
from ptych.core.object import Object
from ptych.core.pupil import Pupil


class IlluminationGains(nn.Module):
    def __init__(self, num_illuminations: int) -> None:
        super().__init__()
        self.log_gains = nn.Parameter(torch.zeros(num_illuminations))

    def forward(self) -> Float[Tensor, "illumination"]:
        return torch.exp(self.log_gains)


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class DarkfieldBackgrounds(nn.Module):
    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        illumination_kx: Float[Tensor, "illumination"],
        illumination_ky: Float[Tensor, "illumination"],
        *,
        object_to_capture_ratio: int,
        pupil_cutoff_cyc_per_px: Tensor | float,
    ) -> None:
        super().__init__()
        num_illuminations = measured_intensity_batch.shape[1]
        flat = (
            measured_intensity_batch.detach()
            .permute(1, 0, 2, 3)
            .reshape(num_illuminations, -1)
            .cpu()
        )
        kth_index = max(1, int(0.01 * flat.shape[1]))
        background = flat.kthvalue(kth_index, dim=1).values.clamp_min(1e-8)
        illumination_radius = torch.sqrt(
            (illumination_kx.detach().cpu() / object_to_capture_ratio).square()
            + (illumination_ky.detach().cpu() / object_to_capture_ratio).square()
        )
        pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().cpu()
        darkfield_mask = illumination_radius > pupil_cutoff
        background = torch.where(
            darkfield_mask,
            background,
            torch.full_like(background, 1e-8),
        )

        self.raw_backgrounds = nn.Parameter(_inverse_softplus(background))
        self.register_buffer("darkfield_mask", darkfield_mask.to(torch.float32))

    def forward(self) -> Float[Tensor, "illumination"]:
        darkfield_mask = cast(Tensor, self.darkfield_mask)
        return F.softplus(self.raw_backgrounds) * darkfield_mask


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
        predicted_intensities_full_res = (
            predicted_intensities_full_res
            * self.illumination_gains()[None, :, None, None]
        )
        patch_batch_size, num_illuminations, object_size, _ = (
            predicted_intensities_full_res.shape
        )
        return (
            F.avg_pool2d(
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
            + self.darkfield_backgrounds()[None, :, None, None]
        )
