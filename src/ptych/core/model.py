from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ptych.core.forward import FPMForwardModel
from ptych.core.object import Object, SigmoidObject
from ptych.core.pupil import Pupil

_DARKFIELD_SCATTER_RANK = 2
_OBJECT_CLASS: type[Object] | type[SigmoidObject] = SigmoidObject


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

    def incoherent_intensity(self) -> Float[Tensor, "illumination"]:
        darkfield_mask = cast(Tensor, self.darkfield_mask)
        return F.softplus(self.raw_backgrounds) * darkfield_mask


class DarkfieldScatter(nn.Module):
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
        _, num_illuminations, height, width = measured_intensity_batch.shape
        illumination_radius = torch.sqrt(
            (illumination_kx.detach().cpu() / object_to_capture_ratio).square()
            + (illumination_ky.detach().cpu() / object_to_capture_ratio).square()
        )
        pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().cpu()
        darkfield_mask = illumination_radius > pupil_cutoff

        if darkfield_mask.any():
            seed = (
                measured_intensity_batch.detach()
                .cpu()[:, darkfield_mask]
                .mean(dim=1)
                .clamp_min(1e-8)
            )
        else:
            seed = torch.full(
                (measured_intensity_batch.shape[0], height, width),
                1e-8,
                dtype=measured_intensity_batch.dtype,
            )
        seed = 0.02 * seed[:, None] / _DARKFIELD_SCATTER_RANK
        seed = seed.expand(-1, _DARKFIELD_SCATTER_RANK, -1, -1).clone()
        coefficients = torch.ones(num_illuminations, _DARKFIELD_SCATTER_RANK)

        self.raw_basis = nn.Parameter(_inverse_softplus(seed))
        self.raw_coefficients = nn.Parameter(_inverse_softplus(coefficients))
        self.register_buffer("darkfield_mask", darkfield_mask.to(torch.float32))

    def incoherent_intensity(
        self,
        illumination_slice: slice,
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        basis = F.softplus(self.raw_basis)
        coefficients = F.softplus(self.raw_coefficients)[illumination_slice]
        darkfield_mask = cast(Tensor, self.darkfield_mask)[illumination_slice]
        coefficients = coefficients * darkfield_mask[:, None]
        return torch.einsum("brhw,ir->bihw", basis, coefficients)


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
        self.object = _OBJECT_CLASS(measured_intensity_batch, object_to_capture_ratio)
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
        self.forward_model = FPMForwardModel(object_grid_size)
        self.register_buffer(
            "illumination_kx", illumination_kx / object_to_capture_ratio
        )
        self.register_buffer(
            "illumination_ky", illumination_ky / object_to_capture_ratio
        )

    def forward(
        self,
        illumination_slice: slice | None = None,
    ) -> Float[Tensor, "patch_batch illumination height width"]:
        if illumination_slice is None:
            illumination_slice = slice(None)

        object_tensor = self.object()
        illumination_kx = cast(Tensor, self.illumination_kx)[illumination_slice]
        illumination_ky = cast(Tensor, self.illumination_ky)[illumination_slice]
        complex_image_fields = self.forward_model(
            object_tensor,
            self.pupil(),
            illumination_kx,
            illumination_ky,
        )
        complex_image_fields = complex_image_fields * torch.sqrt(
            self.illumination_gains()[illumination_slice][None, :, None, None]
        )
        predicted_intensities_full_res = complex_image_fields.abs().square()
        patch_batch_size, num_illuminations, object_size, _ = complex_image_fields.shape
        predicted_low_res = F.avg_pool2d(
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
        return (
            predicted_low_res
            + self.darkfield_backgrounds.incoherent_intensity()[illumination_slice][
                None, :, None, None
            ]
            + self.darkfield_scatter.incoherent_intensity(illumination_slice)
        )
