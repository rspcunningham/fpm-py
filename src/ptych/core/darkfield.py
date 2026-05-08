from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

_DARKFIELD_SCATTER_RANK = 2
_DARKFIELD_BACKGROUND_QUANTILE = 0.01


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _darkfield_mask(
    illumination_kx: Float[Tensor, "illumination"],
    illumination_ky: Float[Tensor, "illumination"],
    *,
    object_to_capture_ratio: int,
    pupil_cutoff_cyc_per_px: Tensor | float,
) -> Tensor:
    illumination_radius = torch.sqrt(
        (illumination_kx.detach().cpu() / object_to_capture_ratio).square()
        + (illumination_ky.detach().cpu() / object_to_capture_ratio).square()
    )
    pupil_cutoff = torch.as_tensor(pupil_cutoff_cyc_per_px).detach().cpu()
    return illumination_radius > pupil_cutoff


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
        kth_index = max(1, int(_DARKFIELD_BACKGROUND_QUANTILE * flat.shape[1]))
        background = flat.kthvalue(kth_index, dim=1).values.clamp_min(1e-8)
        darkfield_mask = _darkfield_mask(
            illumination_kx,
            illumination_ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px,
        )
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
        darkfield_mask = _darkfield_mask(
            illumination_kx,
            illumination_ky,
            object_to_capture_ratio=object_to_capture_ratio,
            pupil_cutoff_cyc_per_px=pupil_cutoff_cyc_per_px,
        )

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
