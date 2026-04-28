from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex
from torch import Tensor

from ptych.core.zernike import zernike_basis_tensors


def _radius_logit(
    pupil_cutoff_cyc_per_px: Tensor,
    min_value: float,
    max_value: float,
) -> Tensor:
    frac = (pupil_cutoff_cyc_per_px - min_value) / (max_value - min_value)
    frac = torch.clamp(frac, 1e-6, 1.0 - 1e-6)
    return torch.log(frac / (1.0 - frac))


def _bounded_radius_from_logit(
    radius_logit: Tensor, min_value: float, max_value: float
) -> Tensor:
    return min_value + (max_value - min_value) * torch.sigmoid(radius_logit)


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def _init_tensor(value: Tensor | float) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    return torch.tensor(value)


class Pupil(nn.Module):
    def __init__(
        self,
        object_grid_size: int,
        num_phase_terms: int = 21,
        num_amplitude_terms: int = 11,
        *,
        phase_coeffs: Tensor | None = None,
        raw_amplitude_coeffs: Tensor | None = None,
        pupil_cutoff_cyc_per_px: Tensor | float = 0.2,
        patch_batch_size: int | None = None,
        edge_width_px: float = 2.0,
        pupil_cutoff_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        patch_batch_size = patch_batch_size or 1
        self.object_grid_size = object_grid_size
        self.num_phase_terms = num_phase_terms
        self.num_amplitude_terms = num_amplitude_terms
        self.patch_batch_size = patch_batch_size
        self.edge_width_px = edge_width_px

        rho_pixels, angular_parts, radial_coeffs, radial_powers = zernike_basis_tensors(
            object_grid_size,
            num_phase_terms,
            num_amplitude_terms,
        )
        self.register_buffer("rho_pixels", rho_pixels)
        self.register_buffer("angular_parts", angular_parts)
        self.register_buffer("radial_coeffs", radial_coeffs)
        self.register_buffer("radial_powers", radial_powers)

        if phase_coeffs is None:
            phase = torch.zeros(num_phase_terms)
        else:
            phase = _init_tensor(phase_coeffs)

        if raw_amplitude_coeffs is None:
            raw_amplitude = torch.zeros(num_amplitude_terms)
            raw_amplitude[0] = _inverse_softplus(torch.ones(()))
        else:
            raw_amplitude = _init_tensor(raw_amplitude_coeffs)

        pupil_cutoff = _init_tensor(pupil_cutoff_cyc_per_px)

        if phase.ndim == 1:
            phase = phase.reshape(1, -1).repeat(patch_batch_size, 1)
        elif phase.shape[0] == 1:
            phase = phase.repeat(patch_batch_size, 1)

        if raw_amplitude.ndim == 1:
            raw_amplitude = raw_amplitude.reshape(1, -1).repeat(patch_batch_size, 1)
        elif raw_amplitude.shape[0] == 1:
            raw_amplitude = raw_amplitude.repeat(patch_batch_size, 1)

        if pupil_cutoff.ndim == 0:
            pupil_cutoff = pupil_cutoff.reshape(1).repeat(patch_batch_size)
        elif pupil_cutoff.ndim == 1 and pupil_cutoff.shape[0] == 1:
            pupil_cutoff = pupil_cutoff.repeat(patch_batch_size)

        self.phase_coeffs = nn.Parameter(phase)
        self.raw_amplitude_coeffs = nn.Parameter(raw_amplitude)
        if pupil_cutoff_bounds is None:
            pupil_cutoff_values = pupil_cutoff.detach().flatten().cpu()
            if torch.any(pupil_cutoff_values <= 0):
                raise ValueError(
                    "pupil_cutoff_cyc_per_px must be positive; "
                    f"got {pupil_cutoff_values.tolist()}"
                )
            self.min_pupil_cutoff_cyc_per_px = 0.8 * float(pupil_cutoff_values.min())
            self.max_pupil_cutoff_cyc_per_px = 1.2 * float(pupil_cutoff_values.max())
        else:
            self.min_pupil_cutoff_cyc_per_px, self.max_pupil_cutoff_cyc_per_px = (
                pupil_cutoff_bounds
            )
        self.pupil_cutoff_logit = nn.Parameter(
            _radius_logit(
                pupil_cutoff,
                self.min_pupil_cutoff_cyc_per_px,
                self.max_pupil_cutoff_cyc_per_px,
            )
        )

    @property
    def pupil_cutoff_cyc_per_px(self) -> Tensor:
        return _bounded_radius_from_logit(
            self.pupil_cutoff_logit,
            self.min_pupil_cutoff_cyc_per_px,
            self.max_pupil_cutoff_cyc_per_px,
        )

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        pupil_cutoff_cyc_per_px = self.pupil_cutoff_cyc_per_px
        rho_pixels = cast(Tensor, self.rho_pixels)
        angular_parts = cast(Tensor, self.angular_parts)
        radial_coeffs = cast(Tensor, self.radial_coeffs)
        radial_powers = cast(Tensor, self.radial_powers)

        rho_norm = rho_pixels[None] / (
            pupil_cutoff_cyc_per_px[:, None, None] * self.object_grid_size
        )
        rho = torch.clamp(rho_norm, max=1.0)

        num_phase = self.phase_coeffs.shape[-1]
        num_amplitude = self.raw_amplitude_coeffs.shape[-1]
        max_terms = max(num_phase, num_amplitude)
        rho_powers = (
            rho[:, None, None] ** radial_powers[None, :max_terms, :, None, None]
        )
        radial = (radial_coeffs[None, :max_terms, :, None, None] * rho_powers).sum(
            dim=2
        )
        terms = radial * angular_parts[None, :max_terms]

        phase = torch.einsum("ti,tihw->thw", self.phase_coeffs, terms[:, :num_phase])
        amplitude_logits = torch.einsum(
            "ti,tihw->thw",
            self.raw_amplitude_coeffs,
            terms[:, :num_amplitude],
        )
        amplitude = F.softplus(amplitude_logits)
        pupil = amplitude * torch.exp(1j * phase)

        aperture = torch.sigmoid(
            (
                pupil_cutoff_cyc_per_px[:, None, None] * self.object_grid_size
                - rho_pixels[None]
            )
            / self.edge_width_px
        )
        return pupil * aperture


def pupil_cutoff_cyc_per_px_from_optics(
    numerical_aperture: float,
    wavelength_m: float,
    sensor_pixel_size_m: float,
    magnification: float,
    object_to_capture_ratio: int,
) -> float:
    object_pixel_size_m = sensor_pixel_size_m / (
        magnification * object_to_capture_ratio
    )
    return (numerical_aperture / wavelength_m) * object_pixel_size_m
