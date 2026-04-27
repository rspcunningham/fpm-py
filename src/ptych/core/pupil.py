from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex
from torch import Tensor

from ptych.core.zernike import zernike_basis_tensors


def _raw_bounded_radius(
    radius_fraction: Tensor,
    min_value: float,
    max_value: float,
) -> Tensor:
    frac = (radius_fraction - min_value) / (max_value - min_value)
    frac = torch.clamp(frac, 1e-6, 1.0 - 1e-6)
    return torch.log(frac / (1.0 - frac))


def bounded_radius(raw_radius: Tensor, min_value: float, max_value: float) -> Tensor:
    return min_value + (max_value - min_value) * torch.sigmoid(raw_radius)


def _raw_softplus_value(value: Tensor) -> Tensor:
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
        num_amp_terms: int = 11,
        *,
        phase_coeffs: Tensor | None = None,
        amp_coeffs: Tensor | None = None,
        radius_fraction: Tensor | float = 0.2,
        num_tiles: int | None = None,
        edge_width_px: float = 2.0,
        radius_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        num_tiles = num_tiles or 1
        self.object_grid_size = object_grid_size
        self.num_phase_terms = num_phase_terms
        self.num_amp_terms = num_amp_terms
        self.num_tiles = num_tiles
        self.edge_width_px = edge_width_px

        rho_pixels, angular_parts, radial_coeffs, radial_powers = zernike_basis_tensors(
            object_grid_size,
            num_phase_terms,
            num_amp_terms,
        )
        self.register_buffer("rho_pixels", rho_pixels)
        self.register_buffer("angular_parts", angular_parts)
        self.register_buffer("radial_coeffs", radial_coeffs)
        self.register_buffer("radial_powers", radial_powers)

        if phase_coeffs is None:
            phase = torch.zeros(num_phase_terms)
        else:
            phase = _init_tensor(phase_coeffs)

        if amp_coeffs is None:
            amp = torch.zeros(num_amp_terms)
            amp[0] = _raw_softplus_value(torch.ones(()))
        else:
            amp = _init_tensor(amp_coeffs)

        radius = _init_tensor(radius_fraction)

        if phase.ndim == 1:
            phase = phase.reshape(1, -1).repeat(num_tiles, 1)
        elif phase.shape[0] == 1:
            phase = phase.repeat(num_tiles, 1)

        if amp.ndim == 1:
            amp = amp.reshape(1, -1).repeat(num_tiles, 1)
        elif amp.shape[0] == 1:
            amp = amp.repeat(num_tiles, 1)

        if radius.ndim == 0:
            radius = radius.reshape(1).repeat(num_tiles)
        elif radius.ndim == 1 and radius.shape[0] == 1:
            radius = radius.repeat(num_tiles)

        self.phase_coeffs = nn.Parameter(phase)
        self.amp_coeffs = nn.Parameter(amp)
        if radius_bounds is None:
            radius_values = radius.detach().flatten().cpu()
            if torch.any(radius_values <= 0):
                raise ValueError(
                    f"radius_fraction must be positive; got {radius_values.tolist()}"
                )
            self.min_radius = 0.8 * float(radius_values.min())
            self.max_radius = 1.2 * float(radius_values.max())
        else:
            self.min_radius, self.max_radius = radius_bounds
        self.raw_radius = nn.Parameter(
            _raw_bounded_radius(radius, self.min_radius, self.max_radius)
        )

    @property
    def radius_fraction(self) -> Tensor:
        return bounded_radius(self.raw_radius, self.min_radius, self.max_radius)

    def forward(self) -> Complex[Tensor, "T N N"]:
        radius_fraction = self.radius_fraction
        rho_pixels = cast(Tensor, self.rho_pixels)
        angular_parts = cast(Tensor, self.angular_parts)
        radial_coeffs = cast(Tensor, self.radial_coeffs)
        radial_powers = cast(Tensor, self.radial_powers)

        rho_norm = rho_pixels[None] / (
            radius_fraction[:, None, None] * self.object_grid_size
        )
        rho = torch.clamp(rho_norm, max=1.0)

        num_phase = self.phase_coeffs.shape[-1]
        num_amp = self.amp_coeffs.shape[-1]
        max_terms = max(num_phase, num_amp)
        rho_powers = (
            rho[:, None, None] ** radial_powers[None, :max_terms, :, None, None]
        )
        radial = (radial_coeffs[None, :max_terms, :, None, None] * rho_powers).sum(
            dim=2
        )
        terms = radial * angular_parts[None, :max_terms]

        phase = torch.einsum("ti,tihw->thw", self.phase_coeffs, terms[:, :num_phase])
        amp_raw = torch.einsum("ti,tihw->thw", self.amp_coeffs, terms[:, :num_amp])
        amplitude = F.softplus(amp_raw)
        pupil = amplitude * torch.exp(1j * phase)

        aperture = torch.sigmoid(
            (radius_fraction[:, None, None] * self.object_grid_size - rho_pixels[None])
            / self.edge_width_px
        )
        return pupil * aperture


def radius_fraction_from_optics(
    numerical_aperture: float,
    wavelength_m: float,
    sensor_pixel_size_m: float,
    magnification: float,
    object_to_capture_ratio: int,
) -> float:
    dx_obj = sensor_pixel_size_m / (magnification * object_to_capture_ratio)
    return (numerical_aperture / wavelength_m) * dx_obj
