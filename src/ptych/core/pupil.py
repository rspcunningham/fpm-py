import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


def _noll_to_nm(j: int) -> tuple[int, int]:
    j_current = 1
    n = 0
    while True:
        for m in range(-n, n + 1, 2):
            if j_current == j:
                if m != 0:
                    m = abs(m) if j % 2 == 0 else -abs(m)
                return n, m
            j_current += 1
        n += 1
        if n > 100:
            raise ValueError(f"Noll index j={j} too large")


def _radial_terms(n: int, m_abs: int) -> tuple[list[float], list[int]]:
    coeffs: list[float] = []
    powers: list[int] = []
    for k in range((n - m_abs) // 2 + 1):
        numerator = math.factorial(n - k)
        denominator = (
            math.factorial(k)
            * math.factorial((n + m_abs) // 2 - k)
            * math.factorial((n - m_abs) // 2 - k)
        )
        coeffs.append((1 if k % 2 == 0 else -1) * numerator / denominator)
        powers.append(n - 2 * k)
    return coeffs, powers


def _basis_tensors(
    object_grid_size: int,
    num_phase_terms: int,
    num_amp_terms: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    coords = torch.arange(object_grid_size, device=device, dtype=dtype)
    coords = torch.where(
        coords >= object_grid_size / 2, coords - object_grid_size, coords
    )
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    rho_pixels = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(grid_y, grid_x)

    max_terms = max(num_phase_terms, num_amp_terms)
    angular_parts: list[Tensor] = []
    radial_coeffs: list[list[float]] = []
    radial_powers: list[list[int]] = []
    for j in range(1, max_terms + 1):
        n, m = _noll_to_nm(j)
        if m > 0:
            angular_parts.append(torch.cos(m * theta))
        elif m < 0:
            angular_parts.append(torch.sin(abs(m) * theta))
        else:
            angular_parts.append(torch.ones_like(theta))

        coeffs, powers = _radial_terms(n, abs(m))
        radial_coeffs.append(coeffs)
        radial_powers.append(powers)

    max_k = max(len(coeffs) for coeffs in radial_coeffs)
    coeff_tensor = torch.zeros(max_terms, max_k, device=device, dtype=dtype)
    power_tensor = torch.zeros(max_terms, max_k, device=device, dtype=dtype)
    for i, (coeffs, powers) in enumerate(zip(radial_coeffs, radial_powers)):
        coeff_tensor[i, : len(coeffs)] = torch.tensor(
            coeffs, device=device, dtype=dtype
        )
        power_tensor[i, : len(powers)] = torch.tensor(
            powers, device=device, dtype=dtype
        )

    return rho_pixels, torch.stack(angular_parts), coeff_tensor, power_tensor


def _default_phase_coeffs(
    num_terms: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    return torch.zeros(num_terms, device=device, dtype=dtype)


def _default_amp_coeffs(
    num_terms: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    coeffs = torch.zeros(num_terms, device=device, dtype=dtype)
    coeffs[0] = 1.0
    return coeffs


def _as_tensor(
    value: Tensor | float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _tile_params(value: Tensor, num_tiles: int | None) -> Tensor:
    if num_tiles is None:
        return value
    if value.ndim == 0:
        return value.reshape(1).repeat(num_tiles)
    if value.ndim == 1:
        return value.reshape(1, -1).repeat(num_tiles, 1)
    if value.shape[0] == num_tiles:
        return value
    raise ValueError(f"Expected first dimension {num_tiles}, got {tuple(value.shape)}")


def _tile_radius(value: Tensor, num_tiles: int | None) -> Tensor:
    if num_tiles is None:
        return value
    if value.ndim == 0:
        return value.reshape(1).repeat(num_tiles)
    if value.ndim == 1 and value.shape[0] == num_tiles:
        return value
    if value.ndim == 1 and value.shape[0] == 1:
        return value.repeat(num_tiles)
    raise ValueError(
        f"Expected radius shape () or ({num_tiles},), got {tuple(value.shape)}"
    )


def _radius_bounds(radius_fraction: Tensor, margin: float = 0.2) -> tuple[float, float]:
    radius = radius_fraction.detach().flatten().cpu()
    if torch.any(radius <= 0):
        raise ValueError(f"radius_fraction must be positive; got {radius.tolist()}")
    return (1.0 - margin) * float(radius.min()), (1.0 + margin) * float(radius.max())


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


def init_raw_bounded_radius(
    value: float,
    min_value: float,
    max_value: float,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    if not min_value < value < max_value:
        raise ValueError(
            "value must be between min_value and max_value; "
            f"got value={value}, min={min_value}, max={max_value}"
        )
    raw = _raw_bounded_radius(torch.tensor(value, device=device), min_value, max_value)
    return raw.requires_grad_(True) if requires_grad else raw


def _evaluate(
    phase_coeffs: Float[Tensor, "num_phase"],
    amp_coeffs: Float[Tensor, "num_amp"],
    radius_fraction: Tensor,
    rho_pixels: Tensor,
    angular_parts: Tensor,
    radial_coeffs: Tensor,
    radial_powers: Tensor,
    object_grid_size: int,
    use_softplus: bool,
) -> Complex[Tensor, "N N"]:
    rho_norm = rho_pixels / (radius_fraction * object_grid_size)
    rho = torch.clamp(rho_norm, max=1.0)

    num_phase = len(phase_coeffs)
    num_amp = len(amp_coeffs)
    max_terms = max(num_phase, num_amp)
    rho_powers = rho[None, None] ** radial_powers[:max_terms, :, None, None]
    radial = (radial_coeffs[:max_terms, :, None, None] * rho_powers).sum(dim=1)
    terms = radial * angular_parts[:max_terms]

    phase = torch.einsum("i,ihw->hw", phase_coeffs, terms[:num_phase])
    amp_raw = torch.einsum("i,ihw->hw", amp_coeffs, terms[:num_amp])
    amplitude = F.softplus(amp_raw) if use_softplus else amp_raw
    return amplitude * torch.exp(1j * phase)


def _aperture(
    radius_fraction: Tensor,
    rho_pixels: Tensor,
    object_grid_size: int,
    edge_width_px: Tensor | float,
) -> Tensor:
    if not isinstance(edge_width_px, Tensor):
        edge_width_px = torch.tensor(
            edge_width_px,
            device=rho_pixels.device,
            dtype=rho_pixels.dtype,
        )
    return torch.sigmoid(
        (radius_fraction * object_grid_size - rho_pixels) / edge_width_px
    )


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
        use_softplus: bool = True,
        radius_bounds: tuple[float, float] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        target_device = device or (
            phase_coeffs.device if isinstance(phase_coeffs, Tensor) else "cpu"
        )
        self.object_grid_size = object_grid_size
        self.num_phase_terms = num_phase_terms
        self.num_amp_terms = num_amp_terms
        self.edge_width_px = edge_width_px
        self.use_softplus = use_softplus

        rho_pixels, angular_parts, radial_coeffs, radial_powers = _basis_tensors(
            object_grid_size,
            num_phase_terms,
            num_amp_terms,
            device=target_device,
            dtype=dtype,
        )
        self.register_buffer("rho_pixels", rho_pixels)
        self.register_buffer("angular_parts", angular_parts)
        self.register_buffer("radial_coeffs", radial_coeffs)
        self.register_buffer("radial_powers", radial_powers)

        phase = (
            _default_phase_coeffs(num_phase_terms, device=target_device, dtype=dtype)
            if phase_coeffs is None
            else phase_coeffs.detach().clone().to(device=target_device, dtype=dtype)
        )
        amp = (
            _default_amp_coeffs(num_amp_terms, device=target_device, dtype=dtype)
            if amp_coeffs is None
            else amp_coeffs.detach().clone().to(device=target_device, dtype=dtype)
        )
        radius = _as_tensor(radius_fraction, device=target_device, dtype=dtype)
        phase = _tile_params(phase, num_tiles)
        amp = _tile_params(amp, num_tiles)
        radius = _tile_radius(radius, num_tiles)

        self.phase_coeffs = nn.Parameter(phase)
        self.amp_coeffs = nn.Parameter(amp)
        self.min_radius, self.max_radius = radius_bounds or _radius_bounds(radius)
        self.raw_radius = nn.Parameter(
            _raw_bounded_radius(radius, self.min_radius, self.max_radius)
        )

    @property
    def radius_fraction(self) -> Tensor:
        return bounded_radius(self.raw_radius, self.min_radius, self.max_radius)

    def tile(self, tile_idx: int) -> "Pupil":
        if self.phase_coeffs.ndim == 1:
            phase = self.phase_coeffs
            amp = self.amp_coeffs
            radius = self.radius_fraction
        else:
            phase = self.phase_coeffs[tile_idx]
            amp = self.amp_coeffs[tile_idx]
            radius = self.radius_fraction[tile_idx]
        return Pupil(
            self.object_grid_size,
            num_phase_terms=self.num_phase_terms,
            num_amp_terms=self.num_amp_terms,
            phase_coeffs=phase.detach(),
            amp_coeffs=amp.detach(),
            radius_fraction=radius.detach(),
            edge_width_px=self.edge_width_px,
            use_softplus=self.use_softplus,
            device=phase.device,
            dtype=phase.dtype,
        )

    def forward(self) -> Complex[Tensor, "N N"] | Complex[Tensor, "T N N"]:
        if self.phase_coeffs.ndim == 1:
            return self._forward_one(
                self.phase_coeffs,
                self.amp_coeffs,
                self.radius_fraction,
            )
        return torch.stack(
            [
                self._forward_one(
                    self.phase_coeffs[tile_idx],
                    self.amp_coeffs[tile_idx],
                    self.radius_fraction[tile_idx],
                )
                for tile_idx in range(self.phase_coeffs.shape[0])
            ]
        )

    def _forward_one(
        self,
        phase_coeffs: Tensor,
        amp_coeffs: Tensor,
        radius_fraction: Tensor,
    ) -> Tensor:
        rho_pixels = cast(Tensor, self.rho_pixels)
        pupil = _evaluate(
            phase_coeffs,
            amp_coeffs,
            radius_fraction,
            rho_pixels,
            cast(Tensor, self.angular_parts),
            cast(Tensor, self.radial_coeffs),
            cast(Tensor, self.radial_powers),
            self.object_grid_size,
            self.use_softplus,
        )
        aperture = _aperture(
            radius_fraction,
            rho_pixels,
            self.object_grid_size,
            self.edge_width_px,
        )
        return pupil * aperture.to(pupil.real.dtype)


def radius_fraction_from_optics(
    numerical_aperture: float,
    wavelength_m: float,
    sensor_pixel_size_m: float,
    magnification: float,
    object_to_capture_ratio: int,
) -> float:
    dx_obj = sensor_pixel_size_m / (magnification * object_to_capture_ratio)
    return (numerical_aperture / wavelength_m) * dx_obj
