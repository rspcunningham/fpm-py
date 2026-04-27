import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


class PupilBasis:
    """Precomputed radius-independent basis terms."""

    rho_pixels: Float[Tensor, "N N"]
    angular_parts: Float[Tensor, "max_terms N N"]
    radial_coeffs: Float[Tensor, "max_terms max_k"]
    radial_powers: Float[Tensor, "max_terms max_k"]
    size: int
    num_phase_terms: int
    num_amp_terms: int

    def __init__(
        self,
        rho_pixels: Tensor,
        angular_parts: Tensor,
        radial_coeffs: Tensor,
        radial_powers: Tensor,
        size: int,
        num_phase_terms: int,
        num_amp_terms: int,
    ) -> None:
        self.rho_pixels = rho_pixels
        self.angular_parts = angular_parts
        self.radial_coeffs = radial_coeffs
        self.radial_powers = radial_powers
        self.size = size
        self.num_phase_terms = num_phase_terms
        self.num_amp_terms = num_amp_terms

    def to(self, device: torch.device | str) -> "PupilBasis":
        return PupilBasis(
            rho_pixels=self.rho_pixels.to(device),
            angular_parts=self.angular_parts.to(device),
            radial_coeffs=self.radial_coeffs.to(device),
            radial_powers=self.radial_powers.to(device),
            size=self.size,
            num_phase_terms=self.num_phase_terms,
            num_amp_terms=self.num_amp_terms,
        )


class Pupil(NamedTuple):
    phase_coeffs: Tensor
    amp_coeffs: Tensor
    basis: PupilBasis
    radius_fraction: Tensor


def _as_basis_tensor(value: Tensor | float, basis: PupilBasis) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(
        value, device=basis.rho_pixels.device, dtype=basis.rho_pixels.dtype
    )


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


def _polar_grid(
    size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    coords = torch.arange(size, device=device, dtype=dtype)
    coords = torch.where(coords >= size / 2, coords - size, coords)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.sqrt(grid_x**2 + grid_y**2), torch.atan2(grid_y, grid_x)


def make_basis(
    size: int,
    num_phase_terms: int = 21,
    num_amp_terms: int = 11,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PupilBasis:
    rho_pixels, theta = _polar_grid(size, device=device, dtype=dtype)
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

    return PupilBasis(
        rho_pixels=rho_pixels,
        angular_parts=torch.stack(angular_parts),
        radial_coeffs=coeff_tensor,
        radial_powers=power_tensor,
        size=size,
        num_phase_terms=num_phase_terms,
        num_amp_terms=num_amp_terms,
    )


def _evaluate_pupil(
    phase_coeffs: Float[Tensor, "num_phase"],
    amp_coeffs: Float[Tensor, "num_amp"],
    basis: PupilBasis,
    radius_fraction: Tensor | float,
    use_softplus: bool,
) -> Complex[Tensor, "N N"]:
    radius_fraction = _as_basis_tensor(radius_fraction, basis)
    rho_norm = basis.rho_pixels / (radius_fraction * basis.size)
    rho = torch.clamp(rho_norm, max=1.0)

    num_phase = len(phase_coeffs)
    num_amp = len(amp_coeffs)
    max_terms = max(num_phase, num_amp)
    rho_powers = rho[None, None] ** basis.radial_powers[:max_terms, :, None, None]
    radial = (basis.radial_coeffs[:max_terms, :, None, None] * rho_powers).sum(dim=1)
    terms = radial * basis.angular_parts[:max_terms]

    phase = torch.einsum("i,ihw->hw", phase_coeffs, terms[:num_phase])
    amp_raw = torch.einsum("i,ihw->hw", amp_coeffs, terms[:num_amp])
    amplitude = F.softplus(amp_raw) if use_softplus else amp_raw

    return amplitude * torch.exp(1j * phase)


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

    frac = (value - min_value) / (max_value - min_value)
    frac = min(max(frac, 1e-6), 1.0 - 1e-6)
    raw = math.log(frac / (1.0 - frac))
    tensor = torch.tensor(raw, device=device, dtype=torch.float32)
    return tensor.requires_grad_(True) if requires_grad else tensor


def bounded_radius(raw_radius: Tensor, min_value: float, max_value: float) -> Tensor:
    return min_value + (max_value - min_value) * torch.sigmoid(raw_radius)


def make_aperture(
    basis: PupilBasis,
    radius_fraction: Tensor | float,
    edge_width_px: Tensor | float = 2.0,
) -> Float[Tensor, "N N"]:
    radius_fraction = _as_basis_tensor(radius_fraction, basis)
    edge_width_px = _as_basis_tensor(edge_width_px, basis)
    return torch.sigmoid(
        (radius_fraction * basis.size - basis.rho_pixels) / edge_width_px
    )


def make_pupil(
    phase_coeffs: Float[Tensor, "num_phase"],
    amp_coeffs: Float[Tensor, "num_amp"],
    basis: PupilBasis,
    radius_fraction: Tensor | float,
    edge_width_px: Tensor | float = 2.0,
    use_softplus: bool = True,
) -> Complex[Tensor, "N N"]:
    pupil = _evaluate_pupil(
        phase_coeffs,
        amp_coeffs,
        basis,
        radius_fraction,
        use_softplus=use_softplus,
    )
    aperture = make_aperture(basis, radius_fraction, edge_width_px)
    return pupil * aperture.to(pupil.real.dtype)


def init_phase_coeffs(
    num_terms: int = 21,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    coeffs = torch.zeros(num_terms, device=device)
    return coeffs.requires_grad_(True) if requires_grad else coeffs


def init_amp_coeffs(
    num_terms: int = 11,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    coeffs = torch.zeros(num_terms, device=device)
    coeffs[0] = 1.0
    return coeffs.requires_grad_(True) if requires_grad else coeffs


def init_radius(
    value: float = 0.2,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    radius = torch.tensor(value, device=device)
    return radius.requires_grad_(True) if requires_grad else radius


def make_ideal_pupil(
    object_grid_size: int,
    numerical_aperture: float,
    wavelength_m: float,
    sensor_pixel_size_m: float,
    magnification: float,
    object_to_capture_ratio: int,
    num_phase_terms: int = 1,
    num_amp_terms: int = 1,
    device: torch.device | str | None = None,
) -> Pupil:
    dx_obj = sensor_pixel_size_m / (magnification * object_to_capture_ratio)
    radius_fraction = (numerical_aperture / wavelength_m) * dx_obj
    target_device = device or "cpu"

    return Pupil(
        phase_coeffs=init_phase_coeffs(
            num_phase_terms, device=target_device, requires_grad=False
        ),
        amp_coeffs=init_amp_coeffs(
            num_amp_terms, device=target_device, requires_grad=False
        ),
        basis=make_basis(
            object_grid_size,
            num_phase_terms=num_phase_terms,
            num_amp_terms=num_amp_terms,
            device=target_device,
        ),
        radius_fraction=init_radius(
            radius_fraction, device=target_device, requires_grad=False
        ),
    )
