"""
Functional Zernike pupil generation with learnable radius.

Pure-function API matching the jax-like style of forward.py/inverse.py.

The basis precomputes radius-independent components:
- Angular parts (cos/sin of theta)
- Polynomial coefficients (factorial ratios)

Radius-dependent evaluation happens at runtime for differentiability.

Usage:
    # Precompute basis once (no rad_fraction needed)
    basis = precompute_zernike_basis(size=256)

    # Create learnable coefficients and radius
    phase_coeffs = torch.zeros(21, requires_grad=True)
    amp_coeffs = torch.zeros(11)
    amp_coeffs[0] = 1.0
    amp_coeffs = amp_coeffs.requires_grad_(True)
    rad_fraction = torch.tensor(0.2, requires_grad=True)

    # In training loop:
    pupil = make_zernike_pupil(phase_coeffs, amp_coeffs, basis, rad_fraction)
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor
from jaxtyping import Complex, Float


class ZernikeBasis:
    """Container for precomputed Zernike basis components.

    Precomputes radius-independent quantities:
    - rho_pixels: radial distance in pixels (unnormalized)
    - theta: angular coordinates
    - angular_parts: cos(m*θ) or sin(|m|*θ) for each term
    - poly_coeffs: polynomial coefficients for each term
    - poly_powers: corresponding powers for each term
    """

    rho_pixels: Float[Tensor, "N N"]
    theta: Float[Tensor, "N N"]
    angular_parts: Float[Tensor, "max_terms N N"]
    poly_coeffs: list[Tensor]  # list of [num_k] tensors
    poly_powers: list[Tensor]  # list of [num_k] tensors (integer powers)
    size: int
    num_phase_terms: int
    num_amp_terms: int

    def __init__(
        self,
        rho_pixels: Tensor,
        theta: Tensor,
        angular_parts: Tensor,
        poly_coeffs: list[Tensor],
        poly_powers: list[Tensor],
        size: int,
        num_phase_terms: int,
        num_amp_terms: int,
    ) -> None:
        self.rho_pixels = rho_pixels
        self.theta = theta
        self.angular_parts = angular_parts
        self.poly_coeffs = poly_coeffs
        self.poly_powers = poly_powers
        self.size = size
        self.num_phase_terms = num_phase_terms
        self.num_amp_terms = num_amp_terms

    def to(self, device: torch.device | str) -> "ZernikeBasis":
        """Move basis to specified device."""
        return ZernikeBasis(
            rho_pixels=self.rho_pixels.to(device),
            theta=self.theta.to(device),
            angular_parts=self.angular_parts.to(device),
            poly_coeffs=[c.to(device) for c in self.poly_coeffs],
            poly_powers=[p.to(device) for p in self.poly_powers],
            size=self.size,
            num_phase_terms=self.num_phase_terms,
            num_amp_terms=self.num_amp_terms,
        )


class ZernikeParams(NamedTuple):
    """Container for Zernike pupil parameterization.

    Used with solve_inverse to learn pupil via Zernike coefficients
    instead of raw pixel values.
    """

    phase_coeffs: Tensor  # [num_phase] - phase aberration coefficients
    amp_coeffs: Tensor  # [num_amp] - amplitude coefficients
    basis: ZernikeBasis  # precomputed basis (not learned)
    rad_fraction: Tensor  # fraction of N for pupil radius, can be learnable


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index (1-based) to (n, m) radial and azimuthal orders."""
    j_current = 1
    n = 0
    while True:
        for m in range(-n, n + 1, 2):
            if j_current == j:
                if m != 0:
                    if j % 2 == 0:
                        m = abs(m)
                    else:
                        m = -abs(m)
                return n, m
            j_current += 1
        n += 1
        if n > 100:
            raise ValueError(f"Noll index j={j} too large")


def radial_polynomial(n: int, m_abs: int, rho: Tensor) -> Tensor:
    """Compute radial Zernike polynomial R_n^|m|(rho)."""
    result = torch.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        num = math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + m_abs) // 2 - k)
            * math.factorial((n - m_abs) // 2 - k)
        )
        sign = 1 if k % 2 == 0 else -1
        coeff = sign * num / den
        result = result + coeff * (rho ** (n - 2 * k))
    return result


def zernike_polynomial(n: int, m: int, rho: Tensor, theta: Tensor) -> Tensor:
    """Compute single Zernike polynomial Z_n^m(rho, theta)."""
    radial = radial_polynomial(n, abs(m), rho)
    if m > 0:
        return radial * torch.cos(m * theta)
    elif m < 0:
        return radial * torch.sin(abs(m) * theta)
    else:
        return radial


def make_polar_grid_fft(
    size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Create polar coordinates in FFT-native layout (center at index 0,0).

    Returns:
        rho: Radial distance in pixels [N, N]
        theta: Angular coordinate [N, N]
    """
    coords = torch.arange(size, device=device, dtype=dtype)
    coords = torch.where(coords >= size / 2, coords - size, coords)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    rho = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(grid_y, grid_x)
    return rho, theta


def compute_radial_coefficients(n: int, m_abs: int) -> tuple[list[float], list[int]]:
    """Compute radial polynomial coefficients and powers for R_n^|m|(rho).

    Returns:
        coeffs: list of coefficients
        powers: list of corresponding powers (n - 2*k for each k)
    """
    coeffs: list[float] = []
    powers: list[int] = []
    for k in range((n - m_abs) // 2 + 1):
        num = math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + m_abs) // 2 - k)
            * math.factorial((n - m_abs) // 2 - k)
        )
        sign = 1 if k % 2 == 0 else -1
        coeffs.append(sign * num / den)
        powers.append(n - 2 * k)
    return coeffs, powers


def precompute_zernike_basis(
    size: int,
    num_phase_terms: int = 21,
    num_amp_terms: int = 11,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ZernikeBasis:
    """Precompute radius-independent Zernike basis components.

    Args:
        size: Grid size (output pupil is size x size)
        num_phase_terms: Number of Zernike terms for phase (Noll 1 to N)
        num_amp_terms: Number of Zernike terms for amplitude
        device: torch device
        dtype: torch dtype for basis arrays

    Returns:
        ZernikeBasis containing precomputed components for runtime evaluation
    """
    rho_pixels, theta = make_polar_grid_fft(size, device=device, dtype=dtype)

    max_terms = max(num_phase_terms, num_amp_terms)

    # Precompute angular parts and polynomial coefficients for each term
    angular_parts_list: list[Tensor] = []
    poly_coeffs_list: list[Tensor] = []
    poly_powers_list: list[Tensor] = []

    for j in range(1, max_terms + 1):
        n, m = noll_to_nm(j)

        # Angular part: cos(m*theta), sin(|m|*theta), or ones
        if m > 0:
            angular = torch.cos(m * theta)
        elif m < 0:
            angular = torch.sin(abs(m) * theta)
        else:
            angular = torch.ones_like(theta)
        angular_parts_list.append(angular)

        # Radial polynomial coefficients and powers
        coeffs, powers = compute_radial_coefficients(n, abs(m))
        poly_coeffs_list.append(torch.tensor(coeffs, device=device, dtype=dtype))
        poly_powers_list.append(torch.tensor(powers, device=device, dtype=torch.long))

    angular_parts = torch.stack(angular_parts_list)  # [max_terms, N, N]

    return ZernikeBasis(
        rho_pixels=rho_pixels,
        theta=theta,
        angular_parts=angular_parts,
        poly_coeffs=poly_coeffs_list,
        poly_powers=poly_powers_list,
        size=size,
        num_phase_terms=num_phase_terms,
        num_amp_terms=num_amp_terms,
    )


def evaluate_zernike_term(
    rho_norm: Tensor,
    angular: Tensor,
    coeffs: Tensor,
    powers: Tensor,
) -> Tensor:
    """Evaluate a single Zernike term: radial_poly(rho_norm) * angular.

    Args:
        rho_norm: Normalized radial coordinate [N, N]
        angular: Precomputed angular part [N, N]
        coeffs: Polynomial coefficients [num_k]
        powers: Corresponding powers [num_k]

    Returns:
        Evaluated Zernike polynomial [N, N]
    """
    # Evaluate radial polynomial: sum of coeff_k * rho^power_k
    radial = torch.zeros_like(rho_norm)
    for c, p in zip(coeffs, powers):
        radial = radial + c * (rho_norm ** p)
    return radial * angular


def make_zernike_pupil(
    phase_coeffs: Float[Tensor, "num_phase"],
    amp_coeffs: Float[Tensor, "num_amp"],
    basis: ZernikeBasis,
    rad_fraction: Tensor | float,
    use_softplus: bool = True,
    mask_sharpness: float = 50.0,
) -> Complex[Tensor, "N N"]:
    """Generate complex pupil from Zernike coefficients with differentiable radius.

    Args:
        phase_coeffs: Phase Zernike coefficients [num_phase_terms]
        amp_coeffs: Amplitude Zernike coefficients [num_amp_terms]
        basis: Precomputed ZernikeBasis from precompute_zernike_basis()
        rad_fraction: Pupil radius as fraction of tensor width (can be learnable tensor)
        use_softplus: If True, apply softplus to ensure non-negative amplitude
            (useful for optimization). If False, use raw linear combination
            (useful for ground-truth generation with exact amplitude control).
        mask_sharpness: Steepness of soft mask sigmoid (higher = sharper edge)

    Returns:
        Complex pupil tensor [N, N] with DC at (0, 0) in FFT-native coords
    """
    # Convert rad_fraction to tensor if needed
    if not isinstance(rad_fraction, Tensor):
        rad_fraction = torch.tensor(rad_fraction, device=basis.rho_pixels.device, dtype=basis.rho_pixels.dtype)

    # Compute normalized rho (differentiable w.r.t. rad_fraction)
    radius_pixels = rad_fraction * basis.size
    rho_norm = basis.rho_pixels / radius_pixels

    # Soft mask for gradient flow (sigmoid centered at rho_norm=1)
    mask = torch.sigmoid((1.0 - rho_norm) * mask_sharpness)

    # Evaluate phase Zernike terms and sum with coefficients
    num_phase = len(phase_coeffs)
    phase_terms: list[Tensor] = []
    for j in range(num_phase):
        z = evaluate_zernike_term(
            rho_norm,
            basis.angular_parts[j],
            basis.poly_coeffs[j],
            basis.poly_powers[j],
        )
        phase_terms.append(z)
    phase_basis = torch.stack(phase_terms)  # [num_phase, N, N]
    phase = torch.einsum("i,ihw->hw", phase_coeffs, phase_basis)

    # Evaluate amplitude Zernike terms and sum with coefficients
    num_amp = len(amp_coeffs)
    amp_terms: list[Tensor] = []
    for j in range(num_amp):
        z = evaluate_zernike_term(
            rho_norm,
            basis.angular_parts[j],
            basis.poly_coeffs[j],
            basis.poly_powers[j],
        )
        amp_terms.append(z)
    amp_basis = torch.stack(amp_terms)  # [num_amp, N, N]
    amp_raw = torch.einsum("i,ihw->hw", amp_coeffs, amp_basis)
    amplitude = torch.nn.functional.softplus(amp_raw) if use_softplus else amp_raw

    # Construct complex pupil with soft mask
    pupil = amplitude * torch.exp(1j * phase) * mask

    return pupil


def init_phase_coeffs(
    num_terms: int = 21,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    """Initialize phase coefficients (all zeros = no aberration)."""
    coeffs = torch.zeros(num_terms, device=device)
    if requires_grad:
        coeffs = coeffs.requires_grad_(True)
    return coeffs


def init_amp_coeffs(
    num_terms: int = 11,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    """Initialize amplitude coefficients (piston=1, rest=0 = uniform transmission)."""
    coeffs = torch.zeros(num_terms, device=device)
    coeffs[0] = 1.0
    if requires_grad:
        coeffs = coeffs.requires_grad_(True)
    return coeffs


def init_rad_fraction(
    value: float = 0.2,
    device: torch.device | str = "cpu",
    requires_grad: bool = True,
) -> Tensor:
    """Initialize pupil radius fraction (as fraction of tensor width).

    Args:
        value: Initial radius as fraction of tensor size (e.g., 0.2 = 20% of width)
        device: torch device
        requires_grad: Whether the radius should be learnable

    Returns:
        Scalar tensor for rad_fraction
    """
    rad = torch.tensor(value, device=device)
    if requires_grad:
        rad = rad.requires_grad_(True)
    return rad


