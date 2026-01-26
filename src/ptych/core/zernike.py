"""
Functional Zernike pupil generation for fixed-radius pupils.

Pure-function API matching the jax-like style of forward.py/inverse.py.

Usage:
    # Precompute basis once
    basis = precompute_zernike_basis(size=256, rad_fraction=0.2)

    # Create learnable coefficients
    phase_coeffs = torch.zeros(21, requires_grad=True)
    amp_coeffs = torch.zeros(11)
    amp_coeffs[0] = 1.0
    amp_coeffs = amp_coeffs.requires_grad_(True)

    # In training loop:
    pupil = make_zernike_pupil(phase_coeffs, amp_coeffs, basis)
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor
from jaxtyping import Complex, Float


class ZernikeBasis:
    """Container for precomputed Zernike basis arrays."""

    phase_basis: Float[Tensor, "num_phase N N"]
    amp_basis: Float[Tensor, "num_amp N N"]
    mask: Float[Tensor, "N N"]
    size: int
    rad_fraction: float
    num_phase_terms: int
    num_amp_terms: int

    def __init__(
        self,
        phase_basis: Tensor,
        amp_basis: Tensor,
        mask: Tensor,
        size: int,
        rad_fraction: float,
        num_phase_terms: int,
        num_amp_terms: int,
    ) -> None:
        self.phase_basis = phase_basis
        self.amp_basis = amp_basis
        self.mask = mask
        self.size = size
        self.rad_fraction = rad_fraction
        self.num_phase_terms = num_phase_terms
        self.num_amp_terms = num_amp_terms

    def to(self, device: torch.device | str) -> "ZernikeBasis":
        """Move basis to specified device."""
        return ZernikeBasis(
            phase_basis=self.phase_basis.to(device),
            amp_basis=self.amp_basis.to(device),
            mask=self.mask.to(device),
            size=self.size,
            rad_fraction=self.rad_fraction,
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


def precompute_zernike_basis(
    size: int,
    rad_fraction: float,
    num_phase_terms: int = 21,
    num_amp_terms: int = 11,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ZernikeBasis:
    """Precompute Zernike basis for a fixed pupil radius.

    Args:
        size: Grid size (output pupil is size x size)
        rad_fraction: Pupil radius as fraction of tensor width.
            E.g., rad_fraction=0.1 with size=100 gives radius=10 pixels.
        num_phase_terms: Number of Zernike terms for phase (Noll 1 to N)
        num_amp_terms: Number of Zernike terms for amplitude
        device: torch device
        dtype: torch dtype for basis arrays

    Returns:
        ZernikeBasis containing precomputed arrays
    """
    rho_pixels, theta = make_polar_grid_fft(size, device=device, dtype=dtype)

    # Normalize rho so that rho=1 at pupil edge
    radius_pixels = rad_fraction * size
    rho_norm = rho_pixels / radius_pixels

    # Binary mask for pupil region
    mask = (rho_norm <= 1.0).to(dtype)

    # Compute Zernike polynomials
    max_terms = max(num_phase_terms, num_amp_terms)
    basis_list: list[Tensor] = []
    for j in range(1, max_terms + 1):
        n, m = noll_to_nm(j)
        z = zernike_polynomial(n, m, rho_norm, theta)
        basis_list.append(z)

    basis = torch.stack(basis_list)  # [max_terms, N, N]

    return ZernikeBasis(
        phase_basis=basis[:num_phase_terms],
        amp_basis=basis[:num_amp_terms],
        mask=mask,
        size=size,
        rad_fraction=rad_fraction,
        num_phase_terms=num_phase_terms,
        num_amp_terms=num_amp_terms,
    )


def make_zernike_pupil(
    phase_coeffs: Float[Tensor, "num_phase"],
    amp_coeffs: Float[Tensor, "num_amp"],
    basis: ZernikeBasis,
    use_softplus: bool = True,
) -> Complex[Tensor, "N N"]:
    """Generate complex pupil from Zernike coefficients.

    Args:
        phase_coeffs: Phase Zernike coefficients [num_phase_terms]
        amp_coeffs: Amplitude Zernike coefficients [num_amp_terms]
        basis: Precomputed ZernikeBasis from precompute_zernike_basis()
        use_softplus: If True, apply softplus to ensure non-negative amplitude
            (useful for optimization). If False, use raw linear combination
            (useful for ground-truth generation with exact amplitude control).

    Returns:
        Complex pupil tensor [N, N] with DC at (0, 0) in FFT-native coords
    """
    # Phase: linear combination of Zernike polynomials
    phase = torch.einsum("i,ihw->hw", phase_coeffs, basis.phase_basis)

    # Amplitude: linear combination, optionally passed through softplus
    amp_raw = torch.einsum("i,ihw->hw", amp_coeffs, basis.amp_basis)
    amplitude = torch.nn.functional.softplus(amp_raw) if use_softplus else amp_raw

    # Construct complex pupil with mask
    pupil = amplitude * torch.exp(1j * phase) * basis.mask

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


# Quick test
if __name__ == "__main__":
    print("=== Functional Zernike Test ===\n")

    size = 100
    rad_fraction = 0.2

    # Precompute basis
    basis = precompute_zernike_basis(size, rad_fraction)
    print(f"Basis: phase={basis.phase_basis.shape}, amp={basis.amp_basis.shape}")
    print(f"Radius: {rad_fraction * size} pixels\n")

    # Create learnable coefficients
    phase_coeffs = init_phase_coeffs(basis.num_phase_terms)
    amp_coeffs = init_amp_coeffs(basis.num_amp_terms)

    # Generate pupil
    pupil = make_zernike_pupil(phase_coeffs, amp_coeffs, basis)
    print(f"Pupil: shape={pupil.shape}, dtype={pupil.dtype}")
    print(f"Center |pupil[0,0]|: {pupil[0, 0].abs():.4f}")
    print(f"Outside |pupil[30,0]|: {pupil[30, 0].abs():.6f}")

    # Test gradients
    loss = pupil.abs().sum()
    loss.backward()

    print(f"\nGradients:")
    print(f"  phase_coeffs.grad: {phase_coeffs.grad is not None}")
    print(f"  amp_coeffs.grad[:3]: {amp_coeffs.grad[:3]}")
