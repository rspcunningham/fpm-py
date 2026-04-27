import math

import torch
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


def zernike_basis_tensors(
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
