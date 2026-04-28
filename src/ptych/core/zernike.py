import math

import torch
from torch import Tensor


def _noll_to_nm(j: int) -> tuple[int, int]:
    if j < 1:
        raise ValueError(f"Noll index j={j} must be positive")

    n = 0
    first_j_for_order = 1
    while j >= first_j_for_order + n + 1:
        first_j_for_order += n + 1
        n += 1

    order_index = j - first_j_for_order
    if n % 2 == 0:
        if order_index == 0:
            return n, 0
        m_abs = 2 * ((order_index + 1) // 2)
    else:
        m_abs = 2 * (order_index // 2) + 1

    return n, m_abs if j % 2 == 0 else -m_abs


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


def zernike_num_terms(max_radial_order: int) -> int:
    if max_radial_order < 0:
        raise ValueError(
            f"max_radial_order must be non-negative; got {max_radial_order}"
        )
    return (max_radial_order + 1) * (max_radial_order + 2) // 2


def zernike_basis_tensors(
    object_grid_size: int,
    max_radial_order: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    coords = torch.arange(object_grid_size, dtype=torch.get_default_dtype())
    coords = torch.where(
        coords >= object_grid_size / 2, coords - object_grid_size, coords
    )
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    rho_pixels = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(grid_y, grid_x)

    max_terms = zernike_num_terms(max_radial_order)
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
    coeff_tensor = torch.zeros(max_terms, max_k)
    power_tensor = torch.zeros(max_terms, max_k)
    for i, (coeffs, powers) in enumerate(zip(radial_coeffs, radial_powers)):
        coeff_tensor[i, : len(coeffs)] = torch.tensor(coeffs)
        power_tensor[i, : len(powers)] = torch.tensor(powers)

    return rho_pixels, torch.stack(angular_parts), coeff_tensor, power_tensor
