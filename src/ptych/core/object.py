import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


class Object(nn.Module):
    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        object_to_capture_ratio: int,
    ) -> None:
        super().__init__()
        init_amp = F.interpolate(
            measured_intensity_batch[:, 0].unsqueeze(1),
            scale_factor=object_to_capture_ratio,
            mode="nearest",
        ).squeeze(1)
        init_amplitude = torch.sqrt(init_amp + 1e-8).detach().clamp_min(1e-6)
        self.raw_amplitude = nn.Parameter(
            (init_amplitude + torch.log(-torch.expm1(-init_amplitude))).detach()
        )
        self.phase = nn.Parameter(torch.zeros_like(init_amp))

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        return F.softplus(self.raw_amplitude) * torch.exp(1j * self.phase)
