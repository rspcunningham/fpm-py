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
        init_amplitude_low_res = torch.sqrt(
            measured_intensity_batch[:, 0].clamp_min(0) + 1e-8
        )
        init_amplitude = F.interpolate(
            init_amplitude_low_res.unsqueeze(1),
            scale_factor=object_to_capture_ratio,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        init_amplitude = init_amplitude.detach().clamp(1e-6, 1 - 1e-6)
        self.raw_amplitude = nn.Parameter(torch.logit(init_amplitude).detach())
        self.phase = nn.Parameter(torch.zeros_like(init_amplitude))

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        return torch.sigmoid(self.raw_amplitude) * torch.exp(1j * self.phase)
