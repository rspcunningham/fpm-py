import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


class Object(nn.Module):
    def __init__(
        self,
        measured_intensities: Float[Tensor, "T B n n"],
        object_to_capture_ratio: int,
    ) -> None:
        super().__init__()
        init_amp = F.interpolate(
            measured_intensities[:, 0].unsqueeze(1),
            scale_factor=object_to_capture_ratio,
            mode="nearest",
        ).squeeze(1)
        self.amplitude = nn.Parameter(torch.sqrt(init_amp + 1e-8).detach())
        self.phase = nn.Parameter(torch.zeros_like(init_amp))

    def forward(self) -> Complex[Tensor, "T N N"]:
        return self.amplitude * torch.exp(1j * self.phase)
