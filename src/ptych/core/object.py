import math

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
        self.amplitude = nn.Parameter(torch.sqrt(init_amp + 1e-8).detach())
        self.phase = nn.Parameter(torch.zeros_like(init_amp))

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        return self.amplitude * torch.exp(1j * self.phase)


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class UpsampleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()

        if kernel_size not in (1, 3, 5):
            raise ValueError("kernel_size must be 1, 3, or 5")

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = (
            nn.GroupNorm(_valid_group_count(out_channels), out_channels)
            if norm
            else nn.Identity()
        )

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class ObjectPriorX(nn.Module):
    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        object_to_capture_ratio: int,
        z_hw: tuple[int, int] = (16, 16),
        z_channels: int = 32,
        channels: tuple[int, ...] = (64, 64, 48, 32),
        basis_channels: int = 8,
        kernel_size: int = 3,
        amp_mode: str = "bounded",
        amp_init: float = 0.7,
        phase_range: float | None = math.pi,
        phase_init: float = 0.0,
    ) -> None:
        super().__init__()

        patch_batch_size, _, capture_height, capture_width = (
            measured_intensity_batch.shape
        )
        self.out_hw = (
            capture_height * object_to_capture_ratio,
            capture_width * object_to_capture_ratio,
        )
        self.amp_mode = amp_mode
        self.phase_range = phase_range

        z = torch.randn(patch_batch_size, z_channels, z_hw[0], z_hw[1])
        self.register_buffer("z", z)

        blocks: list[nn.Module] = []
        in_channels = z_channels
        for out_channels in channels:
            blocks.append(
                UpsampleConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    norm=True,
                    activation="leaky_relu",
                )
            )
            in_channels = out_channels

        self.trunk = nn.Sequential(*blocks)
        self.basis = nn.Sequential(
            nn.Conv2d(in_channels, basis_channels, kernel_size=1),
            nn.GroupNorm(_valid_group_count(basis_channels), basis_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Conv2d(basis_channels, 2, kernel_size=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        head_bias = self.head.bias
        assert head_bias is not None

        with torch.no_grad():
            if amp_mode == "bounded":
                if not (0.0 < amp_init < 1.0):
                    raise ValueError(
                        "For bounded amplitude, amp_init must be in (0, 1)"
                    )
                head_bias[0] = math.log(amp_init / (1.0 - amp_init))
            elif amp_mode in ("positive", "log"):
                head_bias[0] = math.log(max(amp_init, 1e-6))
            else:
                raise ValueError("amp_mode must be 'bounded', 'positive', or 'log'")

            if phase_range is None:
                head_bias[1] = phase_init
            else:
                normalized_phase = phase_init / phase_range
                normalized_phase = max(min(normalized_phase, 0.999), -0.999)
                head_bias[1] = math.atanh(normalized_phase)

    def _map_amplitude(self, raw: Tensor) -> Tensor:
        if self.amp_mode == "bounded":
            return torch.sigmoid(raw)

        if self.amp_mode == "positive":
            return F.softplus(raw) + 1e-6

        if self.amp_mode == "log":
            return torch.exp(torch.clamp(raw, min=-8.0, max=8.0))

        raise RuntimeError("Invalid amp_mode")

    def _map_phase(self, raw: Tensor) -> Tensor:
        if self.phase_range is None:
            return raw
        return self.phase_range * torch.tanh(raw)

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        x = self.trunk(self.z)
        x = F.interpolate(
            x,
            size=self.out_hw,
            mode="bilinear",
            align_corners=False,
        )

        basis = self.basis(x)
        raw = self.head(basis)
        amplitude = self._map_amplitude(raw[:, 0:1])
        phase = self._map_phase(raw[:, 1:2])

        return torch.polar(amplitude.squeeze(1), phase.squeeze(1))

    @property
    def num_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
