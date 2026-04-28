import math
from typing import Optional, Tuple

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
    """
    Pick a GroupNorm group count that divides channels.
    """
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


class UpsampleConvBlock(nn.Module):
    """
    One decoder block:

        bilinear upsample -> convolution -> normalization -> activation

    kernel_size=1 gives a Deep-Decoder-like prior.
    kernel_size=3 gives a more standard convolutional DIP-style prior.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        norm: bool = True,
        activation: str = "leaky_relu",
    ):
        super().__init__()

        if kernel_size not in (1, 3, 5):
            raise ValueError("Use kernel_size 1, 3, or 5.")

        pad = kernel_size // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=pad,
        )

        self.norm = (
            nn.GroupNorm(_valid_group_count(out_ch), out_ch) if norm else nn.Identity()
        )

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ObjectPriorX(nn.Module):
    """
    Untrained neural prior for a complex 2D object.

    The network outputs:

        amplitude A(r)
        phase     phi(r)
        object    O(r) = A(r) * exp(i phi(r))

    The important architectural choice is that amplitude and phase share
    the same decoder trunk and the same low-dimensional basis h(r).
    """

    def __init__(
        self,
        measured_intensity_batch: Float[
            Tensor, "patch_batch illumination height width"
        ],
        object_to_capture_ratio: int,
        z_hw: Tuple[int, int] = (16, 16),
        z_channels: int = 32,
        channels: Tuple[int, ...] = (64, 64, 48, 32),
        basis_channels: int = 8,
        kernel_size: int = 3,
        amp_mode: str = "bounded",
        amp_init: float = 0.7,
        phase_range: Optional[float] = math.pi,
        phase_init: float = 0.0,
    ):
        """
        Args:
            measured_intensity_batch:
                Measured capture batch. Used to match the Object constructor and
                derive patch batch size and output spatial size.

            object_to_capture_ratio:
                Upsampling ratio from capture grid to object grid.

            z_hw:
                Spatial size of the fixed random input.

            z_channels:
                Number of channels in the fixed random input.

            channels:
                Decoder widths. Smaller is usually more regularized.

            basis_channels:
                Number of shared feature maps used to generate both amplitude
                and phase. This is the main joint-regularization knob.

            kernel_size:
                1: Deep-Decoder-like, stronger low-complexity prior.
                3: DIP-like convolutional prior.

            amp_mode:
                "bounded": amplitude in [0, 1].
                "positive": amplitude > 0 via softplus.
                "log": amplitude = exp(raw).

            phase_range:
                If not None, phase is constrained to
                    [-phase_range, phase_range]
                using tanh.
                If None, phase is unconstrained.

            amp_init:
                Approximate initial amplitude.

            phase_init:
                Approximate initial phase.
        """
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

        # Fixed random input. Each patch gets a separate latent image.
        z = torch.randn(patch_batch_size, z_channels, z_hw[0], z_hw[1])
        self.register_buffer("z", z)

        # Shared decoder trunk.
        blocks = []
        c = z_channels
        for c_next in channels:
            blocks.append(
                UpsampleConvBlock(
                    c,
                    c_next,
                    kernel_size=kernel_size,
                    norm=True,
                    activation="leaky_relu",
                )
            )
            c = c_next

        self.trunk = nn.Sequential(*blocks)

        # Narrow shared basis h(r).
        # Amplitude and phase are both generated from this same feature basis.
        self.basis = nn.Sequential(
            nn.Conv2d(c, basis_channels, kernel_size=1),
            nn.GroupNorm(_valid_group_count(basis_channels), basis_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Late split into amplitude and phase.
        # Output channel 0: raw amplitude
        # Output channel 1: raw phase
        self.head = nn.Conv2d(basis_channels, 2, kernel_size=1)

        # Small output initialization keeps the initial object near a simple field.
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)

        with torch.no_grad():
            if amp_mode == "bounded":
                if not (0.0 < amp_init < 1.0):
                    raise ValueError(
                        "For bounded amplitude, amp_init must be in (0, 1)."
                    )
                self.head.bias[0] = math.log(amp_init / (1.0 - amp_init))

            elif amp_mode in ("positive", "log"):
                self.head.bias[0] = math.log(max(amp_init, 1e-6))

            else:
                raise ValueError("amp_mode must be 'bounded', 'positive', or 'log'.")

            if phase_range is None:
                self.head.bias[1] = phase_init
            else:
                normalized = phase_init / phase_range
                normalized = max(min(normalized, 0.999), -0.999)
                self.head.bias[1] = math.atanh(normalized)

    def _map_amplitude(self, raw: torch.Tensor) -> torch.Tensor:
        if self.amp_mode == "bounded":
            return torch.sigmoid(raw)

        if self.amp_mode == "positive":
            return F.softplus(raw) + 1e-6

        if self.amp_mode == "log":
            return torch.exp(torch.clamp(raw, min=-8.0, max=8.0))

        raise RuntimeError("Invalid amp_mode.")

    def _map_phase(self, raw: torch.Tensor) -> torch.Tensor:
        if self.phase_range is None:
            return raw

        return self.phase_range * torch.tanh(raw)

    def forward(self) -> Complex[Tensor, "patch_batch object_height object_width"]:
        x = self.trunk(self.z)

        # Resize exactly to the desired object size.
        x = F.interpolate(
            x,
            size=self.out_hw,
            mode="bilinear",
            align_corners=False,
        )

        h = self.basis(x)
        raw = self.head(h)

        raw_amp = raw[:, 0:1]
        raw_phase = raw[:, 1:2]

        amplitude = self._map_amplitude(raw_amp)
        phase = self._map_phase(raw_phase)

        return torch.polar(amplitude.squeeze(1), phase.squeeze(1))

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SharedSupportObjectPriorX(nn.Module):
    """
    Stronger joint prior.

    The decoder produces K soft support maps S_k(r).
    Amplitude and phase are both mixtures over the same S_k(r):

        A(r)   = sum_k S_k(r) A_k
        phi(r) = sum_k S_k(r) phi_k

    This enforces co-located structure more strongly than a simple two-channel head.
    """

    def __init__(
        self,
        out_hw: Tuple[int, int],
        z_hw: Tuple[int, int] = (16, 16),
        z_channels: int = 32,
        channels: Tuple[int, ...] = (64, 64, 48, 32),
        num_components: int = 8,
        kernel_size: int = 3,
        phase_range: float = math.pi,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.out_hw = tuple(out_hw)
        self.num_components = num_components
        self.phase_range = phase_range
        self.temperature = temperature

        z = torch.randn(1, z_channels, z_hw[0], z_hw[1])
        self.register_buffer("z", z)

        blocks = []
        c = z_channels
        for c_next in channels:
            blocks.append(UpsampleConvBlock(c, c_next, kernel_size=kernel_size))
            c = c_next

        self.trunk = nn.Sequential(*blocks)

        # Produces K spatial component logits.
        self.support_logits = nn.Conv2d(c, num_components, kernel_size=1)

        # Per-component amplitude and phase values.
        self.amp_atoms_raw = nn.Parameter(torch.zeros(num_components))
        self.phase_atoms_raw = nn.Parameter(torch.zeros(num_components))

    def forward(self) -> Tensor:
        x = self.trunk(self.z)

        x = F.interpolate(
            x,
            size=self.out_hw,
            mode="bilinear",
            align_corners=False,
        )

        logits = self.support_logits(x)

        # Shared support maps.
        supports = torch.softmax(logits / self.temperature, dim=1)

        amp_atoms = torch.sigmoid(self.amp_atoms_raw)
        phase_atoms = self.phase_range * torch.tanh(self.phase_atoms_raw)

        amplitude = torch.einsum("bkhw,k->bhw", supports, amp_atoms)[:, None]
        phase = torch.einsum("bkhw,k->bhw", supports, phase_atoms)[:, None]

        obj = torch.complex(
            amplitude * torch.cos(phase),
            amplitude * torch.sin(phase),
        )

        return obj
