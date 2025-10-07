from __future__ import annotations
from dataclasses import dataclass, field
import torch
from typing import Sequence

# ----------  per-capture  ----------
@dataclass(slots=True, frozen=True)
class ImageCapture:
    image: torch.Tensor          # 2-D, float32, any device
    k_vector: torch.Tensor       # (2,), float32, same device

    def __post_init__(self):
        if self.image.ndim != 2:
            raise ValueError("image must be 2-D (H, W)")
        if self.k_vector.shape != (2,):
            raise ValueError("k_vector must have shape (2,)")
        if self.image.dtype != torch.float32:
            object.__setattr__(self, "image", self.image.float())
        if self.k_vector.dtype != torch.float32:
            object.__setattr__(self, "k_vector", self.k_vector.float())

# ----------  acquisition constants  ----------
@dataclass(slots=True, frozen=True)
class AcquisitionSettings:
    du: float                    # lattice spacing in k-space
    pixel_size: float            # sensor pixel pitch in metres
    wavelength: float            # illumination wavelength in metres
    # add NA, exposure, etc. here

    def __post_init__(self):
        if self.du <= 0:
            raise ValueError("du must be positive")
        if self.pixel_size <= 0 or self.wavelength <= 0:
            raise ValueError("pixel_size and wavelength must be positive")

# ----------  series container  ----------
@dataclass(slots=True)
class ImageSeries:
    captures: Sequence[ImageCapture]
    settings: AcquisitionSettings
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # --- constructor validation ---
    def __post_init__(self):
        if not self.captures:
            raise ValueError("captures list cannot be empty")

        # Move everything to declared device exactly once
        self.to(self.device)

        # Validate shapes are consistent
        h, w = self.captures[0].image.shape
        for c in self.captures:
            if c.image.shape != (h, w):
                raise ValueError("all images must share shape (H, W)")

    # --- convenience properties ---
    @property
    def image_size(self) -> torch.Tensor:
        # returns tensor [H, W] on CPU (unit-free)
        return torch.tensor(self.captures[0].image.shape, dtype=torch.int32)

    @property
    def max_k(self) -> torch.Tensor:
        # magnitude of largest k-vector present
        kv = torch.stack([c.k_vector for c in self.captures], dim=0)
        return kv.abs().max(dim=0).values          # shape (2,)

    # --- device mover ---
    def to(self, device: torch.device | str) -> "ImageSeries":
        device = torch.device(device)
        if device == self.device:
            return self

        moved: list[ImageCapture] = []
        for c in self.captures:
            moved.append(
                ImageCapture(
                    image=c.image.to(device, non_blocking=True),
                    k_vector=c.k_vector.to(device)
                )
            )
        self.captures = moved          # mutate in place for simplicity
        self.device = device
        return self
