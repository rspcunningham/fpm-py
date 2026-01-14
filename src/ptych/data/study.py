from pathlib import Path
import torch
import numpy as np
from jaxtyping import Float

from ptych.data.types import StudyManifest, LedPosition

class PtychStudy:
    manifest: StudyManifest
    captures: Float[torch.Tensor, "B n n"] # [B, n, n] float on (0, 1)
    kx_batch: Float[torch.Tensor, "B"] # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ky_batch: Float[torch.Tensor, "B"] # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)

    def __init__(
        self,
        manifest: StudyManifest,
        captures: Float[torch.Tensor, "B n n"], # [B, n, n] float on (0, 1)
        kx_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
        ky_batch: Float[torch.Tensor, "B"], # [B] float on (-0.5, 0.5) (normalized for an n * n grid!)
    ):
        self.manifest = manifest
        self.captures = captures
        self.kx_batch = kx_batch
        self.ky_batch = ky_batch

    @classmethod
    def from_disk(cls, dir_path: str | Path) -> 'PtychStudy':
