from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ptych.data.bayer import demosaic
from ptych.data.study import PtychStudy
from ptych.reconstruct import CaptureRegion, _crop_captures, _valid_study_captures

_RGB_REFERENCE_WAVELENGTHS_M = (
    625e-9,  # red
    525e-9,  # green
    470e-9,  # blue
)


def prepare_study_capture_rgb(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
) -> Float[torch.Tensor, "3 H W"]:
    if capture_index < 0 or capture_index >= study.captures.shape[0]:
        raise IndexError(
            f"capture_index {capture_index} out of range for {study.captures.shape[0]} captures"
        )

    capture = study.captures[capture_index:capture_index + 1]
    rgb = demosaic(capture).squeeze(0)

    if capture_region is not None:
        rgb = _crop_captures(rgb, capture_region)

    return rgb


def _channel_index_for_wavelength(wavelength_m: float) -> int:
    return min(
        range(len(_RGB_REFERENCE_WAVELENGTHS_M)),
        key=lambda idx: abs(wavelength_m - _RGB_REFERENCE_WAVELENGTHS_M[idx]),
    )


def _prepare_study_capture_channel(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
) -> Float[torch.Tensor, "H W"]:
    rgb = prepare_study_capture_rgb(
        study,
        capture_index,
        capture_region=capture_region,
    )
    valid_captures = _valid_study_captures(study)
    capture_meta = valid_captures[capture_index]
    channel_index = _channel_index_for_wavelength(capture_meta.wavelength)
    return rgb[channel_index]


def _normalize_rgb_for_display(rgb: Float[torch.Tensor, "3 H W"]) -> np.ndarray:
    arr = rgb.detach().cpu().numpy().astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)

    lo, hi = np.percentile(finite, [0.5, 99.5]).astype(np.float32)
    if float(hi) <= float(lo):
        hi = np.float32(np.max(finite, initial=1.0))
        lo = np.float32(0.0)

    clipped = np.clip(arr, lo, hi)
    denom = float(hi - lo)
    if denom <= 0.0:
        normalized = np.zeros_like(clipped, dtype=np.float32)
    else:
        normalized = (clipped - lo) / denom

    return np.transpose(normalized, (1, 2, 0))


def _normalize_scalar_for_display(image: Float[torch.Tensor, "H W"]) -> np.ndarray:
    arr = image.detach().cpu().numpy().astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [0.5, 99.5]).astype(np.float32)
    if float(hi) <= float(lo):
        hi = np.float32(np.max(finite, initial=1.0))
        lo = np.float32(0.0)

    clipped = np.clip(arr, lo, hi)
    denom = float(hi - lo)
    if denom <= 0.0:
        return np.zeros_like(clipped, dtype=np.float32)

    return (clipped - lo) / denom


def show_study_capture(
    study: PtychStudy,
    capture_index: int,
    capture_region: CaptureRegion | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    mode: Literal["reconstruction", "rgb", "r", "g", "b"] = "reconstruction",
) -> Figure:
    created_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    valid_captures = _valid_study_captures(study)
    capture_meta = valid_captures[capture_index]
    resolved_title = title
    if resolved_title is None:
        resolved_title = (
            f"{capture_meta.filename} | "
            f"lambda={capture_meta.wavelength * 1e9:.0f} nm | "
            f"k=({float(study.kx_batch[capture_index]):.4f}, {float(study.ky_batch[capture_index]):.4f})"
        )

    if mode == "rgb":
        rgb = prepare_study_capture_rgb(
            study,
            capture_index,
            capture_region=capture_region,
        )
        ax.imshow(_normalize_rgb_for_display(rgb))
    elif mode == "reconstruction":
        image = _prepare_study_capture_channel(
            study,
            capture_index,
            capture_region=capture_region,
        )
        ax.imshow(_normalize_scalar_for_display(image), cmap="gray")
    else:
        rgb = prepare_study_capture_rgb(
            study,
            capture_index,
            capture_region=capture_region,
        )
        channel_index = {"r": 0, "g": 1, "b": 2}[mode]
        ax.imshow(rgb[channel_index].detach().cpu().numpy(), cmap="gray")
        resolved_title = f"{resolved_title} | {mode.upper()}"

    ax.set_title(resolved_title)
    ax.set_axis_off()

    backend = plt.get_backend().lower()
    if created_figure and "agg" not in backend:
        plt.show()

    return fig
