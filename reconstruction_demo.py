import os
from typing import Any

import numpy as np
import torch
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt

from ptych import solve_tiled, PtychStudy
from ptych.core.pupil import make_ideal_pupil, make_zernike_pupil, ZernikeParams
from interpolation import interpolate_green

#BASE_DIR = "./demo/synthetic"
BASE_DIR = "./demo/real"
OUTPUT_DIR = f"{BASE_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

study = PtychStudy.from_disk(BASE_DIR)

# Center-crop to 256x256 for quick validation (2x2 grid of 128 tiles)
roi_size = 128
crop_size = 256
_, h, w = study.captures.shape
sh, sw = (h - crop_size) // 2, (w - crop_size) // 2

n_captures = 37
captures = study.captures[:n_captures, sh:sh + crop_size, sw:sw + crop_size]
captures = interpolate_green(captures)
captures = captures / captures.max()
n_rows = captures.shape[1] // roi_size
n_cols = captures.shape[2] // roi_size

# Initialize pupil from NA and optical parameters
upsample_ratio = 4
N = roi_size * upsample_ratio

pupil = make_ideal_pupil(
    N=N,
    NA=0.13,
    wavelength_m=study.manifest.captures[0].wavelength,
    sensor_pixel_size_m=study.manifest.sensor_pixel_size,
    magnification=study.manifest.magnification,
    downsample_ratio=upsample_ratio,
    num_phase_terms=20,
    num_amp_terms=20,
)


def save_tensor(tensor: torch.Tensor, path: str):
    np.save(path, tensor.cpu().numpy())


def render_scalar_image(tensor: torch.Tensor, mode: str = "intensity") -> np.ndarray:
    arr = tensor.detach().cpu().numpy()

    if mode == "intensity":
        return np.abs(arr).astype(np.float32) ** 2
    if mode == "amplitude":
        return np.abs(arr).astype(np.float32)
    if mode == "phase":
        return np.angle(arr).astype(np.float32)

    raise ValueError(f"Unsupported render mode: {mode}")


def normalize_preview(
    arr: np.ndarray,
    *,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    tone_map: str = "gamma",
    gamma: float = 2.0,
    log_gain: float = 100.0,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [lower_pct, upper_pct])
    if hi <= lo:
        max_val = float(np.max(finite, initial=0.0))
        if max_val <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        normalized = np.clip(arr / max_val, 0.0, 1.0)
    else:
        windowed = np.clip(arr, lo, hi)
        normalized = (windowed - lo) / (hi - lo)

    if tone_map == "linear":
        return normalized
    if tone_map == "gamma":
        return normalized ** gamma
    if tone_map == "log":
        return np.log1p(log_gain * normalized) / np.log1p(log_gain)

    raise ValueError(f"Unsupported tone map: {tone_map}")


def save_preview_png(
    tensor: torch.Tensor,
    path: str,
    *,
    mode: str = "intensity",
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
    tone_map: str = "gamma",
    gamma: float = 2.0,
) -> None:
    arr = render_scalar_image(tensor, mode=mode)

    if mode == "phase":
        arr_norm = (arr + np.pi) / (2 * np.pi)
        arr_norm = np.clip(arr_norm, 0.0, 1.0)
    else:
        arr_norm = normalize_preview(
            arr,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
            tone_map=tone_map,
            gamma=gamma,
        )

    arr_u8 = np.asarray(np.rint(arr_norm * 255.0), dtype=np.uint8)
    Image.fromarray(arr_u8).save(path)


def save_metrics_summary(
    batch_metrics_records: list[dict[str, Any]],
    *,
    n_rows: int,
    n_cols: int,
    path: str,
) -> None:
    sns.set_theme(style="darkgrid")
    fig, (ax_loss, ax_capture) = plt.subplots(1, 2, figsize=(14, 5))

    for batch_idx, record in enumerate(batch_metrics_records):
        tiles = record["tiles"]
        metrics = record["metrics"]
        loss = np.asarray(metrics["loss"], dtype=np.float32)
        tile_loss = np.asarray(metrics["tile_loss"], dtype=np.float32)
        capture_loss = np.asarray(metrics["capture_loss"], dtype=np.float32)
        epochs = np.arange(len(loss))
        batch_label = f"batch {batch_idx + 1}"
        loss_log = np.log10(np.clip(loss, 1e-12, None))

        total_label = "total" if batch_idx == 0 else f"total ({batch_label})"
        ax_loss.plot(epochs, loss_log, label=total_label, color="black", linewidth=2.2)

        for tile_idx, (r, c) in enumerate(tiles):
            tile_loss_log = np.log10(np.clip(tile_loss[:, tile_idx], 1e-12, None))
            ax_loss.plot(epochs, tile_loss_log, label=f"tile ({r},{c})", alpha=0.9)

        ax_capture.plot(np.arange(capture_loss.shape[1]), capture_loss[-1], label=batch_label)

    ax_loss.set_title("Log Loss by Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("log10(loss)")
    ax_loss.legend(ncol=2, fontsize="small")

    ax_capture.set_title("Final Capture Loss")
    ax_capture.set_xlabel("Capture index")
    ax_capture.set_ylabel("Loss")
    if len(batch_metrics_records) > 1:
        ax_capture.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


batch_metrics_records: list[dict[str, Any]] = []


def on_tile_complete(r: int, c: int, obj: torch.Tensor, tile_pupil: ZernikeParams, metrics: dict[str, Any]):
    # Save tensors
    save_tensor(obj, f"{OUTPUT_DIR}/tile_{r}_{c}_object.npy")
    pupil_tensor = make_zernike_pupil(tile_pupil.phase_coeffs, tile_pupil.amp_coeffs, tile_pupil.basis, tile_pupil.rad_fraction)
    save_tensor(pupil_tensor, f"{OUTPUT_DIR}/tile_{r}_{c}_pupil.npy")

    print(f"Tile ({r},{c}) done — rad_fraction: {tile_pupil.rad_fraction.item():.6f}")


def on_batch_complete(batch_tiles: list[tuple[int, int]], _: ZernikeParams, metrics: dict[str, Any]) -> None:
    batch_metrics_records.append({
        "tiles": list(batch_tiles),
        "metrics": metrics,
    })


result = solve_tiled(
    captures,
    study.kx_batch[:n_captures],
    study.ky_batch[:n_captures],
    pupil,
    roi_size=roi_size,
    upsample_ratio=upsample_ratio,
    torch_device="mps",
    on_tile_complete=on_tile_complete,
    on_batch_complete=on_batch_complete,
    tile_batch_size=4,
)

save_metrics_summary(
    batch_metrics_records,
    n_rows=n_rows,
    n_cols=n_cols,
    path=f"{OUTPUT_DIR}/reconstruction_metrics.png",
)

# Save stitched result
save_tensor(result, f"{OUTPUT_DIR}/stitched_object.npy")
save_preview_png(result, f"{OUTPUT_DIR}/stitched_object.png", mode="intensity")
print(f"Stitched result shape: {result.shape}")
