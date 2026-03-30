from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
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
    path: Path,
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
    batch_metrics: list[dict[str, object]],
    *,
    path: Path,
) -> None:
    sns.set_theme(style="darkgrid")
    fig, (ax_loss, ax_capture) = plt.subplots(1, 2, figsize=(14, 5))

    for batch_idx, record in enumerate(batch_metrics):
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
    if len(batch_metrics) > 1:
        ax_capture.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
