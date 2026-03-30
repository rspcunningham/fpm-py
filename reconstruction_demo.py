from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from ptych import PtychStudy, solve_tiled
from ptych.core.pupil import ZernikeParams, make_ideal_pupil, make_zernike_pupil
from ptych.data.bayer import demosaic
from preview_utils import save_preview_png, save_tensor


# Dataset and output settings
study = PtychStudy.load("usaf_test")
OUTPUT_DIR = Path("./results")

# Reconstruction geometry settings
ROI_SIZE = 64
CROP_SIZE = 256
N_CAPTURES = 61

UPSAMPLE_RATIO = 8
NA = 0.13  # Used to generate the initial pupil guess; still a free parameter.
NUM_PHASE_TERMS = 20
NUM_AMP_TERMS = 20

# Optimization and runtime settings
TORCH_DEVICE = "mps"  # Switch to "cpu" or "cuda".
TILE_BATCH_SIZE = 16
EPOCHS = 150

TileCompleteCallback = Callable[[int, int, torch.Tensor, ZernikeParams, dict[str, Any]], None]
BatchCompleteCallback = Callable[[list[tuple[int, int]], ZernikeParams, dict[str, Any]], None]


# Plotting and artifact helpers
def save_metrics_summary(
    batch_metrics_records: list[dict[str, Any]],
    *,
    path: Path,
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
    plt.close(fig)


# Reconstruction-specific data preparation
def prepare_captures(
    study: PtychStudy,
    *,
    n_captures: int,
    crop_size: int,
) -> torch.Tensor:
    _, height, width = study.captures.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    captures = study.captures[:n_captures, top:top + crop_size, left:left + crop_size]
    captures = demosaic(captures)[:, 1]
    return captures / captures.max()


def build_initial_pupil(
    study: PtychStudy,
    *,
    roi_size: int,
    upsample_ratio: int,
    na: float,
    num_phase_terms: int,
    num_amp_terms: int,
) -> ZernikeParams:
    return make_ideal_pupil(
        N=roi_size * upsample_ratio,
        NA=na,
        wavelength_m=study.manifest.captures[0].wavelength,
        sensor_pixel_size_m=study.manifest.sensor_pixel_size,
        magnification=study.manifest.magnification,
        downsample_ratio=upsample_ratio,
        num_phase_terms=num_phase_terms,
        num_amp_terms=num_amp_terms,
    )


def build_output_callbacks(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], TileCompleteCallback, BatchCompleteCallback]:
    batch_metrics_records: list[dict[str, Any]] = []

    def on_tile_complete(
        r: int,
        c: int,
        obj: torch.Tensor,
        tile_pupil: ZernikeParams,
        _metrics: dict[str, Any],
    ) -> None:
        # Save per-tile object and pupil artifacts as soon as each tile completes.
        save_tensor(obj, output_dir / f"tile_{r}_{c}_object.npy")
        pupil_tensor = make_zernike_pupil(
            tile_pupil.phase_coeffs,
            tile_pupil.amp_coeffs,
            tile_pupil.basis,
            tile_pupil.rad_fraction,
        )
        save_tensor(pupil_tensor, output_dir / f"tile_{r}_{c}_pupil.npy")

        print(f"Tile ({r},{c}) done; rad_fraction: {tile_pupil.rad_fraction.item():.6f}")

    def on_batch_complete(
        batch_tiles: list[tuple[int, int]],
        _pupil: ZernikeParams,
        metrics: dict[str, Any],
    ) -> None:
        # Accumulate per-batch metrics for the summary plot at the end.
        batch_metrics_records.append({
            "tiles": list(batch_tiles),
            "metrics": metrics,
        })

    return batch_metrics_records, on_tile_complete, on_batch_complete


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare study data.
    captures = prepare_captures(
        study,
        n_captures=N_CAPTURES,
        crop_size=CROP_SIZE,
    )
    pupil = build_initial_pupil(
        study,
        roi_size=ROI_SIZE,
        upsample_ratio=UPSAMPLE_RATIO,
        na=NA,
        num_phase_terms=NUM_PHASE_TERMS,
        num_amp_terms=NUM_AMP_TERMS,
    )
    batch_metrics_records, on_tile_complete, on_batch_complete = build_output_callbacks(
        OUTPUT_DIR
    )

    # Run tiled reconstruction.
    result = solve_tiled(
        captures,
        study.kx_batch[:N_CAPTURES],
        study.ky_batch[:N_CAPTURES],
        pupil,
        roi_size=ROI_SIZE,
        upsample_ratio=UPSAMPLE_RATIO,
        epochs=EPOCHS,
        torch_device=TORCH_DEVICE,
        on_tile_complete=on_tile_complete,
        on_batch_complete=on_batch_complete,
        tile_batch_size=TILE_BATCH_SIZE,
    )

    # Save reconstruction artifacts.
    save_metrics_summary(
        batch_metrics_records,
        path=OUTPUT_DIR / "reconstruction_metrics.png",
    )

    save_tensor(result, OUTPUT_DIR / "stitched_object.npy")
    save_preview_png(result, OUTPUT_DIR / "stitched_object.png", mode="intensity")
    print(f"Stitched result shape: {result.shape}")


if __name__ == "__main__":
    main()
