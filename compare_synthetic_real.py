import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptych import PtychStudy


REAL_DATASET = "usaf-test"
SYNTHETIC_DATASET_ROOT = Path("results/synthetic_usaf_test")
OUTPUT_DIR = Path("results/compare_synthetic_real")


def _center_crop_bounds(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> tuple[slice, slice]:
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    if target_h > source_h or target_w > source_w:
        raise ValueError(
            f"Target shape {target_shape} exceeds source shape {source_shape}; cannot center-crop."
        )
    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2
    return slice(top, top + target_h), slice(left, left + target_w)


def _pair_metrics(real_image: np.ndarray, synthetic_image: np.ndarray) -> dict[str, float]:
    real_flat = real_image.reshape(-1).astype(np.float64)
    synthetic_flat = synthetic_image.reshape(-1).astype(np.float64)
    diff = synthetic_flat - real_flat

    bias = float(diff.mean())
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    max_abs = float(np.max(np.abs(diff)))

    real_mean = float(real_flat.mean())
    real_std = float(real_flat.std())
    synthetic_mean = float(synthetic_flat.mean())
    synthetic_std = float(synthetic_flat.std())

    real_centered = real_flat - real_mean
    synthetic_centered = synthetic_flat - synthetic_mean
    denom = np.sqrt(np.sum(real_centered ** 2) * np.sum(synthetic_centered ** 2))
    corrcoef = float(np.sum(real_centered * synthetic_centered) / denom) if denom > 0 else float("nan")

    design = np.column_stack([synthetic_flat, np.ones_like(synthetic_flat)])
    scale, offset = np.linalg.lstsq(design, real_flat, rcond=None)[0]
    synthetic_affine = scale * synthetic_flat + offset
    affine_diff = synthetic_affine - real_flat
    affine_mae = float(np.mean(np.abs(affine_diff)))
    affine_rmse = float(np.sqrt(np.mean(np.square(affine_diff))))

    return {
        "real_mean": real_mean,
        "real_std": real_std,
        "real_min": float(real_flat.min()),
        "real_max": float(real_flat.max()),
        "synthetic_mean": synthetic_mean,
        "synthetic_std": synthetic_std,
        "synthetic_min": float(synthetic_flat.min()),
        "synthetic_max": float(synthetic_flat.max()),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "corrcoef": corrcoef,
        "affine_scale": float(scale),
        "affine_offset": float(offset),
        "affine_mae": affine_mae,
        "affine_rmse": affine_rmse,
    }


def _save_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No metric rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_summary_json(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    payload = {
        "summary": summary,
        "best_corr_frame": max(rows, key=lambda row: row["corrcoef"]),
        "worst_rmse_frame": max(rows, key=lambda row: row["rmse"]),
        "worst_affine_rmse_frame": max(rows, key=lambda row: row["affine_rmse"]),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _save_overview_plot(rows: list[dict[str, Any]], path: Path) -> None:
    indices = [row["capture_index"] for row in rows]
    rmse = [row["rmse"] for row in rows]
    affine_rmse = [row["affine_rmse"] for row in rows]
    corrcoef = [row["corrcoef"] for row in rows]
    real_mean = [row["real_mean"] for row in rows]
    synthetic_mean = [row["synthetic_mean"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(indices, rmse, label="RMSE")
    axes[0, 0].plot(indices, affine_rmse, label="Affine RMSE")
    axes[0, 0].set_title("Per-Capture Error")
    axes[0, 0].set_xlabel("Capture Index")
    axes[0, 0].set_ylabel("Error")
    axes[0, 0].legend()

    axes[0, 1].plot(indices, corrcoef, color="tab:green")
    axes[0, 1].set_title("Per-Capture Correlation")
    axes[0, 1].set_xlabel("Capture Index")
    axes[0, 1].set_ylabel("Correlation")
    axes[0, 1].set_ylim(-1.0, 1.0)

    axes[1, 0].scatter(real_mean, synthetic_mean, s=16, alpha=0.8)
    axes[1, 0].set_title("Mean Intensity: Real vs Synthetic")
    axes[1, 0].set_xlabel("Real Mean")
    axes[1, 0].set_ylabel("Synthetic Mean")

    axes[1, 1].hist(rmse, bins=20, alpha=0.7, label="RMSE")
    axes[1, 1].hist(affine_rmse, bins=20, alpha=0.7, label="Affine RMSE")
    axes[1, 1].set_title("Error Distribution")
    axes[1, 1].set_xlabel("Error")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_frame_panel(
    capture_index: int,
    filename: str,
    real_image: np.ndarray,
    synthetic_image: np.ndarray,
    metrics: dict[str, float],
    path: Path,
) -> None:
    diff = synthetic_image - real_image
    image_vmin = float(min(real_image.min(), synthetic_image.min()))
    image_vmax = float(max(real_image.max(), synthetic_image.max()))
    diff_limit = float(np.quantile(np.abs(diff), 0.995))
    diff_limit = max(diff_limit, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Capture {capture_index:03d} ({filename}) | "
        f"RMSE={metrics['rmse']:.4f} Corr={metrics['corrcoef']:.4f} "
        f"AffineRMSE={metrics['affine_rmse']:.4f}"
    )

    im0 = axes[0].imshow(real_image, cmap="gray", vmin=image_vmin, vmax=image_vmax)
    axes[0].set_title("Real (center crop)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(synthetic_image, cmap="gray", vmin=image_vmin, vmax=image_vmax)
    axes[1].set_title("Synthetic")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-diff_limit, vmax=diff_limit)
    axes[2].set_title("Synthetic - Real")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    real_study = PtychStudy.load(REAL_DATASET)
    synthetic_study = PtychStudy.from_disk(SYNTHETIC_DATASET_ROOT)
    real_data_dir = Path.home() / ".cache" / "ptych" / "datasets" / REAL_DATASET / "data"
    synthetic_data_dir = SYNTHETIC_DATASET_ROOT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_dir = OUTPUT_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    real_manifest = real_study.manifest
    synthetic_manifest = synthetic_study.manifest
    real_filenames = [capture.filename for capture in real_manifest.captures if capture.led_positions]
    synthetic_filenames = [capture.filename for capture in synthetic_manifest.captures if capture.led_positions]

    if real_filenames != synthetic_filenames:
        raise ValueError("Real and synthetic capture filenames are not aligned.")

    real_shape = (
        int(real_study.captures.shape[-2]),
        int(real_study.captures.shape[-1]),
    )
    synthetic_shape = (
        int(synthetic_study.captures.shape[-2]),
        int(synthetic_study.captures.shape[-1]),
    )
    crop_rows, crop_cols = _center_crop_bounds(real_shape, synthetic_shape)

    rows: list[dict[str, Any]] = []
    cached_images: dict[int, tuple[np.ndarray, np.ndarray, dict[str, float], str]] = {}

    for capture_index, filename in enumerate(real_filenames):
        real_image = np.asarray(real_study.captures[capture_index].detach().cpu().numpy(), dtype=np.float32)
        synthetic_image = np.asarray(synthetic_study.captures[capture_index].detach().cpu().numpy(), dtype=np.float32)
        real_image = real_image[crop_rows, crop_cols]
        if real_image.shape != synthetic_image.shape:
            raise ValueError(
                f"Shape mismatch for {filename}: real crop {real_image.shape}, synthetic {synthetic_image.shape}"
            )

        metrics = _pair_metrics(real_image, synthetic_image)
        row: dict[str, Any] = {
            "capture_index": capture_index,
            "filename": filename,
            **metrics,
        }
        rows.append(row)
        cached_images[capture_index] = (real_image, synthetic_image, metrics, filename)

    summary = {
        "real_dataset_dir": str(real_data_dir),
        "synthetic_dataset_dir": str(synthetic_data_dir),
        "comparison_crop": {
            "top": crop_rows.start,
            "bottom": crop_rows.stop,
            "left": crop_cols.start,
            "right": crop_cols.stop,
            "shape": list(synthetic_shape),
        },
        "capture_count": len(rows),
        "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "median_rmse": float(np.median([row["rmse"] for row in rows])),
        "mean_affine_rmse": float(np.mean([row["affine_rmse"] for row in rows])),
        "median_affine_rmse": float(np.median([row["affine_rmse"] for row in rows])),
        "mean_corrcoef": float(np.mean([row["corrcoef"] for row in rows])),
        "median_corrcoef": float(np.median([row["corrcoef"] for row in rows])),
    }

    _save_metrics_csv(rows, OUTPUT_DIR / "metrics.csv")
    _save_summary_json(rows, summary, OUTPUT_DIR / "summary.json")
    _save_overview_plot(rows, OUTPUT_DIR / "overview.png")

    representative_indices = {0, len(rows) // 2, len(rows) - 1}
    representative_indices.update(
        row["capture_index"] for row in sorted(rows, key=lambda row: row["rmse"], reverse=True)[:3]
    )
    representative_indices.update(
        row["capture_index"] for row in sorted(rows, key=lambda row: row["corrcoef"])[:3]
    )

    for capture_index in sorted(representative_indices):
        real_image, synthetic_image, metrics, filename = cached_images[capture_index]
        _save_frame_panel(
            capture_index=capture_index,
            filename=filename,
            real_image=real_image,
            synthetic_image=synthetic_image,
            metrics=metrics,
            path=frame_dir / f"capture_{capture_index:03d}.png",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
