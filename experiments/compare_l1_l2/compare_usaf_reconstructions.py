import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


@dataclass(frozen=True)
class RoiSpec:
    name: str
    x: int
    y: int
    width: int
    height: int
    kind: str
    orientation: str | None = None
    note: str = ""
    bar_count: int | None = None


DEFAULT_ROIS = [
    RoiSpec(
        name="left_coarse_vertical",
        x=90,
        y=20,
        width=240,
        height=170,
        kind="bars",
        orientation="vertical",
        note="three coarse vertical bars",
        bar_count=3,
    ),
    RoiSpec(
        name="right_coarse_horizontal",
        x=840,
        y=15,
        width=150,
        height=115,
        kind="bars",
        orientation="horizontal",
        note="top-right horizontal triplet",
        bar_count=3,
    ),
    RoiSpec(
        name="right_coarse_vertical",
        x=660,
        y=15,
        width=160,
        height=150,
        kind="bars",
        orientation="vertical",
        note="top-right vertical triplet",
        bar_count=3,
    ),
    RoiSpec(
        name="center_background",
        x=350,
        y=330,
        width=260,
        height=260,
        kind="background",
        note="empty field region",
    ),
    RoiSpec(
        name="center_vertical_seam",
        x=452,
        y=320,
        width=120,
        height=320,
        kind="seam",
        orientation="vertical",
        note="crosses stitched vertical seam",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare L1 and L2 stitched USAF reconstructions with fixed ROI diagnostics."
    )
    parser.add_argument(
        "--l1-path",
        type=Path,
        default=Path("demo/real/output/stitched_object.npy"),
    )
    parser.add_argument(
        "--l2-path",
        type=Path,
        default=Path("research/demo_real_l1_vs_l2/output_amp_l2/stitched_object.npy"),
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("research/demo_real_l1_vs_l2/l1_vs_l2_usaf_analysis.png"),
        help="Optional output path for the comparison figure.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("research/demo_real_l1_vs_l2/l1_vs_l2_usaf_metrics.csv"),
        help="Optional output path for the metrics CSV.",
    )
    return parser.parse_args()


def load_intensity(path: Path) -> np.ndarray:
    field = np.load(path)
    return np.abs(field).astype(np.float32) ** 2


def render_pair_preview(l1: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([l1.ravel(), l2.ravel()]).astype(np.float32)
    lo, hi = np.percentile(stacked, [0.5, 99.5])
    if hi <= lo:
        zero = np.zeros_like(l1, dtype=np.float32)
        return zero, zero

    def normalize(arr: np.ndarray) -> np.ndarray:
        preview = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return preview ** 2.0

    return normalize(l1), normalize(l2)


def crop(arr: np.ndarray, roi: RoiSpec) -> np.ndarray:
    return arr[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width]


def compute_profile(arr: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "vertical":
        return arr.mean(axis=0)
    if orientation == "horizontal":
        return arr.mean(axis=1)
    raise ValueError(f"Unsupported orientation: {orientation}")


def mean_abs_gradient(profile: np.ndarray) -> float:
    if profile.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(profile))))


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float32)

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def blur_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma)
    radius = kernel.size // 2
    padded = np.pad(profile, (radius,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def make_bar_template(length: int, start: float, bar_width: float, bar_count: int) -> np.ndarray:
    coords = np.arange(length, dtype=np.float32)
    template = np.ones(length, dtype=np.float32)
    for idx in range(bar_count):
        left = start + 2.0 * idx * bar_width
        right = left + bar_width
        mask = (coords >= left) & (coords < right)
        template[mask] = 0.0
    return template


def fit_linear_template(target: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([target, np.ones_like(target)])
    coeffs, *_ = np.linalg.lstsq(design, observed, rcond=None)
    scale = float(coeffs[0])
    offset = float(coeffs[1])
    return scale, offset


def fit_bar_profile(profile: np.ndarray, bar_count: int) -> dict[str, float | np.ndarray]:
    length = profile.size
    total_width_min = 0.35 * length
    total_width_max = 0.85 * length
    total_widths = np.linspace(total_width_min, total_width_max, 36)
    sigmas = np.linspace(0.0, 8.0, 17)

    best: dict[str, float | np.ndarray] | None = None
    for total_width in total_widths:
        bar_width = total_width / (2 * bar_count - 1)
        max_start = length - total_width
        if bar_width < 1.0 or max_start < 0.0:
            continue

        start_candidates = np.linspace(0.0, max_start, max(8, int(max_start // 3) + 1))
        for start in start_candidates:
            template = make_bar_template(length, float(start), float(bar_width), bar_count)
            for sigma in sigmas:
                blurred = blur_profile(template, float(sigma))
                scale, offset = fit_linear_template(blurred, profile)
                if scale < 0.0:
                    continue

                fitted = scale * blurred + offset
                rmse = float(np.sqrt(np.mean((fitted - profile) ** 2)))
                if best is None or rmse < best["fit_rmse"]:
                    best = {
                        "fit_rmse": rmse,
                        "fit_sigma": float(sigma),
                        "fit_bar_width": float(bar_width),
                        "fit_start": float(start),
                        "fit_scale": scale,
                        "fit_offset": offset,
                        "fit_profile": fitted,
                        "fit_template": blurred,
                    }

    if best is None:
        raise RuntimeError("Failed to fit blurred bar template.")
    return best


def summarize_bar_roi(arr: np.ndarray, orientation: str, bar_count: int) -> dict[str, float | np.ndarray]:
    profile = compute_profile(arr, orientation)
    dark_mean, bright_mean = np.percentile(profile, [10, 90])
    modulation = float((bright_mean - dark_mean) / (bright_mean + dark_mean + 1e-12))
    fit = fit_bar_profile(profile, bar_count)
    contrast = max(float(bright_mean - dark_mean), 1e-12)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "dark_mean": float(dark_mean),
        "bright_mean": float(bright_mean),
        "modulation": modulation,
        "profile_grad": mean_abs_gradient(profile),
        "fit_rmse": float(fit["fit_rmse"]),
        "fit_nrmse": float(float(fit["fit_rmse"]) / contrast),
        "fit_sigma": float(fit["fit_sigma"]),
        "fit_bar_width": float(fit["fit_bar_width"]),
        "fit_start": float(fit["fit_start"]),
        "fit_scale": float(fit["fit_scale"]),
        "fit_offset": float(fit["fit_offset"]),
        "profile": profile,
        "fit_profile": np.asarray(fit["fit_profile"], dtype=np.float32),
    }


def summarize_background_roi(arr: np.ndarray) -> dict[str, float]:
    p1, p99 = np.percentile(arr, [1, 99])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p1": float(p1),
        "p99": float(p99),
        "spread_99_1": float(p99 - p1),
    }


def summarize_seam_roi(arr: np.ndarray, orientation: str) -> dict[str, float]:
    if orientation == "vertical":
        center = arr.shape[1] // 2
        seam_jump = float(np.mean(np.abs(arr[:, center - 1] - arr[:, center])))
        baseline = np.abs(np.diff(arr, axis=1))
    elif orientation == "horizontal":
        center = arr.shape[0] // 2
        seam_jump = float(np.mean(np.abs(arr[center - 1, :] - arr[center, :])))
        baseline = np.abs(np.diff(arr, axis=0))
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    baseline_mean = float(np.mean(baseline))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "seam_jump": seam_jump,
        "seam_ratio": float(seam_jump / (baseline_mean + 1e-12)),
    }


def summarize_global_seams(arr: np.ndarray) -> dict[str, float]:
    y_mid = arr.shape[0] // 2
    x_mid = arr.shape[1] // 2
    vertical_jump = float(np.mean(np.abs(arr[:, x_mid - 1] - arr[:, x_mid])))
    horizontal_jump = float(np.mean(np.abs(arr[y_mid - 1, :] - arr[y_mid, :])))
    return {
        "vertical_jump": vertical_jump,
        "horizontal_jump": horizontal_jump,
    }


def compare_metrics(l1: dict[str, float], l2: dict[str, float]) -> dict[str, float]:
    compared: dict[str, float] = {}
    for key in l1:
        if key in l2 and np.isscalar(l1[key]) and np.isscalar(l2[key]):
            compared[f"l1_{key}"] = l1[key]
            compared[f"l2_{key}"] = l2[key]
            compared[f"delta_{key}"] = l2[key] - l1[key]
    return compared


def build_row(roi: RoiSpec, l1_metrics: dict[str, float], l2_metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "roi": roi.name,
        "kind": roi.kind,
        "orientation": roi.orientation or "",
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
        "note": roi.note,
    }
    row.update(compare_metrics(l1_metrics, l2_metrics))
    for key in ("profile", "fit_profile"):
        if key in l1_metrics and key in l2_metrics:
            row[f"l1_{key}"] = l1_metrics[key]
            row[f"l2_{key}"] = l2_metrics[key]
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    scalar_rows: list[dict[str, Any]] = []
    for row in rows:
        scalar_row = {
            key: value
            for key, value in row.items()
            if np.isscalar(value) or isinstance(value, str)
        }
        scalar_rows.append(scalar_row)

    fieldnames: list[str] = []
    for row in scalar_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def plot_analysis(
    l1: np.ndarray,
    l2: np.ndarray,
    rois: list[RoiSpec],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    n_rows = len(rois) + 1
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15, 3.2 * n_rows),
        gridspec_kw={"height_ratios": [1.5] + [1.0] * len(rois)},
    )

    full_preview_l1, _ = render_pair_preview(l1, l2)
    ax_full = axes[0, 0]
    ax_full.imshow(full_preview_l1, cmap="gray", vmin=0.0, vmax=1.0)
    ax_full.set_title("L1 Reference With Analysis ROIs")
    ax_full.set_axis_off()
    for idx, roi in enumerate(rois, start=1):
        rect = patches.Rectangle((roi.x, roi.y), roi.width, roi.height, linewidth=1.5, edgecolor=f"C{idx % 10}", facecolor="none")
        ax_full.add_patch(rect)
        ax_full.text(roi.x + 4, roi.y + 16, str(idx), color=f"C{idx % 10}", fontsize=10, weight="bold")

    ax_text = axes[0, 1]
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            f"{idx}. {roi.name}: {roi.note}"
            for idx, roi in enumerate(rois, start=1)
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    global_l1 = summarize_global_seams(l1)
    global_l2 = summarize_global_seams(l2)
    ax_global = axes[0, 2]
    ax_global.axis("off")
    ax_global.text(
        0.0,
        1.0,
        "\n".join(
            [
                "Global seam diagnostics",
                f"L1 vertical jump: {global_l1['vertical_jump']:.4f}",
                f"L2 vertical jump: {global_l2['vertical_jump']:.4f}",
                f"Delta vertical jump: {global_l2['vertical_jump'] - global_l1['vertical_jump']:.4f}",
                "",
                f"L1 horizontal jump: {global_l1['horizontal_jump']:.4f}",
                f"L2 horizontal jump: {global_l2['horizontal_jump']:.4f}",
                f"Delta horizontal jump: {global_l2['horizontal_jump'] - global_l1['horizontal_jump']:.4f}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    row_lookup = {row["roi"]: row for row in rows}
    for idx, roi in enumerate(rois, start=1):
        ax_l1, ax_l2, ax_diag = axes[idx]
        l1_crop = crop(l1, roi)
        l2_crop = crop(l2, roi)
        l1_preview, l2_preview = render_pair_preview(l1_crop, l2_crop)

        ax_l1.imshow(l1_preview, cmap="gray", vmin=0.0, vmax=1.0)
        ax_l1.set_title(f"{idx}. {roi.name} | L1")
        ax_l1.set_axis_off()

        ax_l2.imshow(l2_preview, cmap="gray", vmin=0.0, vmax=1.0)
        ax_l2.set_title(f"{idx}. {roi.name} | L2")
        ax_l2.set_axis_off()

        row = row_lookup[roi.name]
        if roi.kind == "bars":
            l1_profile = np.asarray(row["l1_profile"], dtype=np.float32)
            l2_profile = np.asarray(row["l2_profile"], dtype=np.float32)
            l1_fit_profile = np.asarray(row["l1_fit_profile"], dtype=np.float32)
            l2_fit_profile = np.asarray(row["l2_fit_profile"], dtype=np.float32)
            ax_diag.plot(l1_profile, label="L1", linewidth=2.0)
            ax_diag.plot(l2_profile, label="L2", linewidth=2.0)
            ax_diag.plot(l1_fit_profile, label="L1 fit", linewidth=1.6, linestyle="--")
            ax_diag.plot(l2_fit_profile, label="L2 fit", linewidth=1.6, linestyle="--")
            ax_diag.set_title("Bar Profile With Fitted Blurred Target")
            ax_diag.set_xlabel("Profile index")
            ax_diag.set_ylabel("Mean intensity")
            ax_diag.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"L1 fit RMSE: {row['l1_fit_rmse']:.4f}",
                        f"L2 fit RMSE: {row['l2_fit_rmse']:.4f}",
                        f"Delta RMSE: {row['delta_fit_rmse']:.4f}",
                        f"L1 sigma: {row['l1_fit_sigma']:.2f}",
                        f"L2 sigma: {row['l2_fit_sigma']:.2f}",
                        f"L1 modulation: {row['l1_modulation']:.3f}",
                        f"L2 modulation: {row['l2_modulation']:.3f}",
                    ]
                ),
                transform=ax_diag.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax_diag.legend(loc="lower right")
        elif roi.kind == "background":
            bins = 40
            ax_diag.hist(l1_crop.ravel(), bins=bins, alpha=0.6, label="L1")
            ax_diag.hist(l2_crop.ravel(), bins=bins, alpha=0.6, label="L2")
            ax_diag.set_title("Background Intensity Histogram")
            ax_diag.set_xlabel("Intensity")
            ax_diag.set_ylabel("Count")
            ax_diag.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"L1 std: {row['l1_std']:.4f}",
                        f"L2 std: {row['l2_std']:.4f}",
                        f"Delta std: {row['delta_std']:.4f}",
                        f"L1 spread: {row['l1_spread_99_1']:.4f}",
                        f"L2 spread: {row['l2_spread_99_1']:.4f}",
                    ]
                ),
                transform=ax_diag.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax_diag.legend(loc="upper right")
        elif roi.kind == "seam":
            l1_profile = compute_profile(l1_crop, roi.orientation or "vertical")
            l2_profile = compute_profile(l2_crop, roi.orientation or "vertical")
            seam_idx = l1_profile.size // 2
            ax_diag.plot(l1_profile, label="L1", linewidth=2.0)
            ax_diag.plot(l2_profile, label="L2", linewidth=2.0)
            ax_diag.axvline(seam_idx, color="black", linestyle="--", alpha=0.6)
            ax_diag.set_title("Seam Cross-Section")
            ax_diag.set_xlabel("Profile index")
            ax_diag.set_ylabel("Mean intensity")
            ax_diag.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"L1 seam jump: {row['l1_seam_jump']:.4f}",
                        f"L2 seam jump: {row['l2_seam_jump']:.4f}",
                        f"Delta seam jump: {row['delta_seam_jump']:.4f}",
                        f"L1 seam ratio: {row['l1_seam_ratio']:.3f}",
                        f"L2 seam ratio: {row['l2_seam_ratio']:.3f}",
                    ]
                ),
                transform=ax_diag.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax_diag.legend(loc="lower right")
        else:
            raise ValueError(f"Unsupported ROI kind: {roi.kind}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    l1_path = args.l1_path
    l2_path = args.l2_path
    figure_path = args.figure_path
    csv_path = args.csv_path

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    l1 = load_intensity(l1_path)
    l2 = load_intensity(l2_path)

    rows: list[dict[str, Any]] = []
    for roi in DEFAULT_ROIS:
        l1_crop = crop(l1, roi)
        l2_crop = crop(l2, roi)

        if roi.kind == "bars":
            l1_metrics = summarize_bar_roi(l1_crop, roi.orientation or "vertical", roi.bar_count or 3)
            l2_metrics = summarize_bar_roi(l2_crop, roi.orientation or "vertical", roi.bar_count or 3)
        elif roi.kind == "background":
            l1_metrics = summarize_background_roi(l1_crop)
            l2_metrics = summarize_background_roi(l2_crop)
        elif roi.kind == "seam":
            l1_metrics = summarize_seam_roi(l1_crop, roi.orientation or "vertical")
            l2_metrics = summarize_seam_roi(l2_crop, roi.orientation or "vertical")
        else:
            raise ValueError(f"Unsupported ROI kind: {roi.kind}")

        rows.append(build_row(roi, l1_metrics, l2_metrics))

    write_csv(rows, csv_path)
    plot_analysis(l1, l2, DEFAULT_ROIS, rows, figure_path)

    print(f"Wrote figure: {figure_path}")
    print(f"Wrote metrics: {csv_path}")
    for row in rows:
        print(
            f"{row['roi']}: "
            f"delta_mean={row.get('delta_mean', float('nan')):.4f}, "
            f"delta_std={row.get('delta_std', float('nan')):.4f}, "
            f"delta_modulation={row.get('delta_modulation', float('nan')):.4f}, "
            f"delta_fit_rmse={row.get('delta_fit_rmse', float('nan')):.4f}, "
            f"delta_seam_jump={row.get('delta_seam_jump', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
