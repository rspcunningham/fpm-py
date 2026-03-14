"""Analyze how image histograms evolve across a sequence of .npy captures.

Usage:
    python -m ptych.tools.histogram_analysis /tmp/usaf/captures
    python -m ptych.tools.histogram_analysis /tmp/usaf/captures --bins 512 --output /tmp/results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray


def load_npy_files(directory: Path) -> list[tuple[str, NDArray]]:
    """Load all .npy files from directory, sorted by name. Returns (name, data) pairs."""
    files = sorted(directory.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {directory}", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        arr = np.load(f)
        # Handle complex: take amplitude
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        results.append((f.stem, arr.astype(np.float64).ravel()))

    print(f"Loaded {len(results)} images from {directory}")
    return results


def compute_histograms(
    images: list[tuple[str, NDArray]],
    bins: int,
    plo: float,
    phi: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute aligned histograms across all images using shared bin edges.

    Returns (hist_matrix [N x bins], bin_edges [bins+1], bin_centers [bins]).
    Bin range is determined by global percentiles across all images.
    """
    # Global percentile range from sampled data
    samples = []
    for _, data in images:
        finite = data[np.isfinite(data)]
        if len(finite) > 10000:
            idx = np.linspace(0, len(finite) - 1, 10000, dtype=int)
            samples.append(finite[idx])
        else:
            samples.append(finite)

    all_samples = np.concatenate(samples)
    vmin = float(np.percentile(all_samples, plo))
    vmax = float(np.percentile(all_samples, phi))
    if vmin == vmax:
        vmax = vmin + 1.0

    bin_edges = np.linspace(vmin, vmax, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist_matrix = np.zeros((len(images), bins), dtype=np.float64)
    for i, (_, data) in enumerate(images):
        finite = data[np.isfinite(data)]
        hist_matrix[i], _ = np.histogram(finite, bins=bin_edges)

    return hist_matrix, bin_edges, bin_centers


def compute_stats(images: list[tuple[str, NDArray]]) -> dict[str, NDArray]:
    """Compute per-image statistics."""
    n = len(images)
    stats: dict[str, NDArray] = {
        "mean": np.zeros(n),
        "median": np.zeros(n),
        "std": np.zeros(n),
        "skewness": np.zeros(n),
        "p2": np.zeros(n),
        "p98": np.zeros(n),
        "entropy": np.zeros(n),
    }

    for i, (_, data) in enumerate(images):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            continue

        stats["mean"][i] = np.mean(finite)
        stats["median"][i] = np.median(finite)
        stats["std"][i] = np.std(finite)
        stats["p2"][i] = np.percentile(finite, 2)
        stats["p98"][i] = np.percentile(finite, 98)

        # Skewness
        mu = stats["mean"][i]
        sigma = stats["std"][i]
        if sigma > 0:
            stats["skewness"][i] = np.mean(((finite - mu) / sigma) ** 3)

        # Shannon entropy from histogram
        h, _ = np.histogram(finite, bins=256)
        h = h[h > 0].astype(np.float64)
        p = h / h.sum()
        stats["entropy"][i] = -np.sum(p * np.log2(p))

    return stats


def compute_emd(hist_matrix: NDArray, bin_centers: NDArray) -> NDArray:
    """Earth Mover's Distance between consecutive frames.

    EMD for 1D distributions = integral of |CDF_a - CDF_b| * bin_width.
    """
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    n = hist_matrix.shape[0]
    emd = np.zeros(n - 1)

    for i in range(n - 1):
        # Normalize to probability distributions
        a = hist_matrix[i]
        b = hist_matrix[i + 1]
        sa, sb = a.sum(), b.sum()
        if sa == 0 or sb == 0:
            continue
        cdf_a = np.cumsum(a / sa)
        cdf_b = np.cumsum(b / sb)
        emd[i] = np.sum(np.abs(cdf_a - cdf_b)) * bin_width

    return emd


def plot(
    images: list[tuple[str, NDArray]],
    hist_matrix: NDArray,
    bin_edges: NDArray,
    bin_centers: NDArray,
    stats: dict[str, NDArray],
    emd: NDArray,
    output: Path,
    log_scale: bool,
) -> None:
    """Generate the analysis figure."""
    n = len(images)
    indices = np.arange(n)
    names = [name for name, _ in images]

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.5], hspace=0.3)

    # --- Row 1: Histogram kymograph ---
    ax_kymo = fig.add_subplot(gs[0])

    # Normalize each row to probability density for fair comparison
    row_sums = hist_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    hist_norm = hist_matrix / row_sums

    if log_scale:
        # Add small epsilon for log scale
        plot_data = hist_norm.copy()
        plot_data[plot_data == 0] = np.nan
        im = ax_kymo.pcolormesh(
            bin_edges,
            np.arange(n + 1) - 0.5,
            plot_data,
            cmap="magma",
            norm=LogNorm(vmin=np.nanpercentile(plot_data, 1), vmax=np.nanmax(plot_data)),
            rasterized=True,
        )
    else:
        im = ax_kymo.pcolormesh(
            bin_edges,
            np.arange(n + 1) - 0.5,
            hist_norm,
            cmap="magma",
            rasterized=True,
        )

    ax_kymo.set_ylabel("Image index")
    ax_kymo.set_xlabel("Intensity")
    ax_kymo.set_title("Histogram kymograph")
    ax_kymo.set_ylim(-0.5, n - 0.5)
    fig.colorbar(im, ax=ax_kymo, label="Density" + (" (log)" if log_scale else ""), pad=0.01)

    # Tick labels: show a subset of filenames if too many
    if n <= 30:
        ax_kymo.set_yticks(indices)
        ax_kymo.set_yticklabels(names, fontsize=6)
    else:
        step = max(1, n // 20)
        tick_idx = indices[::step]
        ax_kymo.set_yticks(tick_idx)
        ax_kymo.set_yticklabels([names[i] for i in tick_idx], fontsize=6)

    # --- Row 2: EMD between consecutive frames ---
    ax_emd = fig.add_subplot(gs[1])
    emd_indices = np.arange(len(emd)) + 0.5  # midpoints between frames
    ax_emd.fill_between(emd_indices, emd, alpha=0.4, color="#3b82f6")
    ax_emd.plot(emd_indices, emd, color="#3b82f6", linewidth=1)
    ax_emd.set_ylabel("EMD")
    ax_emd.set_xlabel("Image index")
    ax_emd.set_title("Earth Mover's Distance (consecutive frames)")
    ax_emd.set_xlim(-0.5, n - 0.5)

    # --- Row 3: Summary statistics ---
    ax_stats = fig.add_subplot(gs[2])

    color_mean = "#3b82f6"
    color_std = "#f59e0b"
    color_entropy = "#10b981"

    ax_stats.plot(indices, stats["mean"], color=color_mean, linewidth=1.2, label="Mean")
    ax_stats.fill_between(
        indices,
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.15,
        color=color_mean,
        label="±1 Std",
    )
    ax_stats.plot(indices, stats["p2"], color=color_mean, linewidth=0.5, linestyle="--", alpha=0.5, label="P2/P98")
    ax_stats.plot(indices, stats["p98"], color=color_mean, linewidth=0.5, linestyle="--", alpha=0.5)

    ax_stats.set_ylabel("Intensity", color=color_mean)
    ax_stats.tick_params(axis="y", labelcolor=color_mean)

    # Entropy on twin axis
    ax_ent = ax_stats.twinx()
    ax_ent.plot(indices, stats["entropy"], color=color_entropy, linewidth=1.2, label="Entropy")
    ax_ent.set_ylabel("Entropy (bits)", color=color_entropy)
    ax_ent.tick_params(axis="y", labelcolor=color_entropy)

    # Combined legend
    lines1, labels1 = ax_stats.get_legend_handles_labels()
    lines2, labels2 = ax_ent.get_legend_handles_labels()
    ax_stats.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    ax_stats.set_xlabel("Image index")
    ax_stats.set_title("Per-image statistics")
    ax_stats.set_xlim(-0.5, n - 0.5)

    fig.savefig(output / "histogram_analysis.png", dpi=180, bbox_inches="tight")
    print(f"Saved {output / 'histogram_analysis.png'}")
    plt.close(fig)


def save_csv(
    images: list[tuple[str, NDArray]],
    stats: dict[str, NDArray],
    emd: NDArray,
    output: Path,
) -> None:
    """Write per-image stats to CSV."""
    path = output / "histogram_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "mean", "median", "std", "skewness", "p2", "p98", "entropy", "emd_next"])
        for i, (name, _) in enumerate(images):
            writer.writerow([
                i,
                name,
                f"{stats['mean'][i]:.6f}",
                f"{stats['median'][i]:.6f}",
                f"{stats['std'][i]:.6f}",
                f"{stats['skewness'][i]:.6f}",
                f"{stats['p2'][i]:.6f}",
                f"{stats['p98'][i]:.6f}",
                f"{stats['entropy'][i]:.4f}",
                f"{emd[i]:.6f}" if i < len(emd) else "",
            ])
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze histogram evolution across a sequence of .npy image captures."
    )
    parser.add_argument("directory", type=Path, help="Directory containing .npy files")
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins (default: 256)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--percentile-range", nargs=2, type=float, default=[1, 99], metavar=("LO", "HI"),
                        help="Percentile range for bin edges (default: 1 99)")
    parser.add_argument("--linear", action="store_true", help="Use linear scale instead of log for kymograph")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")

    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)

    images = load_npy_files(args.directory)
    hist_matrix, bin_edges, bin_centers = compute_histograms(
        images, args.bins, args.percentile_range[0], args.percentile_range[1]
    )
    stats = compute_stats(images)
    emd = compute_emd(hist_matrix, bin_centers)

    plot(images, hist_matrix, bin_edges, bin_centers, stats, emd, output, log_scale=not args.linear)

    if not args.no_csv:
        save_csv(images, stats, emd, output)


if __name__ == "__main__":
    main()
