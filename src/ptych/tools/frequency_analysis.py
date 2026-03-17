"""Analyze how radial power spectra evolve across a sequence of .npy captures.

Usage:
    python -m ptych.tools.frequency_analysis /tmp/usaf/captures
    python -m ptych.tools.frequency_analysis /tmp/usaf/captures --bins 256 --output /tmp/results/
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
    """Load all .npy files from directory, sorted by name. Returns (name, 2D array) pairs."""
    files = sorted(directory.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {directory}", file=sys.stderr)
        sys.exit(1)

    results = []
    for f in files:
        arr = np.load(f)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        arr = arr.astype(np.float64)
        if arr.ndim != 2:
            print(f"  Skipping {f.name} (ndim={arr.ndim}, expected 2)")
            continue
        results.append((f.stem, arr))

    print(f"Loaded {len(results)} images from {directory}")
    return results


def radial_power_spectrum(img: NDArray, n_bins: int) -> tuple[NDArray, NDArray]:
    """Compute radial power spectrum of a 2D image.

    Returns (bin_centers, power) where bin_centers are spatial frequencies
    normalized to [0, 1] (fraction of Nyquist).
    """
    h, w = img.shape

    # Windowing to reduce spectral leakage
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = np.outer(wy, wx)
    img_windowed = (img - img.mean()) * window

    # 2D FFT → power spectrum
    fft = np.fft.fft2(img_windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    # Radial frequency map
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Normalize radius: max meaningful frequency is min(h, w) / 2
    r_max = min(h, w) / 2
    r_norm = r / r_max  # 0 to ~1 (corners go beyond 1)

    # Bin into radial shells
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    radial_power = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (r_norm >= bin_edges[i]) & (r_norm < bin_edges[i + 1])
        if mask.any():
            radial_power[i] = power[mask].mean()

    return bin_centers, radial_power


def compute_all_spectra(
    images: list[tuple[str, NDArray]], n_bins: int
) -> tuple[NDArray, NDArray]:
    """Compute radial power spectra for all images.

    Returns (spectrum_matrix [N x n_bins], freq_centers [n_bins]).
    """
    freq_centers = None
    spectra = []

    for _, img in images:
        centers, power = radial_power_spectrum(img, n_bins)
        if freq_centers is None:
            freq_centers = centers
        spectra.append(power)

    return np.array(spectra), freq_centers  # type: ignore[return-value]


def compute_stats(
    spectrum_matrix: NDArray, freq_centers: NDArray
) -> dict[str, NDArray]:
    """Compute per-image frequency statistics."""
    n = spectrum_matrix.shape[0]
    stats: dict[str, NDArray] = {
        "spectral_centroid": np.zeros(n),
        "cutoff_freq": np.zeros(n),
        "low_band_energy": np.zeros(n),
        "mid_band_energy": np.zeros(n),
        "high_band_energy": np.zeros(n),
        "total_energy": np.zeros(n),
    }

    # Band boundaries (fraction of Nyquist)
    low_hi = 0.15
    mid_hi = 0.5

    for i in range(n):
        power = spectrum_matrix[i]
        total = power.sum()
        stats["total_energy"][i] = total

        if total > 0:
            # Spectral centroid
            stats["spectral_centroid"][i] = np.sum(freq_centers * power) / total

            # Band energies (fraction of total)
            low_mask = freq_centers < low_hi
            mid_mask = (freq_centers >= low_hi) & (freq_centers < mid_hi)
            high_mask = freq_centers >= mid_hi
            stats["low_band_energy"][i] = power[low_mask].sum() / total
            stats["mid_band_energy"][i] = power[mid_mask].sum() / total
            stats["high_band_energy"][i] = power[high_mask].sum() / total

            # Cutoff frequency: where power drops below 1% of peak
            peak = power.max()
            above_threshold = freq_centers[power > peak * 0.01]
            stats["cutoff_freq"][i] = above_threshold[-1] if len(above_threshold) > 0 else 0

    return stats


def compute_spectral_emd(spectrum_matrix: NDArray, freq_centers: NDArray) -> NDArray:
    """EMD between consecutive radial power spectra."""
    df = freq_centers[1] - freq_centers[0] if len(freq_centers) > 1 else 1.0
    n = spectrum_matrix.shape[0]
    emd = np.zeros(n - 1)

    for i in range(n - 1):
        a = spectrum_matrix[i]
        b = spectrum_matrix[i + 1]
        sa, sb = a.sum(), b.sum()
        if sa == 0 or sb == 0:
            continue
        cdf_a = np.cumsum(a / sa)
        cdf_b = np.cumsum(b / sb)
        emd[i] = np.sum(np.abs(cdf_a - cdf_b)) * df

    return emd


def plot(
    images: list[tuple[str, NDArray]],
    spectrum_matrix: NDArray,
    freq_centers: NDArray,
    stats: dict[str, NDArray],
    emd: NDArray,
    output: Path,
) -> None:
    """Generate the analysis figure."""
    n = len(images)
    indices = np.arange(n)
    names = [name for name, _ in images]

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.5], hspace=0.3)

    # --- Row 1: Radial power spectrum kymograph ---
    ax_kymo = fig.add_subplot(gs[0])

    plot_data = spectrum_matrix.copy()
    plot_data[plot_data <= 0] = np.nan
    vmin = np.nanpercentile(plot_data, 5)
    vmax = np.nanpercentile(plot_data, 99.5)

    freq_edges = np.linspace(0, 1, len(freq_centers) + 1)
    idx_edges = np.arange(n + 1) - 0.5

    im = ax_kymo.pcolormesh(
        freq_edges, idx_edges, plot_data,
        cmap="inferno",
        norm=LogNorm(vmin=max(vmin, 1e-20), vmax=vmax),
        rasterized=True,
    )
    ax_kymo.set_ylabel("Image index")
    ax_kymo.set_xlabel("Spatial frequency (fraction of Nyquist)")
    ax_kymo.set_title("Radial power spectrum kymograph")
    ax_kymo.set_ylim(-0.5, n - 0.5)
    fig.colorbar(im, ax=ax_kymo, label="Power (log)", pad=0.01)

    if n <= 30:
        ax_kymo.set_yticks(indices)
        ax_kymo.set_yticklabels(names, fontsize=6)
    else:
        step = max(1, n // 20)
        tick_idx = indices[::step]
        ax_kymo.set_yticks(tick_idx)
        ax_kymo.set_yticklabels([names[i] for i in tick_idx], fontsize=6)

    # --- Row 2: Cutoff frequency + spectral EMD ---
    ax_cut = fig.add_subplot(gs[1])
    ax_cut.plot(indices, stats["cutoff_freq"], color="#3b82f6", linewidth=1.2, label="Cutoff freq")
    ax_cut.set_ylabel("Cutoff freq", color="#3b82f6")
    ax_cut.tick_params(axis="y", labelcolor="#3b82f6")
    ax_cut.set_title("Cutoff frequency & spectral EMD")
    ax_cut.set_xlim(-0.5, n - 0.5)

    ax_emd = ax_cut.twinx()
    emd_indices = np.arange(len(emd)) + 0.5
    ax_emd.fill_between(emd_indices, emd, alpha=0.2, color="#f59e0b")
    ax_emd.plot(emd_indices, emd, color="#f59e0b", linewidth=0.8, label="Spectral EMD")
    ax_emd.set_ylabel("EMD", color="#f59e0b")
    ax_emd.tick_params(axis="y", labelcolor="#f59e0b")

    lines1, labels1 = ax_cut.get_legend_handles_labels()
    lines2, labels2 = ax_emd.get_legend_handles_labels()
    ax_cut.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    # --- Row 3: Spectral centroid + band energies ---
    ax_stats = fig.add_subplot(gs[2])

    ax_stats.plot(indices, stats["spectral_centroid"], color="#3b82f6", linewidth=1.2, label="Spectral centroid")
    ax_stats.set_ylabel("Spectral centroid", color="#3b82f6")
    ax_stats.tick_params(axis="y", labelcolor="#3b82f6")

    ax_band = ax_stats.twinx()
    ax_band.stackplot(
        indices,
        stats["low_band_energy"],
        stats["mid_band_energy"],
        stats["high_band_energy"],
        colors=["#3b82f620", "#f59e0b30", "#10b98140"],
        labels=["Low (<0.15)", "Mid (0.15–0.5)", "High (>0.5)"],
    )
    ax_band.set_ylabel("Band energy fraction")
    ax_band.set_ylim(0, 1)

    lines1, labels1 = ax_stats.get_legend_handles_labels()
    lines2, labels2 = ax_band.get_legend_handles_labels()
    ax_stats.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)

    ax_stats.set_xlabel("Image index")
    ax_stats.set_title("Spectral centroid & band energy distribution")
    ax_stats.set_xlim(-0.5, n - 0.5)

    fig.savefig(output / "frequency_analysis.png", dpi=180, bbox_inches="tight")
    print(f"Saved {output / 'frequency_analysis.png'}")
    plt.close(fig)


def save_csv(
    images: list[tuple[str, NDArray]],
    stats: dict[str, NDArray],
    emd: NDArray,
    output: Path,
) -> None:
    """Write per-image frequency stats to CSV."""
    path = output / "frequency_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "name", "spectral_centroid", "cutoff_freq",
            "low_band_energy", "mid_band_energy", "high_band_energy",
            "total_energy", "spectral_emd_next",
        ])
        for i, (name, _) in enumerate(images):
            writer.writerow([
                i, name,
                f"{stats['spectral_centroid'][i]:.6f}",
                f"{stats['cutoff_freq'][i]:.6f}",
                f"{stats['low_band_energy'][i]:.6f}",
                f"{stats['mid_band_energy'][i]:.6f}",
                f"{stats['high_band_energy'][i]:.6f}",
                f"{stats['total_energy'][i]:.6e}",
                f"{emd[i]:.6f}" if i < len(emd) else "",
            ])
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze radial power spectrum evolution across a sequence of .npy captures."
    )
    parser.add_argument("directory", type=Path, help="Directory containing .npy files")
    parser.add_argument("--bins", type=int, default=128, help="Number of radial frequency bins (default: 128)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output")

    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.directory
    output.mkdir(parents=True, exist_ok=True)

    images = load_npy_files(args.directory)
    spectrum_matrix, freq_centers = compute_all_spectra(images, args.bins)
    stats = compute_stats(spectrum_matrix, freq_centers)
    emd = compute_spectral_emd(spectrum_matrix, freq_centers)

    plot(images, spectrum_matrix, freq_centers, stats, emd, output)

    if not args.no_csv:
        save_csv(images, stats, emd, output)


if __name__ == "__main__":
    main()
