import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# Set Seaborn style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def plot_comparison_with_histograms(
    images: list[NDArray[np.float64]],
    titles: list[str] | None = None,
    figsize: tuple[int, int]=(16, 16),
    reference_idx: int = 0
) -> tuple[Figure, dict[str, list[float] | list[tuple[int, ...]] | list[bool] | list[str]]]:

    """
    Plot multiple images with their histograms, Fourier spectra, and calculate statistical measures.
    Works with images of different dimensions.

    Parameters:
    -----------
    images : list of numpy.ndarray
        List of images to compare
    titles : list, optional
        List of strings for image titles (should match length of images)
    figsize : tuple, optional
        Figure size
    reference_idx : int, optional
        Index of the image to use as reference for comparison metrics (default: 0)

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plots
    stats : dict
        Dictionary of statistical measures for comparison
    """

    # Check inputs
    if not isinstance(images, (list, tuple)) or len(images) < 1:
        raise ValueError("images must be a list/tuple containing at least one image")

    n_images = len(images)

    # Normalize images to 0-1 range
    def normalize(img: NDArray[np.float64]) -> NDArray[np.float64]:
        img_min, img_max = img.min(), img.max()
        return (img - img_min) / (img_max - img_min) if img_max > img_min else img

    # Keep original normalized images for display
    norm_images = [normalize(img) for img in images]

    # Set default titles if not provided
    if titles is None:
        titles = [f"Output {i+1}" for i in range(n_images)]
    elif len(titles) != n_images:
        titles = titles[:n_images] if len(titles) > n_images else titles + [f"Output {i+1}" for i in range(len(titles), n_images)]

    # Calculate statistics for each individual image
    flat_images = [img.flatten() for img in norm_images]

    # Basic statistics (these work regardless of image dimensions)
    stats_dict = {
        'dimensions': [img.shape for img in norm_images],
        'mean': [np.mean(flat) for flat in flat_images],
        'median': [np.median(flat) for flat in flat_images],
        'std_dev': [np.std(flat) for flat in flat_images],
        'min': [np.min(flat) for flat in flat_images],
        'max': [np.max(flat) for flat in flat_images],
        'dynamic_range': [np.max(flat) - np.min(flat) for flat in flat_images],
        'skewness': [stats.skew(flat) for flat in flat_images],
        'kurtosis': [stats.kurtosis(flat) for flat in flat_images],
        'entropy': [stats.entropy(np.histogram(flat, bins=256)[0]) for flat in flat_images]
    }

    # Reference image for comparison
    reference = norm_images[reference_idx]
    flat_reference = flat_images[reference_idx]

    # Comparison metrics (relative to reference image)
    if n_images > 1:
        stats_dict['mse'] = []
        stats_dict['rmse'] = []
        stats_dict['psnr'] = []
        stats_dict['ssim'] = []
        stats_dict['ncc'] = []
        stats_dict['resized'] = []

        for i, img in enumerate(norm_images):
            if i == reference_idx:
                # Self-comparison (perfect scores)
                stats_dict['mse'].append(0.0)
                stats_dict['rmse'].append(0.0)
                stats_dict['psnr'].append(float('inf'))
                stats_dict['ssim'].append(1.0)
                stats_dict['ncc'].append(1.0)
                stats_dict['resized'].append(False)
                continue

            # Resize image to match reference if dimensions differ
            resized = False
            img_comp = img

            if img.shape != reference.shape:
                resized = True
                # Resize to match reference dimensions for comparison
                img_comp = resize(img, reference.shape, anti_aliasing=True, preserve_range=True)
                # Re-normalize after resize to ensure 0-1 range
                img_comp = normalize(img_comp)

            # Mean Squared Error
            mse = np.mean((reference - img_comp) ** 2)
            stats_dict['mse'].append(mse)

            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            stats_dict['rmse'].append(rmse)

            # Peak Signal-to-Noise Ratio
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            else:
                psnr = float('inf')
            stats_dict['psnr'].append(psnr)

            # Structural Similarity Index
            try:
                ssim_value = ssim(reference, img_comp, data_range=1.0)
                stats_dict['ssim'].append(ssim_value)
            except Exception as e:
                stats_dict['ssim'].append(f'Error: {str(e)}')

            # Normalized Cross-Correlation
            flat_comp = img_comp.flatten()
            ncc = np.corrcoef(flat_reference, flat_comp)[0, 1]
            stats_dict['ncc'].append(ncc)

            # Track whether image was resized
            stats_dict['resized'].append(resized)

    # Create figure layout based on number of images
    n_cols = min(3, n_images)  # Max 3 columns

    # Define row types and names
    row_types = ["Image", "Fourier Spectrum", "Histogram"]
    n_row_types = len(row_types)
    n_rows_total = n_row_types + 1  # Row types + statistics table

    # Create GridSpec with appropriate heights
    fig = plt.figure(figsize=figsize, facecolor='white')
    row_heights = [4, 3, 3, 2]  # Images, Fourier spectra, histograms, stats table
    grid = GridSpec(n_rows_total, n_cols + 1, width_ratios=[0.15] + [0.85/n_cols] * n_cols,
                    height_ratios=row_heights, figure=fig)

    # Set up axes for each image, Fourier spectrum, and histogram
    axes_images = []
    axes_fourier = []
    axes_hists = []
    axes_row_labels = []

    # Create row label axes (leftmost column)
    for i in range(n_row_types):
        ax_label = fig.add_subplot(grid[i, 0])
        ax_label.axis('off')
        axes_row_labels.append(ax_label)

    # Create content axes (other columns)
    for i in range(n_images):
        col = (i % n_cols) + 1  # +1 to account for the row label column

        # Image axis
        ax_img = fig.add_subplot(grid[0, col])
        axes_images.append(ax_img)

        # Fourier spectrum axis
        ax_fourier = fig.add_subplot(grid[1, col])
        axes_fourier.append(ax_fourier)

        # Histogram axis
        ax_hist = fig.add_subplot(grid[2, col])
        axes_hists.append(ax_hist)

    # Statistics table axis (spans all columns in the last row)
    ax_stats = fig.add_subplot(grid[-1, :])
    ax_stats.axis('off')

    # Create title with better styling
    fig.suptitle("FPM Reconstruction Comparison",
                 fontsize=16,
                 fontweight='bold')

    # Add row labels with improved styling
    row_label_colors = ['#333333', '#333333', '#333333']
    for i, (label, ax, color) in enumerate(zip(row_types, axes_row_labels, row_label_colors)):
        ax.text(0.5, 0.5, label, ha='center', va='center',
                rotation=90, fontsize=12, fontweight='bold',
                color=color)

    # Define histogram colors - same color for all non-reference
    reference_color = 'red'
    non_reference_color = '#4682b4'  # Steel Blue

    # Add column titles as overall titles with better styling
    for i, (img, ax_img) in enumerate(zip(norm_images, axes_images)):
        # Create column title
        title = titles[i]
        if i == reference_idx:
            title += " (REFERENCE)"
        title += f"\n{img.shape}"

        ax_img.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Mark reference image
        if i == reference_idx:
            for spine in ax_img.spines.values():
                spine.set_color('red')
                spine.set_linewidth(2)

    # Collect all image data to determine global min/max for consistent colormap scaling
    all_img_data = np.concatenate([img.flatten() for img in norm_images])
    all_img_min, all_img_max = all_img_data.min(), all_img_data.max()

    # Collect all Fourier data to determine global min/max for consistent colormap scaling
    all_fourier_data = []
    for img in norm_images:
        f_transform = np.fft.fftshift(np.fft.fft2(img))
        log_spectrum = np.log(np.abs(f_transform) + 1)  # Add 1 to avoid log(0)
        all_fourier_data.append(log_spectrum.flatten())
    all_fourier_data = np.concatenate(all_fourier_data)
    fourier_min, fourier_max = all_fourier_data.min(), all_fourier_data.max()

    # Pre-calculate histogram data to determine appropriate y-axis limits
    hist_counts = []
    hist_bins = []
    bin_count = 100

    for img in norm_images:
        counts, bin_edges = np.histogram(img.flatten(), bins=bin_count, density=True)
        hist_counts.append(counts)
        hist_bins.append((bin_edges[:-1] + bin_edges[1:]) / 2)  # Bin centers

    # Find the maximum count across all histograms for consistent y-axis scaling
    max_count = max([count.max() for count in hist_counts])

    # Plot images, Fourier spectra, and histograms
    for i, (img, ax_img, ax_fourier, ax_hist) in enumerate(zip(norm_images, axes_images, axes_fourier, axes_hists)):
        # --------------------------------
        # Plot image with consistent colormap and scale
        # --------------------------------
        im = ax_img.imshow(img, cmap='viridis', vmin=all_img_min, vmax=all_img_max)
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # Remove grid lines and axis labels from images
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.grid(False)

        # --------------------------------
        # Plot Fourier spectrum with consistent colormap and scale
        # --------------------------------
        f_transform = np.fft.fftshift(np.fft.fft2(img))
        log_spectrum = np.log(np.abs(f_transform) + 1)  # Add 1 to avoid log(0)

        # Get spectrum dimensions
        height, width = log_spectrum.shape

        # Calculate center coordinates
        center_y, center_x = height // 2, width // 2

        # Plot the spectrum
        im_fourier = ax_fourier.imshow(log_spectrum, cmap='viridis', vmin=fourier_min, vmax=fourier_max,
                                       extent=[-center_x, width-center_x-1, height-center_y-1, -center_y])

        # Set appropriate ticks to show coordinates with (0,0) at center
        # Create ticks at regular intervals, including the zero point
        x_max = max(center_x, width - center_x - 1)
        y_max = max(center_y, height - center_y - 1)

        # Choose a reasonable number of ticks (not too crowded)
        n_ticks = min(5, max(width, height) // 50 + 1)
        x_step = max(1, x_max // (n_ticks // 2))
        y_step = max(1, y_max // (n_ticks // 2))

        x_ticks = np.arange(-center_x, width-center_x, x_step)
        if 0 not in x_ticks:
            x_ticks = np.sort(np.append(x_ticks, 0))

        y_ticks = np.arange(-center_y, height-center_y, y_step)
        if 0 not in y_ticks:
            y_ticks = np.sort(np.append(y_ticks, 0))

        # Set ticks
        ax_fourier.set_xticks(x_ticks)
        ax_fourier.set_yticks(y_ticks)

        # Add grid for better visibility of the center
        ax_fourier.grid(True, alpha=0.3)

        # Add colorbar
        fig.colorbar(im_fourier, ax=ax_fourier, fraction=0.046, pad=0.04)

        # --------------------------------
        # Plot histogram with consistent colors
        # --------------------------------
        hist_color = reference_color if i == reference_idx else non_reference_color

        # Plot pre-calculated histogram with controlled y limits
        ax_hist.bar(hist_bins[i], hist_counts[i], width=(1.0/bin_count), color=hist_color, alpha=0.7)

        # Set y-axis limit based on calculated maximum
        ax_hist.set_ylim(0, max_count * 1.1)  # Add 10% padding

        # Add labels
        ax_hist.set_xlabel("Normalized Intensity")
        ax_hist.set_ylabel("Frequency")

        # Set x-axis limits for consistency
        ax_hist.set_xlim(0, 1)

    # Create table of statistics (for up to 5 images to keep it readable)
    display_n = min(5, n_images)
    selected_indices = [reference_idx] + [i for i in range(n_images) if i != reference_idx][:display_n-1]

    # Create base table data with metrics for each image
    table_data = [["Metric"] + [titles[i] for i in selected_indices] + ["Notes"]]

    # Add individual image metrics
    metrics = [
        ("Dimensions", 'dimensions', lambda x: f"{x}"),
        ("Mean", 'mean', lambda x: f"{x:.4f}"),
        ("Std Dev", 'std_dev', lambda x: f"{x:.4f}"),
        ("Entropy", 'entropy', lambda x: f"{x:.4f}")
    ]

    for name, key, formatter in metrics:
        row = [name] + [formatter(stats_dict[key][i]) for i in selected_indices]
        if key != 'dimensions':
            diffs = [abs(stats_dict[key][i] - stats_dict[key][reference_idx])
                    for i in selected_indices if i != reference_idx]
            if diffs:
                row.append(f"Max diff: {max(diffs):.4f}")
            else:
                row.append("")
        else:
            row.append("")
        table_data.append(row)

    # Add comparison metrics (only if multiple images)
    if n_images > 1 and 'mse' in stats_dict:
        comparison_metrics = [
            ("MSE", 'mse', lambda x, r: f"{x:.6f}" + (" (resized)" if r else "")),
            ("PSNR", 'psnr', lambda x, r: f"{x:.2f} dB" + (" (resized)" if r else "")),
            ("SSIM", 'ssim', lambda x, r: f"{x}" if isinstance(x, str) else f"{x:.4f}" + (" (resized)" if r else "")),
            ("NCC", 'ncc', lambda x, r: f"{x:.4f}" + (" (resized)" if r else ""))
        ]

        for name, key, formatter in comparison_metrics:
            row = [name, "Reference"]
            for i in selected_indices:
                if i == reference_idx:
                    continue
                #idx = selected_indices.index(i)
                value = stats_dict[key][i]
                resized = stats_dict['resized'][i]
                row.append(formatter(value, resized))

            # Fill remaining cells if needed
            row.extend([""] * (len(selected_indices) + 1 - len(row)))
            row.append("vs Reference")
            table_data.append(row)

    # Create the table with clean styling
    col_widths = [0.2] + [0.8/display_n] * display_n + [0.2]
    table = ax_stats.table(cellText=table_data, loc='center', cellLoc='center', colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style the table with professional colors
    header_color = '#2c3e50'       # Dark blue for header row
    row_header_color = '#ecf0f1'   # Light grey for row headers
    reference_color_bg = '#ffebee' # Light red for reference column
    even_row_color = '#f8f9fa'     # Light grey for even rows
    odd_row_color = '#ffffff'      # White for odd rows

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor(header_color)
        elif col == 0:  # Header column
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(row_header_color)
        elif col == 1 and row > 0:  # Reference column
            cell.set_facecolor(reference_color_bg)
        else:  # Regular cells with subtle alternating colors
            cell.set_facecolor(even_row_color if row % 2 == 0 else odd_row_color)

        # Add subtle border
        cell.set_edgecolor('#e0e0e0')

    # Better layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)

    return fig, stats_dict
