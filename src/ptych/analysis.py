from matplotlib import pyplot as plt
import torch
import seaborn as sns


def plot_k_vectors(k_pairs: list[tuple[torch.Tensor, torch.Tensor]], labels: list[str], save_path: str | None = None):
    """Plot k-space points before and after adding noise."""
    assert len(k_pairs) == len(labels), "Number of k-pairs and labels must match"

    fig, axes = plt.subplots(1, len(k_pairs), figsize=(7 * len(k_pairs), 6))

    for i, (k_pair, label) in enumerate(zip(k_pairs, labels)):
        ax = axes[i]

        sns.scatterplot(x=k_pair[0].cpu().numpy(), y=k_pair[1].cpu().numpy(), ax=ax, s=50)
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        ax.set_title(f'K-space Points ({label})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comparison(images: list[torch.Tensor], labels: list[str], save_path: str | None = None):
    assert len(images) == len(labels), "Number of images and labels must match"
    n = len(images)

    _ = plt.figure(figsize=(5 * n, 6))

    for i, (im, label) in enumerate(zip(images, labels)):
        _ = plt.subplot(1, n, i + 1)
        _ = plt.imshow(im, cmap='gray')
        _ = plt.title(label)
        _ = plt.axis('off')

    _ = plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    _ = plt.show()

def plot_curves(metric_dict: dict[str, list[float]], save_path: str | None = None):
    assert len(metric_dict) > 0, "No metrics provided"

    _ = plt.figure(figsize=(10, 6))

    for key, values in metric_dict.items():
        _ = plt.plot(values, label=key)

    _ = plt.xlabel('Epoch')
    _ = plt.ylabel('Value')
    _ = plt.legend()
    _ = plt.grid(True, alpha=0.3)

    _ = plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    _ = plt.show()
