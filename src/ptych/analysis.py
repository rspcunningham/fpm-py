from matplotlib import pyplot as plt
import torch
import seaborn as sns


def plot_k_vectors(kx_before: torch.Tensor, ky_before: torch.Tensor,
                   kx_after: torch.Tensor, ky_after: torch.Tensor,
                   save_path: str | None = None):
    """Plot k-space points before and after adding noise."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Before noise
    sns.scatterplot(x=kx_before.cpu().numpy(), y=ky_before.cpu().numpy(), ax=ax1, s=50)
    ax1.set_xlabel('kx')
    ax1.set_ylabel('ky')
    ax1.set_title('K-space Points (Before Noise)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # After noise
    sns.scatterplot(x=kx_after.cpu().numpy(), y=ky_after.cpu().numpy(), ax=ax2, s=50)
    ax2.set_xlabel('kx')
    ax2.set_ylabel('ky')
    ax2.set_title(f'K-space Points (After Noise)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

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
