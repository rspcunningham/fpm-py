from matplotlib import pyplot as plt
import torch


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
