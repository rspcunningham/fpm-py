from matplotlib import pyplot as plt
import torch


def plot_comparison(images: list[torch.Tensor], labels: list[str]):
    assert len(images) == len(labels), "Number of images and labels must match"
    n = len(images)

    _ = plt.figure(figsize=(5 * n, 5))

    for i, (im, label) in enumerate(zip(images, labels)):
        _ = plt.subplot(1, n, i + 1)
        _ = plt.imshow(im, cmap='gray')
        _ = plt.title(label)
        _ = plt.axis('off')

    _ = plt.tight_layout()
    _ = plt.show()
