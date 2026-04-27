from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

from ptych.core.metrics import BatchMetricsRecord


type FloatArray = npt.NDArray[np.float32]


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    np.save(path, tensor.cpu().numpy())


def save_metrics_summary(
    metrics: list[BatchMetricsRecord],
    *,
    path: Path,
) -> None:
    sns.set_theme(style="darkgrid")
    fig, axes = cast(tuple[Figure, tuple[Axes, Axes]], plt.subplots(1, 2, figsize=(14, 5)))
    ax_loss, ax_capture = axes

    for batch_idx, record in enumerate(metrics):
        patches = record["patches"]
        batch_metrics = record["metrics"]
        loss: FloatArray = np.asarray(batch_metrics["loss"], dtype=np.float32)
        patch_loss: FloatArray = np.asarray(
            batch_metrics["patch_loss"],
            dtype=np.float32,
        )
        capture_loss: FloatArray = np.asarray(batch_metrics["capture_loss"], dtype=np.float32)
        epochs = np.arange(len(loss))
        batch_label = f"batch {batch_idx + 1}"
        loss_log: FloatArray = np.asarray(np.log10(np.clip(loss, 1e-12, None)), dtype=np.float32)

        total_label = "total" if batch_idx == 0 else f"total ({batch_label})"
        ax_loss.plot(epochs, loss_log, label=total_label, color="black", linewidth=2.2)

        for patch_idx, (r, c) in enumerate(patches):
            patch_loss_log: FloatArray = np.asarray(
                np.log10(np.clip(patch_loss[:, patch_idx], 1e-12, None)),
                dtype=np.float32,
            )
            ax_loss.plot(epochs, patch_loss_log, label=f"patch ({r},{c})", alpha=0.9)

        final_capture: FloatArray = np.asarray(capture_loss[-1], dtype=np.float32)
        capture_indices = np.arange(final_capture.shape[0], dtype=np.int32)
        ax_capture.plot(capture_indices, final_capture, label=batch_label)

    ax_loss.set_title("Log Loss by Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("log10(loss)")
    ax_loss.legend(ncol=2, fontsize="small")

    ax_capture.set_title("Final Capture Loss")
    ax_capture.set_xlabel("Capture index")
    ax_capture.set_ylabel("Loss")
    if len(metrics) > 1:
        ax_capture.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
