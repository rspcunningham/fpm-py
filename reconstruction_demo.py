import os

import numpy as np
import torch
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt

from ptych import solve_tiled, PtychStudy
from ptych.core.pupil import make_ideal_pupil, make_zernike_pupil, ZernikeParams
from interpolation import interpolate_green

#BASE_DIR = "./demo/synthetic"
BASE_DIR = "./demo/real"
OUTPUT_DIR = f"{BASE_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

study = PtychStudy.from_disk(BASE_DIR)

# Center-crop to 256x256 for quick validation (2x2 grid of 128 tiles)
roi_size = 128
crop_size = 256
_, h, w = study.captures.shape
sh, sw = (h - crop_size) // 2, (w - crop_size) // 2
# Use first 37 captures (3 rings, ~8.92°) — spotlight stays in periphery, not in center crop
n_captures = 37
captures = study.captures[:n_captures, sh:sh + crop_size, sw:sw + crop_size]
captures = interpolate_green(captures)
captures = captures / captures.max()

# Initialize pupil from NA and optical parameters
upsample_ratio = 4
N = roi_size * upsample_ratio
pupil = make_ideal_pupil(
    N=N,
    NA=0.13,
    wavelength_m=study.manifest.captures[0].wavelength,
    sensor_pixel_size_m=study.manifest.sensor_pixel_size,
    magnification=study.manifest.magnification,
    downsample_ratio=upsample_ratio,
    num_phase_terms=20,
    num_amp_terms=20,
)


def save_tensor(tensor: torch.Tensor, path: str):
    np.save(path, tensor.cpu().numpy())


def save_png(tensor: torch.Tensor, path: str):
    arr = tensor.abs().cpu().numpy() ** 2  # intensity = |field|²
    arr_u8 = np.asarray(arr / arr.max() * 255, dtype=np.uint8)
    Image.fromarray(arr_u8).save(path)


def on_tile_complete(r: int, c: int, obj: torch.Tensor, tile_pupil: ZernikeParams, metrics: dict[str, list[float]]):
    # Save tensors
    save_tensor(obj, f"{OUTPUT_DIR}/tile_{r}_{c}_object.npy")
    pupil_tensor = make_zernike_pupil(tile_pupil.phase_coeffs, tile_pupil.amp_coeffs, tile_pupil.basis, tile_pupil.rad_fraction)
    save_tensor(pupil_tensor, f"{OUTPUT_DIR}/tile_{r}_{c}_pupil.npy")

    # Save PNGs
    save_png(obj, f"{OUTPUT_DIR}/tile_{r}_{c}_object.png")
    save_png(pupil_tensor, f"{OUTPUT_DIR}/tile_{r}_{c}_pupil.png")

    print(f"Tile ({r},{c}) done — final loss: {metrics['loss'][-1]:.6f}, rad_fraction: {tile_pupil.rad_fraction.item():.6f}")

    # Save loss curve
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # pyright: ignore[reportAny]
    epochs = range(len(metrics['loss']))
    sns.lineplot(x=list(epochs), y=metrics['loss'], ax=ax1)  # pyright: ignore[reportAny]
    ax1.set_ylabel('Loss')  # pyright: ignore[reportAny]
    ax1.set_title(f'Tile ({r},{c}) Training Metrics')  # pyright: ignore[reportAny]
    sns.lineplot(x=list(epochs), y=np.log(metrics['loss']), ax=ax2)  # pyright: ignore[reportAny]
    ax2.set_ylabel('Log Loss')  # pyright: ignore[reportAny]
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tile_{r}_{c}_metrics.png", dpi=150)
    plt.close()


result = solve_tiled(
    captures,
    study.kx_batch[:n_captures],
    study.ky_batch[:n_captures],
    pupil,
    roi_size=roi_size,
    upsample_ratio=upsample_ratio,
    torch_device="mps",
    on_tile_complete=on_tile_complete,
    tile_batch_size=4,
)

# Save stitched result
save_tensor(result, f"{OUTPUT_DIR}/stitched_object.npy")
save_png(result, f"{OUTPUT_DIR}/stitched_object.png")
print(f"Stitched result shape: {result.shape}")
