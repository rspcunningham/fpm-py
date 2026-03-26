import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ptych import solve_tiled, PtychStudy
from ptych.core.pupil import ZernikeParams, make_zernike_pupil, make_ideal_pupil
from interpolation import interpolate_green

BASE_DIR = "./tmp/6456b6d0-3b2a-4fef-a734-cc64d68bd4ac"
OUTPUT_DIR = f"{BASE_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

study = PtychStudy.from_disk(BASE_DIR)

# Pre-crop to center 256x256 so we get a 2x2 grid of 128x128 tiles
_, h, w = study.captures.shape
sh, sw = (h - 256) // 2, (w - 256) // 2
study.captures = study.captures[:, sh:sh + 256, sw:sw + 256]


def save_tensor(tensor, path: str):
    np.save(path, tensor.cpu().numpy())


# Per-tile callback: save each tile's result and print metrics
def on_tile_complete(r, c, obj, pupil: ZernikeParams, metrics):
    save_tensor(obj, f"{OUTPUT_DIR}/tile_{r}_{c}_object.npy")
    pupil_tensor = make_zernike_pupil(pupil.phase_coeffs, pupil.amp_coeffs, pupil.basis, pupil.rad_fraction)
    save_tensor(pupil_tensor, f"{OUTPUT_DIR}/tile_{r}_{c}_pupil.npy")
    print(f"Tile ({r},{c}) done — final loss: {metrics['loss'][-1]:.6f}, rad_fraction: {pupil.rad_fraction.item():.6f}")

    # Save loss curve per tile
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True) # pyright: ignore[reportAny]
    epochs = range(len(metrics['loss']))
    sns.lineplot(x=list(epochs), y=metrics['loss'], ax=ax1) # pyright: ignore[reportAny]
    ax1.set_ylabel('Loss') # pyright: ignore[reportAny]
    ax1.set_title(f'Tile ({r},{c}) Training Metrics') # pyright: ignore[reportAny]
    sns.lineplot(x=list(epochs), y=np.log(metrics['loss']), ax=ax2) # pyright: ignore[reportAny]
    ax2.set_ylabel('Log Loss') # pyright: ignore[reportAny]
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tile_{r}_{c}_metrics.png", dpi=150)
    plt.close()


captures = interpolate_green(study.captures[:37])
captures = captures / captures.max()

upsample_ratio = 4
N = 128 * upsample_ratio
pupil = make_ideal_pupil(
    N=N,
    NA=0.13,
    wavelength_m=study.manifest.captures[0].wavelength,
    sensor_pixel_size_m=study.manifest.sensor_pixel_size,
    magnification=study.manifest.magnification,
    downsample_ratio=upsample_ratio,
    num_phase_terms=10,
    num_amp_terms=10,
)

result = solve_tiled(
    captures,
    study.kx_batch[:37],
    study.ky_batch[:37],
    pupil,
    roi_size=128,
    upsample_ratio=upsample_ratio,
    torch_device="mps",
    on_tile_complete=on_tile_complete,
    tile_batch_size=4,
)

save_tensor(result, f"{OUTPUT_DIR}/stitched_object.npy")
print(f"Stitched result shape: {result.shape}")
