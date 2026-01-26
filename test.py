import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ptych import solve_inverse, PtychStudy
from ptych.core.zernike import precompute_zernike_basis, make_zernike_pupil, ZernikeParams
from interpolation import interpolate_green

BASE_DIR = "./tmp/new/6456b6d0-3b2a-4fef-a734-cc64d68bd4ac"
#BASE_DIR = "./tmp/test"
OUTPUT_DIR = f"{BASE_DIR}/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

study = PtychStudy.from_disk(BASE_DIR, normalize_by_exposure=True)

# Apply Bayer demosaicing (interpolate green channel)
study.captures = interpolate_green(study.captures)

# Crop captures to square (center crop)
_, h, w = study.captures.shape
crop_size = 128
n_captures = 21
start_h = (h - crop_size) // 2
start_w = (w - crop_size) // 2
captures_cropped = study.captures[:n_captures, start_h:start_h + crop_size, start_w:start_w + crop_size]
captures_cropped /= 65535.0

captures_cropped = captures_cropped - captures_cropped.min()
captures_cropped = captures_cropped / captures_cropped.max()

kx_batch = study.kx_batch[:n_captures]
ky_batch = study.ky_batch[:n_captures]

# Initialize object and pupil with upsampled dimensions
upsample_ratio = 4
eps = 1e-8

init_amp = torch.nn.functional.interpolate(
    captures_cropped[0:1, :, :].unsqueeze(1),  # [B, n, n]
    scale_factor=upsample_ratio,
    mode='nearest'
).squeeze()  #

init_amp = torch.sqrt(init_amp + eps)  # Convert intensity to amplitude
init_phase = torch.zeros_like(init_amp)  # or small random noise

object_tensor = init_amp * torch.exp(1j * init_phase)

# Initialize pupil using Zernike basis
basis = precompute_zernike_basis(crop_size * upsample_ratio, num_phase_terms=10, num_amp_terms=10, rad_fraction=0.047)
phase_coeffs = torch.zeros(basis.num_phase_terms)
amp_coeffs = torch.zeros(basis.num_amp_terms)
amp_coeffs[0] = 1.0  # Piston = uniform amplitude

pupil = ZernikeParams(phase_coeffs, amp_coeffs, basis)

def save_tensor(tensor: torch.Tensor, path: str):
    """Save a tensor as a .npy file."""
    np.save(path, tensor.cpu().numpy())

def save_checkpoint(epoch: int, obj: torch.Tensor):
    save_tensor(obj, f"{OUTPUT_DIR}/checkpoint_{epoch:04d}.npy")

# Save initial object and pupil before reconstruction
save_tensor(captures_cropped[0], f"{OUTPUT_DIR}/first_capture.npy")
save_tensor(object_tensor, f"{OUTPUT_DIR}/initial_object.npy")
initial_pupil_tensor = make_zernike_pupil(pupil.phase_coeffs, pupil.amp_coeffs, pupil.basis)
save_tensor(initial_pupil_tensor, f"{OUTPUT_DIR}/initial_pupil.npy")

object, pupil, metrics = solve_inverse(
    captures_cropped,
    object_tensor,
    initial_pupil_tensor,
    kx_batch,
    ky_batch,
    torch_device="mps",
    on_checkpoint=save_checkpoint,
    checkpoint_interval=100,
    learn_k_vectors=False
)

# Save final object and pupil results
save_tensor(object, f"{OUTPUT_DIR}/object_result.npy")
if isinstance(pupil, ZernikeParams):
    pupil_tensor = make_zernike_pupil(pupil.phase_coeffs, pupil.amp_coeffs, pupil.basis)
else:
    pupil_tensor = pupil
save_tensor(pupil_tensor, f"{OUTPUT_DIR}/pupil_result.npy")

# Plot and save metrics
sns.set_theme(style="darkgrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

epochs = range(len(metrics['loss']))

sns.lineplot(x=list(epochs), y=metrics['loss'], ax=ax1)
ax1.set_ylabel('Loss')
ax1.set_title('Training Metrics')

sns.lineplot(x=list(epochs), y=np.log(metrics['loss']), ax=ax2)
ax2.set_ylabel('Log Loss')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics.png", dpi=150)
plt.close()
