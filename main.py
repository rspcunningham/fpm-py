from ptych import forward_model, solve_inverse, analysis
from ptych.utils import obj_to_amp, normalize_captures
import torch

from initialize import load_object_and_pupil, load_k_vectors

k_pitch = 0.02
noise_norm_std = 1

object, pupil = load_object_and_pupil()
kx_all, ky_all = load_k_vectors(0.2, k_pitch)

def generate_noise(norm_sigma: float, vector: torch.Tensor):
    sigma = norm_sigma * k_pitch
    noise = torch.randn_like(vector) * sigma
    return vector + noise

kx_all_noise = generate_noise(noise_norm_std, kx_all.clone())
ky_all_noise = generate_noise(noise_norm_std, ky_all.clone())

# Plot k-space points before and after noise
analysis.plot_k_vectors([(kx_all, ky_all), (kx_all_noise, ky_all_noise)], ['Original', f'Noise @ σ = {noise_norm_std} k-pitch'])

captures = forward_model(object, pupil, kx_all, ky_all, downsample_factor=2)  # [B, H, W]

output_size = 512
object = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)
pupil = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)

prediction_good, pupil_good, metrics_good = solve_inverse(captures, object, pupil, kx_all, ky_all)
prediction_noisy, pupil_noisy, metrics_noisy = solve_inverse(captures, object, pupil, kx_all_noise, ky_all_noise)
prediction_noisy_learned, pupil_noisy_learned, metrics_noisy_learned = solve_inverse(captures, object, pupil, kx_all_noise, ky_all_noise, learn_k_vectors=True)

analysis.plot_comparison([obj_to_amp(prediction_good), obj_to_amp(prediction_noisy), obj_to_amp(prediction_noisy_learned)], ['Good', 'Noisy', 'Noisy Learned'], "tmp/test_1.png")

analysis.plot_curves(metrics_good, "tmp/loss_good.png")
analysis.plot_curves(metrics_noisy, "tmp/loss_noisy.png")
analysis.plot_curves(metrics_noisy_learned, "tmp/loss_noisy_learned.png")
