from ptych import forward_model, solve_inverse, analysis
from ptych.utils import obj_to_amp, normalize_captures
import torch

from initialize import load_object_and_pupil, load_k_vectors

k_pitch = 0.1
noise_norm_std = 0.1

object_original, pupil_original = load_object_and_pupil()
kx_all, ky_all = load_k_vectors(0.6, k_pitch)

def generate_noise(norm_sigma: float, vector: torch.Tensor):
    sigma = norm_sigma * k_pitch
    noise = torch.randn_like(vector) * sigma
    return vector + noise

#kx_all_noise = generate_noise(noise_norm_std, kx_all.clone())
#ky_all_noise = generate_noise(noise_norm_std, ky_all.clone())

# Plot k-space points before and after noise
#analysis.plot_k_vectors([(kx_all, ky_all), (kx_all_noise, ky_all_noise)], ['Original', f'Noise @ σ = {noise_norm_std} k-pitch'])

captures_good = forward_model(object_original.clone(), pupil_original.clone(), kx_all.clone(), ky_all.clone(), downsample_factor=2)  # [B, H, W]
#captures = forward_model(object.clone(), pupil.clone(), kx_all_noise.clone(), ky_all_noise.clone(), downsample_factor=2)  # [B, H, W]

output_size = 512
object = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)
pupil = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)

prediction_good, _, metrics = solve_inverse(captures_good.clone(), object.clone(), pupil.clone(), kx_all.clone(), ky_all.clone())
#prediction_noisy, _, _ = solve_inverse(captures.clone(), object.clone(), pupil.clone(), kx_all_noise.clone(), ky_all_noise.clone())
#prediction_noisy_learned, _, metrics = solve_inverse(captures.clone(), object.clone(), pupil.clone(), kx_all_noise.clone(), ky_all_noise.clone(), learn_k_vectors=True)
analysis.plot_comparison([obj_to_amp(object_original), obj_to_amp(prediction_good)], ['Original', 'Good'])
#analysis.plot_comparison([obj_to_amp(prediction_good), obj_to_amp(prediction_noisy), obj_to_amp(prediction_noisy_learned)], ['Good', 'Noisy', 'Noisy Learned'])
analysis.plot_curves(metrics)
