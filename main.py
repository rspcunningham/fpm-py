from ptych import forward_model, solve_inverse, analysis
import torch

from initialize import load_object_and_pupil, load_k_vectors

def obj_to_amp(obj: torch.Tensor):
    return (torch.abs(obj) / torch.max(torch.abs(obj))).cpu()

object, pupil = load_object_and_pupil()
kx_all, ky_all = load_k_vectors(0.5, 0.1)

sigma = 0.005
noise_x = torch.randn_like(kx_all) * sigma
noise_y = torch.randn_like(ky_all) * sigma

kx_all_noise = kx_all.clone()
kx_all_noise += noise_x
ky_all_noise = ky_all.clone()
ky_all_noise += noise_y

# Plot k-space points before and after noise
analysis.plot_k_vectors(kx_all, ky_all, kx_all_noise, ky_all_noise)

captures = forward_model(object, pupil, kx_all_noise, ky_all_noise, downsample_factor=2)  # [B, H, W]

output_size = 1024
object = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)
pupil = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)

#prediction_good, _, _ = solve_inverse(captures, object, pupil, kx_all, ky_all)
prediction_noisy, _, _ = solve_inverse(captures, object, pupil, kx_all, ky_all)
prediction_noisy_learned, _, metrics = solve_inverse(captures, object, pupil, kx_all, ky_all, learn_k_vectors=True)

analysis.plot_comparison([obj_to_amp(prediction_noisy), obj_to_amp(prediction_noisy_learned)], ['Noisy', 'Noisy Learned'])
analysis.plot_curves(metrics)
