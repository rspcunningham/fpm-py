from ptych import forward_model, solve_inverse
from ptych.analysis import plot_comparison, plot_curves
import torch
from torchvision.io import read_image, ImageReadMode
from itertools import product

DEVICE = torch.device('mps')
torch.set_default_device(DEVICE)

# load the sample image and set phase = torch.pi * amplitude
amplitude = read_image('data/bars.png', mode=ImageReadMode.GRAY).squeeze(0).float() / 255.0
phase = torch.pi * amplitude
image_complex = (amplitude * torch.exp(1j * phase)).to(DEVICE)

height, width = image_complex.shape
print(f"Image shape: {height}x{width}")

# Create circular pupil
radius = 50
y_coords, x_coords = torch.meshgrid(
    torch.arange(height, dtype=torch.float32),
    torch.arange(width, dtype=torch.float32),
    indexing='ij'
)
center_y, center_x = height / 2, width / 2
distance = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
pupil = (distance <= radius).float()

# create grid of k-vectors
k_vectors: list[tuple[int, int]] = [(k[0], k[1]) for k in product(range(-50, 51, 10), repeat=2)]
zero_idx = k_vectors.index((0, 0))
print(f"total k_vectors: {len(k_vectors)}")

# Generate captures using batched forward model
kx_all = torch.tensor([k[0] for k in k_vectors]).float()
ky_all = torch.tensor([k[1] for k in k_vectors]).float()
captures = forward_model(image_complex, pupil, kx_all, ky_all, downsample_factor=2)  # [B, H, W]

# solve the inverse problem
output_size = 1024
object = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)
pupil = 0.5 * torch.ones(output_size, output_size, dtype=torch.complex64)

pred_O, _, metrics = solve_inverse(captures, object, pupil, kx_all, ky_all)
pred_O_amplitude = torch.abs(pred_O) / torch.max(torch.abs(pred_O))

# Plot analytics
plot_comparison([amplitude.cpu(), captures[zero_idx].cpu(), pred_O_amplitude.cpu()], ['Original', 'Predicted with learned k-vectors', 'Predicted'], 'tmp/adamw.png')
#plot_comparison([pred_O_amplitude.cpu(), pred_P_amplitude.cpu()], ['Object Amplitude', 'Pupil Amplitude'])
plot_curves(metrics)
