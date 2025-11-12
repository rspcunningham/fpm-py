from ptych import forward_model, solve_inverse
import ptych.experimental as exp
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
kx_all = torch.tensor([k[0] for k in k_vectors])
ky_all = torch.tensor([k[1] for k in k_vectors])
captures_batched = forward_model(image_complex, pupil, kx_all, ky_all, downsample_factor=2)  # [B, H, W]
captures = [captures_batched[i] for i in range(len(k_vectors))]

# solve the inverse problem
pred_O, pred_P, metrics = solve_inverse(captures, k_vectors, 1024)
#pred_O, _, metrics = exp.solve_inverse(captures, 512)
pred_O_amplitude = torch.abs(pred_O) / torch.max(torch.abs(pred_O))
pred_P_amplitude = torch.abs(pred_P) / torch.max(torch.abs(pred_P))

# Plot analytics
plot_comparison([amplitude.cpu(), captures[zero_idx].cpu(), pred_O_amplitude.cpu()], ['Original', 'Center Illumination', 'Predicted'], 'tmp/adamw.png')
plot_comparison([pred_O_amplitude.cpu(), pred_P_amplitude.cpu()], ['Object Amplitude', 'Pupil Amplitude'])
plot_curves(metrics)
