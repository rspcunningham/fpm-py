from ptych import forward_model
from ptych.analysis import plot_comparison
from ptych.train import training_loop

from itertools import product
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

amplitude = torch.from_numpy(np.array(Image.open('data/bars.png').convert('L'))) / 255.0
phase = torch.pi * amplitude
image_complex = amplitude * torch.exp(1j * phase)

height, width = image_complex.shape
print(f"Image shape: {height}x{width}")

O = image_complex

# Create circular pupil
radius = 50
y_coords, x_coords = torch.meshgrid(
    torch.arange(height, dtype=torch.float32),
    torch.arange(width, dtype=torch.float32),
    indexing='ij'
)
center_y, center_x = height / 2, width / 2
distance = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
P = (distance <= radius).float()

k_vectors: list[tuple[int, int]] = [(k[0], k[1]) for k in product(range(-50, 51, 10), repeat=2)]
print(f"total k_vectors: {len(k_vectors)}")

captures = [forward_model(O, P, k[0], k[1]) for k in k_vectors]

pred_O, _, losses = training_loop(captures, k_vectors, 512)
pred_amplitude = torch.abs(pred_O)

# Plot loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot comparison
plot_comparison([amplitude, captures[60], pred_amplitude], ['Original', 'Center Illumination', 'Predicted'])
