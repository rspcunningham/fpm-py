import torch
from torchvision.io import read_image, ImageReadMode
import ptych

pytorch_device = ptych.utils.get_default_device()
torch.set_default_device(pytorch_device)
print("Running on: ", pytorch_device)

def load_object_and_pupil() -> tuple[torch.Tensor, torch.Tensor]:
    # load the sample image and set phase = torch.pi * amplitude
    amplitude = read_image('data/bars.png', mode=ImageReadMode.GRAY).squeeze(0).float() / 255.0
    phase = torch.pi * amplitude
    image_complex = (amplitude * torch.exp(1j * phase)).to(pytorch_device)

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

    return image_complex, pupil


def load_k_vectors(width: float, pitch: float) -> tuple[torch.Tensor, torch.Tensor]:

      min_val = -width / 2
      max_val = width / 2

      print(min_val, max_val)

      k_range = torch.arange(min_val, max_val + pitch, pitch)
      ky_grid, kx_grid = torch.meshgrid(k_range, k_range, indexing='ij')

      kx_all = kx_grid.flatten()
      ky_all = ky_grid.flatten()

      print(f"total k_vectors: {len(kx_all)}")

      return kx_all, ky_all
