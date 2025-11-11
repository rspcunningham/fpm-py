# fpm-py

A Python library for Fourier Ptychographic Microscopy (FPM) simulation and reconstruction with GPU acceleration.

## Overview

This library implements both the **forward pass** (reconstruction from captures) and **backward pass** (simulation of captures from an object) for Fourier ptychography. It uses PyTorch for efficient GPU-accelerated computation and provides a clean, type-safe API for research and development.

Fourier ptychography synthesizes a high-resolution, wide-field image by computationally combining multiple low-resolution images captured under varying illumination angles. This creates a synthetic aperture larger than the physical objective NA, enabling resolution enhancement while maintaining a large field of view.

## Installation

```bash
pip install fpm-py
```

Or with `uv` (preferred):
```bash
uv add fpm-py
```

**Requirements**: 3.12 ≥ Python ≥ 3.9, PyTorch ≥ 2.8.0

## Quick Start

```python
import fpm_py as fpm
from fpm_py.utils.data_utils import image_to_tensor
import torch

# Load a test object
obj = image_to_tensor("path/to/image.png", to_complex=True)

# Define k-vectors (illumination angles) - 5×5 grid
grid_size = 5
spacing = 0.2
k_range = torch.linspace(-(grid_size // 2), grid_size // 2, grid_size) * spacing
kx, ky = torch.meshgrid(k_range, k_range, indexing="ij")
k_vectors = torch.stack((kx.flatten(), ky.flatten()), dim=1)

# Simulate FPM captures (backward pass)
dataset = fpm.kvectors_to_image_series(
    obj=obj,
    k_vectors=k_vectors,
    pupil_radius=100,
    wavelength=550e-9,      # 550 nm (green light)
    pixel_size=1e-6,        # 1 μm pixels
    magnification=10.0      # 10× objective
)

# Reconstruct high-resolution image (forward pass)
reconstruction = fpm.reconstruct(dataset, output_scale_factor=4, max_iters=10)

# View magnitude
import matplotlib.pyplot as plt
plt.imshow(reconstruction.abs().cpu().numpy(), cmap='gray')
plt.show()
```

For a more detailed walkthrough, see [example.py](https://github.com/rspcunningham/fpm-py/blob/main/example.py).

## Core Concepts

### Data Structures

#### `ImageCapture`
Represents a single FPM capture with its associated k-vector:
```python
@dataclass(frozen=True)
class ImageCapture:
    image: torch.Tensor      # (H, W) intensity image
    k_vector: torch.Tensor   # (2,) illumination angle in k-space
```

#### `ImageSeries`
Container for multiple captures with acquisition settings:
```python
@dataclass
class ImageSeries:
    captures: Sequence[ImageCapture]
    settings: AcquisitionSettings  # du, wavelength, pixel_size
    device: torch.device
```

The `du` parameter is the **k-space lattice spacing** — it relates pixel coordinates in Fourier space to physical spatial frequencies. Automatically computed as:

```python
du = wavelength / (N × effective_pixel_size)
```

where `effective_pixel_size = pixel_size / magnification`.

### Backward Pass: Simulation

Generate synthetic FPM captures from a known object (inverse of reconstruction):

```python
series = fpm.kvectors_to_image_series(
    obj=complex_field,           # Complex object (amplitude + phase)
    k_vectors=k_vectors,         # (N, 2) illumination directions
    pupil_radius=100,            # Pupil radius in pixels (related to NA)
    wavelength=550e-9,           # Illumination wavelength [m]
    pixel_size=1e-6,             # Camera pixel size [m]
    magnification=10.0,          # Optical magnification
    pupil=None                   # Optional custom pupil function
)
```

**Physics**: For each k-vector `(kx, ky)`, the backward pass:
1. Applies a tilted plane wave: `obj × exp(i(kx·x + ky·y))`
2. Computes Fourier transform and applies pupil mask (optical transfer function)
3. Inverse transforms back to spatial domain
4. Records intensity: `I = |wave|²`

This simulates the effect of different illuminations in a physical FPM system, given a 'gold standard' image (ie. target reconstruction).

### Forward Pass: Reconstruction

Recover the high-resolution complex field from captures (the primary FPM algorithm):

```python
reconstructed = fpm.reconstruct(
    series=image_series,
    output_scale_factor=4,       # Output is 4× larger than input captures
    max_iters=10,                # Number of iterative refinement passes
    pupil_0=None                 # Optional initial pupil estimate
)
```

**Returns**: Complex-valued tensor in spatial domain with shape `(H_out, W_out)` where the output size is either:
- `output_scale_factor × input_size` (if specified)
- Auto-calculated minimum size for full k-space coverage
- Explicitly set via `output_image_size=(H, W)`

#### Reconstruction Algorithm

The implementation uses a **quasi-Newton phase retrieval** approach based on Tian et al. (2014):

1. **Initialization**: First capture initializes the high-res spectrum
2. **Iterative Update Loop**: For each capture at k-vector `(kx, ky)`:
   - Extract relevant Fourier patch from global object estimate
   - Apply pupil function and transform to spatial domain
   - **Intensity constraint**: Replace estimated magnitude with √(measured intensity)
   - **Back-propagate**: Transform corrected wave to Fourier domain
   - **Joint optimization**: Update both object spectrum and pupil using quasi-Newton step
   - **Support constraint**: Enforce binary pupil support

The quasi-Newton update (Tian's method) balances updates between object and pupil:

```python
# Object update
obj_update = pupil* · Δwave / (|pupil|² + α)

# Pupil update
pupil_update = obj* · Δwave / (|obj|² + β)
```

where `Δwave = wave_corrected - wave_estimated` and α, β are regularization parameters.

**Key parameters**:
- `alpha=1.0`: Object update regularization (prevents division by zero)
- `beta=1000.0`: Pupil update regularization (pupil converges slower than object)

### Converting Physical LED Positions

If you have physical LED array coordinates, convert them to k-vectors:

```python
import numpy as np

# LED positions in 3D space (meters), with (0,0,0) at optical axis
led_positions = np.array([
    [0.01, 0.0, 0.05],    # x, y, z for each LED
    [0.0, 0.01, 0.05],
    # ... more LEDs
])

k_vectors = fpm.spatial_to_kvectors(
    led_positions=led_positions,
    wavelength=550e-9,
    pixel_size=1e-6,
    sample_to_lens_distance=0.05,      # 50 mm
    lens_to_sensor_distance=0.50       # 500 mm (10× magnification)
)

# Now use k_vectors with kvectors_to_image_series()
```

## Evaluation & Visualization

Compare reconstruction quality:

```python
from fpm_py import plot_comparison_with_histograms

# Create comparison plots with histograms and Fourier spectra
fig, stats = plot_comparison_with_histograms(
    images=[target, recon_1iter, recon_10iter],
    titles=["Ground Truth", "1 Iteration", "10 Iterations"],
    reference_idx=0  # Use first image as reference for stat calculations
)
plt.show()

# Access quantitative metrics
print(f"SSIM: {stats['ssim']}")
print(f"PSNR: {stats['psnr']}")
```

The `stats` dictionary includes:
- **SSIM**: Structural similarity index
- **PSNR**: Peak signal-to-noise ratio
- **MSE/RMSE**: Mean squared error metrics
- **NCC**: Normalized cross-correlation
- Plus histogram statistics (mean, std, skewness, kurtosis, entropy)

## Implementation Details

### Device Management

The library automatically selects the best available device (CUDA > MPS > CPU):

```python
from fpm_py import best_device

device = best_device()  # Auto-selects GPU if available
series = series.to(device)  # Move data to device
```

### FFT Conventions

All Fourier transforms use **centered** FFTs via `fftshift`/`ifftshift`:

```python
def ft(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))

def ift(x):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x)))
```

This ensures DC component is at the array center, matching the physics convention where k=0 is the optical axis.

### Memory Optimization

- All computations use `torch.no_grad()` (no autograd overhead)
- Float32 precision throughout (GPU-optimized)
- In-place operations where possible (`add_`, `mul_`, `div_`)
- Output dimensions forced to even numbers for FFT efficiency

### Coordinate System

**k-space**: The k-vector `(kx, ky)` represents the **transverse** component of the wavevector, scaled by effective pixel size:

```
k_scaled = (2π/λ) sin(θ) × (pixel_size / magnification)
```

where θ is the illumination angle from the optical axis.

**Fourier patch extraction**: For k-vector `(kx, ky)`, the crop location in the global spectrum is:

```python
x = center_x + int(kx / du)
y = center_y + int(ky / du)
```

This shift-and-multiply theorem maps illumination angles to Fourier space shifts.

## Architecture

```
src/fpm_py/
├── core/
│   ├── structs.py      # Data structures (ImageCapture, ImageSeries)
│   ├── forward.py      # Reconstruction algorithm (forward pass)
│   └── backward.py     # Simulation (backward pass)
├── utils/
│   ├── math_utils.py   # FFT helpers, geometry, overlap operations
│   └── data_utils.py   # Image loading, device selection
└── analysis/
    └── metrics.py      # SSIM, PSNR, visualization
```

**Note**: The naming convention:
- `forward.py` = **reconstruction** (captures → high-res image) - the forward pass
- `backward.py` = **simulation** (high-res image → captures) - the backward pass (inverse of reconstruction)

## References

This implementation is based on:

1. **Tian et al. (2014)** - "Multiplexed coded illumination for Fourier Ptychography with an LED array microscope" - Quasi-Newton optimization
2. **Zheng et al. (2013)** - "Wide-field, high-resolution Fourier ptychographic microscopy" - Original FPM paper
3. **Ou et al. (2015)** - "Embedded pupil function recovery for Fourier ptychographic microscopy" - Pupil recovery method

For the mathematical foundations of Fourier ptychography, see: https://en.wikipedia.org/wiki/Fourier_ptychography

## Example: Full Pipeline

See [`example.py`](example.py) for a complete working example that:
1. Loads a test image (USAF resolution target)
2. Generates a 5×5 grid of k-vectors
3. Simulates FPM captures
4. Reconstructs with 1 and 10 iterations
5. Visualizes results with comparison metrics

## License

MIT License - See LICENSE file for details.

## Contributing

Issues and pull requests welcome at: https://github.com/rspcunningham/fpm-py

---

**Version**: 2.0.0 | **Author**: Robin Cunningham
