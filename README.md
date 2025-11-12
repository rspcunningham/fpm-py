# Fourier Ptychographic Microscopy - PyTorch Implementation

A clean, GPU-accelerated implementation of Fourier ptychographic microscopy (FPM) using differentiable physics and
automatic differentiation. This repository demonstrates both the forward imaging model and the inverse reconstruction
problem using gradient-based optimization.

## Overview

Fourier ptychography is a computational imaging technique that combines measurements from multiple illumination angles to
synthesize a high-resolution, quantitative phase image. This implementation leverages PyTorch's autodiff to solve the
inverse problem: recovering both the complex object and pupil function from intensity-only measurements.

## Forward Model

The forward model implements the physics of Fourier ptychographic imaging:

```
Given: O (complex object), P (pupil function), {k_i} (illumination angles)
Returns: {I_i} (intensity measurements)
```

**Physical Process (per illumination angle):**

1. **Phase Ramp Application**: Multiply the object by `exp(i·2π·(kx·x + ky·y)/N)` to simulate tilted plane wave
illumination. In the Fourier domain, this shifts the spectrum.

2. **Fourier Transform**: Transform the tilted object to frequency space where the pupil filtering occurs.

3. **Pupil Filtering**: Multiply by the pupil function P, which represents the limited numerical aperture of the imaging
system. Each illumination angle probes a different region of the object's Fourier spectrum.

4. **Inverse Transform**: Return to spatial domain to obtain the complex field at the detector.

5. **Intensity Measurement**: Square the absolute value to simulate camera measurement, `I = |field|²`. This loses phase
information, making reconstruction non-trivial.

**Implementation** (`src/ptych/forward.py`):
The forward model is fully vectorized and processes all illumination angles in parallel using batched operations. The key
insight is that multiplication by a phase ramp in spatial domain corresponds to a shift in frequency domain, allowing
efficient GPU computation of the entire imaging process. (ie, this is why we use the phase ramp instead of just shifting the spectrum in k space)

## Synthetic Data Generation

The example code (`main.py`) generates synthetic measurements:

1. **Ground Truth Object**: Loads a test image (bars pattern) and constructs a complex object where amplitude and phase
are related. The complex representation `O = A·exp(i·φ)` encodes both absorption (amplitude) and optical path length
(phase).

2. **Pupil Function**: Creates a circular aperture (radius 50 pixels) representing the objective lens. This finite
aperture limits the spatial frequencies that can be captured in any single measurement.

3. **Illumination Grid**: Generates an 11×11 grid of illumination angles (121 total), simulating an LED array. Each angle
is parameterized by wavevector components (kx, ky).

4. **Forward Simulation**: Passes the object, pupil, and illumination angles through the forward model to generate 121
simulated intensity images. Each measurement captures different frequency content due to the varying illumination angles.

This synthetic data allows validation of the reconstruction algorithm against known ground truth.

## Training Loop & Inverse Problem

The training loop (`src/ptych/train.py`) solves the inverse problem:

```
Given: {I_i} (measured intensities), {k_i} (known illumination angles)
Find: O (object), P (pupil) that minimize ||forward_model(O, P, k) - I||²
```

**Algorithm:**

1. **Initialization**: Both object O and pupil P are initialized as complex-valued tensors (0.5 + 0i) with gradients
enabled. Starting from a neutral initialization avoids biasing the reconstruction.

2. **Forward Pass**: For each iteration, the current estimates of O and P are passed through the forward model with all
illumination angles to predict what the measurements should be.

3. **Loss Computation**: Mean squared error between predicted intensities and measured intensities across all pixels and
all captures. This differentiable loss allows gradient-based optimization.

4. **Backpropagation**: PyTorch's autograd computes gradients ∂loss/∂O and ∂loss/∂P by backpropagating through the entire
forward model, including FFTs, complex multiplications, and absolute value operations.

5. **Optimization Step**: AdamW optimizer updates both O and P simultaneously using the computed gradients. Adaptive
learning rates help navigate the non-convex optimization landscape.

**Key Insight**: Although individual measurements lose phase information (cameras measure intensity only), the redundancy
from multiple illumination angles provides sufficient constraints. The overlapping regions in Fourier space from different
illuminations enable recovery of both amplitude and phase through iterative optimization.

**Convergence**: The optimization typically converges in ~50 epochs, with the loss curve showing rapid initial descent
followed by refinement. The reconstructed object closely matches the ground truth, demonstrating successful phase
retrieval.

## Code Structure

```
fpm_py/
├── main.py                    # Complete example: data generation → reconstruction
├── data/
│   └── bars.png              # Test image (512×512 grayscale)
├── src/ptych/
│   ├── __init__.py           # Package exports (forward_model)
│   ├── forward.py            # Forward model implementation
│   ├── train.py              # Training loop for inverse problem
│   └── analysis.py           # Visualization utilities
└── pyproject.toml            # Project dependencies
```

## Onboarding

### Prerequisites

Install `uv` (fast Python package manager):
```bash
brew install uv
```

### Configuration

Edit `main.py` line 9 to set the appropriate device for your system:

- **NVIDIA GPU**: `device = "cuda"`
- **Apple Silicon (M1/M2/M3)**: `device = "mps"`
- **CPU only**: `device = "cpu"`

### Running the Example

```bash
uv run main.py
```

This will:
1. Load the test image and create synthetic measurements (121 captures)
2. Run 50 epochs of reconstruction optimization
3. Display loss convergence and comparison plots
4. Show the successfully reconstructed amplitude and phase

Expected runtime: ~2-5 seconds on GPU, ~30-60 seconds on CPU.

## Implementation Details

**Computational Efficiency:**
- Fully batched operations process all illumination angles simultaneously
- GPU acceleration via PyTorch
- Complex-valued gradients handled automatically by PyTorch autograd

**Numerical Considerations:**
- Complex64 precision (32-bit float for real and imaginary components)
- Proper FFT centering via fftshift/ifftshift
- k-vector normalization by image dimensions for correct phase ramp scaling

**Dependencies:**
- PyTorch 2.9.0+ (computation + autodiff)
- torchvision (image I/O)
- matplotlib (visualization)
- tqdm (progress bars)
