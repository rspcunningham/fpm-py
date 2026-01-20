# FPM-Py: Fourier Ptychography in PyTorch

Fourier Ptychographic Microscopy (FPM) reconstructs high-resolution images from multiple low-resolution captures taken under varying illumination angles.

## Getting Started

1. **Clone and install**
   ```bash
   git clone https://github.com/rspcunningham/fpm-py.git
   cd fpm-py
   uv sync
   ```

2. **Generate synthetic captures**
   ```bash
   uv run synthetic_demo.py
   ```
   Reads `demo/gold.png` (a USAF resolution target) and `demo/info.json` (LED positions), then simulates what a camera would capture for each illumination angle. The forward model applies a circular pupil filter and downsamples to produce low-resolution intensity images saved as `.npy` files in `demo/captures/`.

3. **Run reconstruction**
   ```bash
   uv run reconstruction_demo.py
   ```
   Loads the captures and jointly optimizes an object and pupil estimate to minimize the difference between predicted and measured intensities. Outputs `demo/object_result.png`.

4. **(Optional) Visualize captures**
   ```bash
   uv run save_captures_as_png.py
   ```
   Converts the `.npy` capture files to PNGs for inspection. Useful for verifying that off-axis illuminations produce shifted/tilted versions of the scene.

## Requirements

Requires [uv](https://github.com/astral-sh/uv). Install with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
