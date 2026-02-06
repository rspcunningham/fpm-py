# FPM-Py: Fourier Ptychography in PyTorch

Fourier Ptychographic Microscopy (FPM) reconstructs high-resolution images from multiple low-resolution captures taken under varying illumination angles.

For up-to-date documentation and to ask questions about this repo, please [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rspcunningham/fpm-py)

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

## Data Format

The `demo/` directory follows the Cunningham Lab FPM data format. Each study consists of:
- `info.json` — manifest with optical parameters and LED positions
- `captures/` — directory of `.npy` intensity images

See [INFO_JSON_SCHEMA.md](INFO_JSON_SCHEMA.md) for the full schema specification.

## Requirements

Requires [uv](https://github.com/astral-sh/uv). Install with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
