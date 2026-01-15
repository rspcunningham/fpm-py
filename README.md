# FPM-Py: Fourier Ptychography in PyTorch

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/rspcunningham/fpm-py.git
   cd fpm-py
   ```

2. Install the required dependencies:
   ```bash
   uv sync
   ```

3. Generate demo synthetic data:
   ```bash
   uv run synthetic_demo.py
   ```

4. Run a reconstruction:
   ```bash
   uv run reconstruction_demo.py
   ```

Note: if you don't have uv, install it first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
