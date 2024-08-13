# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

This library implements the Fourier ptychography technique in python to allow continued research and development. Existing implementation are undocumented, unreadable, and in MATLAB :frowning:.

## Structure

```plaintext
FPM-PY/
├── .github/
│   └── workflows/
│       └── publish.yml
├── .pytest_cache/
├── datasets/
│   └── example.pt
├── fpm_py/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── algorithm.py
│   ├── data.py
│   ├── iteration_terminators.py
│   ├── optimizers.py
│   └── utils.py
├── tests/
│   ├── __pycache__/
│   ├── __init__.py
│   └── test_utils.py
├── .gitignore
├── example.py
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

Example usage can be found in [`example.py`](https://github.com/rspcunningham/fpm-py/blob/get-algo-working-with-real-data/example.py) and below under the 'Usage' heading.

The main module source is located under `/fpm_py`:

- `algorithm.py`: contains the core fpm algorithm function.
- `data.py`: contains dataclasses for `ImageCapture` and `ImageSeries`, the core datastructures used by fpm.
  - `ImageCapture`: a single image and its associated k-space vector [k_x, k_y].
  - `ImageSeries`: a complete set of `ImageCapture`s (the stack), and capture series metadata needed for reconstruction. When creating a series, only (`optical_magnification` and sensor `pixel_size`) or `effective_magnification` are needed. Device is automatically assigned by pytorch following the heirarchy: `cuda > mps > cpu`.
  - Functions to load a dataset (ie an `ImageSeries`) or save one to the disk.
- `iteration_terminators.py`: functions that return `True` when iteration should be stopped.
- `optimizers.py`: functions that update the object and pupil using the updated wavefunction.
- `utils.py`: some helpful functions and abstractions.

The dataset in `/datasets/example.pt` is for a USAF bar pattern. It is saved as a pickled dictionary.

Test cases for the module are under `/tests` --> these need to be worked on, lol.

## Installation

To use in your project: `pip install fpm-py`

## Usage

See example usage in `example.py`, and below:

```python3
import fpm_py as fpm
import matplotlib.pyplot as plt

# load example dataset
dataset = fpm.ImageSeries.from_dict("datasets/example.pt")

# reconstruct the object
output = fpm.reconstruct(dataset)

# plot the output
plt.imshow(output.abs().cpu().numpy(), cmap="gray")
plt.show()
```

## Upcoming

1. Performance metric(s), to assess how well the resulting reconstruction worked.
2. Full testing
3. Complete documentation
4. Public 10k+ image dataset
