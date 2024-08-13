# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

## Installation

```bash
pip install fpm-py
```

## Usage

```python3
import fpm_py as fpm
import matplotlib.pyplot as plt

# load example dataset
dataset = fpm.ImageSeries.from_dict("datasets/example.pt")

output = fpm.reconstruct(dataset)

plt.imshow(output.abs().cpu().numpy(), cmap="gray")
plt.show()
```

## Upcoming

1. Performance metric(s), to assess how well the resulting reconstruction worked.
2. Full testing
3. Complete documentation
4. Public 10k+ image dataset
