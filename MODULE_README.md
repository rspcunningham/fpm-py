# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

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
