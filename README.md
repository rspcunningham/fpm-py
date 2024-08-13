# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

## Installation

```bash
pip install fpm-py
```

## Usage

See `example.py` for how to use the module.

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

1. More `optimizer` and `iteration_terminator`s
2. Complete documentation
3. Public 10k+ image dataset
4. Performance metrics, to assess how well the resulting reconstruction worked.
5. Full testing
