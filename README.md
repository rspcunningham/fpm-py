# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

Base functionality:

```bash
pip install fpm-py
```

The library exposes the `reconstruct` function for running the algorithm:

```python3
import fpm_py as fpm

output = fpm.reconstruct(stack)
```
