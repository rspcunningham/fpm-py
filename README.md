# fpm-py

Fourier ptychography is a computational image reconstruction technique that allows one high-resolution microscopic image to be acquired by combining multiple lower-resolution images captured at differing optical illumination angles.

More information about Fourier ptychography can be found [here](https://en.wikipedia.org/wiki/Fourier_ptychography).

## Instalation

```bash
pip install fpm-py
```

## Usage

The library exposes the `ImageCapture` data class and the `reconstruct` function for running the algorithm. Data should be provided through the Stack (somewhat analogous to a DataLoader in pytorch). The Stack is a list of ImageCapture objects, following the signature:

```python3
class ImageCapture:
    image: np.ndarray # the image to be fed into the algorithm.
    k_vector: np.ndarray # k-space vector of the form [k_x, k_y] associated with this image capture.
```

To built the high-resolution image, simply run the `reconstruct` function. The only required parameter is `effective_magnification`, which is a property of the hardware used. It is the ratio of the image magnification (determined by the lens) and the physical size of each pixel on the imaging sensor, in microns (determined by the sensor).

```python3
import fpm_py as fpm

output = fpm.reconstruct(stack, effective_magnification)
```

## Upcoming

1. More `optimizer` and `iteration_terminator`s
2. Complete documentation
3. Public 10k+ image dataset
4. Performance metrics, to assess how well the resulting reconstruction worked.
5. Full testing

For access to [experimental](https://github.com/rspcunningham/fpm-py/fpm_py/experimental) features, simply:

```python3
import experimental from fpm-py as fpm
```
