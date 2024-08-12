# load image_series.pt

from fpm_py.data import ImageSeries, ImageCapture
import fpm_py as fpm
import torch
import matplotlib.pyplot as plt

# Load the image series
data = ImageSeries.from_dict('datasets/data.pt')
print(data)

output = fpm.reconstruct(data, output_scale_factor=4)