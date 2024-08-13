# load image_series.pt

from fpm_py.data import ImageSeries, ImageCapture
import fpm_py as fpm
import torch
import matplotlib.pyplot as plt

import time

# Load the image series
data = ImageSeries.from_dict('datasets/data_2.pt')

#plt.imshow(data.image_stack[0].image.cpu().numpy(), cmap='gray')

start = time.time()
output = fpm.reconstruct(data, output_scale_factor=4)
end = time.time()

print(f"Time elapsed: {end - start} seconds")
print(output)

plt.imshow(output.abs().cpu().numpy(), cmap='gray')
plt.show()