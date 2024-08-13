import fpm_py as fpm
import matplotlib.pyplot as plt

from fpm_py.optimizers import simple_grad_descent

# load example dataset
dataset = fpm.ImageSeries.from_dict("datasets/example.pt")

output = fpm.reconstruct(dataset, output_scale_factor=4, optimizer=simple_grad_descent)

plt.imshow(output.abs().cpu().numpy(), cmap="gray")
plt.show()