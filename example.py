import fpm_py as fpm
import matplotlib.pyplot as plt

# load example dataset
dataset = fpm.ImageSeries.from_dict("datasets/example.pt")

# reconstruct the object
output = fpm.reconstruct(dataset)

# plot the output
plt.imshow(output.abs().cpu().numpy(), cmap="gray")
plt.show()
