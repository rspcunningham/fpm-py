import fpm_py as fpm
import matplotlib.pyplot as plt

from fpm_py.evaluation import sail

original = fpm.ImageSeries.from_dict("datasets/example.pt")
recon = fpm.reconstruct(original)

plt.imshow(recon.abs().cpu().numpy(), cmap="gray")
plt.show()

sail_metric = sail(original, recon)
print(f"The SAIL metric for this reoncstruction is: {sail_metric}")