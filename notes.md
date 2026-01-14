```python
predicted_intensities = F.avg_pool2d(predicted_intensities, kernel_size=downsample_factor, stride=downsample_factor)
```
