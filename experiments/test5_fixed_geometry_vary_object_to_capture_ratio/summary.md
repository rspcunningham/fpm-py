# Test 5 Summary

Measured settings: `CROP_SIZE=128`, `ROI_SIZE=128`, `tile_batch_size=1`, `N_CAPTURES=61`, `EPOCHS=1000`, `device=mps`.

## Measured runs

- `UPSAMPLE_RATIO=4`: total runtime `56.09s`, final loss `0.004387`, high-frequency power fraction `0.060100`.
- `UPSAMPLE_RATIO=8`: total runtime `219.99s`, final loss `0.004409`, high-frequency power fraction `0.006935`.
- `UPSAMPLE_RATIO=16`: total runtime `1077.48s`, final loss `0.004385`, high-frequency power fraction `0.002071`.

## Main result

- Fastest runtime: `UPSAMPLE_RATIO=4` at `56.09s`.
- Best final loss: `UPSAMPLE_RATIO=16` at `0.004385`.
- Highest measured high-frequency power fraction: `UPSAMPLE_RATIO=4`.
- `16x` was `19.21x` slower than `4x`.
- Relative to `4x`, `16x` changed final loss by `-0.04%` and high-frequency power fraction by `-96.55%`.

## Interpretation

- Runtime grew steeply with `UPSAMPLE_RATIO`, and the jump from `8x` to `16x` was much larger than the jump from `4x` to `8x`.
- `UPSAMPLE_RATIO=16` slightly improved final loss over `4x`, but at a very large runtime cost.
- The spectrum and preview comparisons should be read together with the runtime plot: extra pixels are only valuable if the added high-frequency structure looks consistent rather than merely sharper.
- This test used a single-tile geometry to isolate the upsample effect and avoid tile-batching confounds.
