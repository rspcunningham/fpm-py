# Test 1 Summary

Fixed settings: `CROP_SIZE=256`, `N_CAPTURES=61`, `UPSAMPLE_RATIO=4`, `tile_batch_size=4`, `EPOCHS=1000`, `device=mps`.

## Main result

- Lowest mean total runtime: `ROI_SIZE=128` (4 tiles) at `220.79s`.
- Best mean final loss: `ROI_SIZE=64` (16 tiles) at `0.007768`.
- Relative to `ROI_SIZE=256`, the fastest condition is `1.44x` faster on mean total runtime.
- Relative to `ROI_SIZE=256`, the best-loss condition reduces final loss by `6.00%`.

## Condition Means

- `ROI_SIZE=256` (1 tiles): mean runtime `318.06 ± 100.34s`, mean runtime/epoch `0.318062 ± 0.100337s`, mean final loss `0.008264 ± 0.000000`.
- `ROI_SIZE=128` (4 tiles): mean runtime `220.79 ± 61.28s`, mean runtime/epoch `0.220793 ± 0.061277s`, mean final loss `0.008135 ± 0.000000`.
- `ROI_SIZE=64` (16 tiles): mean runtime `258.34 ± 102.80s`, mean runtime/epoch `0.064585 ± 0.025700s`, mean final loss `0.007768 ± 0.000000`.
