# Test 2 Summary

Measured settings: `ROI_SIZE=128`, `tile_batch_size=4`, `N_CAPTURES=61`, `UPSAMPLE_RATIO=4`, `EPOCHS=1000`, `device=mps`.

## Measured runs

- `CROP_SIZE=128` (1 tiles, 1 batch): total runtime `53.68s`, runtime/epoch `0.053684s`, final loss `0.004387`.
- `CROP_SIZE=256` (4 tiles, 1 batch): total runtime `164.68s`, runtime/epoch `0.164681s`, final loss `0.008135`.
- `CROP_SIZE=512` (16 tiles, 4 batches): total runtime `970.81s`, runtime/epoch `0.242702s`, final loss `0.007514`.

## Extrapolation

- Fitted empirical power law from measured total runtime: `runtime ~= 0.00191444 * crop_size^2.0883`.
- Projected `CROP_SIZE=2304`: `5.59` hours.
- Projected `CROP_SIZE=2432`: `6.26` hours.
- These projections assume the current scaling regime continues; they are not direct measurements.
