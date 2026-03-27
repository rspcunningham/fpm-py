# Reconstruction Scaling Plan

This file now tracks experiment status and the remaining scaling questions for `reconstruction_demo.py`.

## Status

- Test 1: complete
- Test 2: complete
- Test 3: in progress
- Test 5: pending
- Test 4: pending

## Test 1: Fixed field of view, vary tile size

Status: complete

Question:
- For a fixed reconstructed field of view, how does runtime change when the same crop is partitioned into more, smaller tiles?

Fixed settings:
- `CROP_SIZE=256`
- `N_CAPTURES=61`
- `UPSAMPLE_RATIO=4`
- `tile_batch_size=4`
- `EPOCHS=1000`
- `device=mps`

Sweep:
- `ROI_SIZE in {256, 128, 64}`
- Conditions were run twice each

Main result:
- Fastest mean runtime: `ROI_SIZE=128` at `220.79s`
- Best mean final loss: `ROI_SIZE=64` at `0.007768`
- Relative to `ROI_SIZE=256`, the fastest condition was `1.44x` faster on mean total runtime

Artifacts:
- `experiments/test1_fixed_fov_vary_tile_size/test1_comparison.png`
- `experiments/test1_fixed_fov_vary_tile_size/test1_summary.csv`
- `experiments/test1_fixed_fov_vary_tile_size/test1_condition_means.csv`
- `experiments/test1_fixed_fov_vary_tile_size/summary.md`
- Per-run stitched objects and metric plots are in `experiments/test1_fixed_fov_vary_tile_size/test1_*`

Interpretation:
- Runtime does not improve monotonically with smaller tiles.
- `ROI_SIZE=128` was the best throughput point in this sweep.
- `ROI_SIZE=64` improved loss slightly, but did not beat `ROI_SIZE=128` on mean total runtime.

## Test 2: Fixed tile size, vary field of view

Status: complete

Question:
- With tile size fixed, how does runtime scale as the reconstructed field of view grows?

Fixed settings:
- `ROI_SIZE=128`
- `tile_batch_size=4`
- `N_CAPTURES=61`
- `UPSAMPLE_RATIO=4`
- `EPOCHS=1000`
- `device=mps`

Measured runs:
- `CROP_SIZE=128`: `53.68s`, final loss `0.004387`
- `CROP_SIZE=256`: `164.68s`, final loss `0.008135`
- `CROP_SIZE=512`: `970.81s`, final loss `0.007514`

Main result:
- Measured runtime followed an empirical power law of approximately `runtime ~= 0.00191444 * crop_size^2.0883`
- Projected `CROP_SIZE=2304`: `5.59` hours
- Projected `CROP_SIZE=2432`: `6.26` hours

Artifacts:
- `experiments/test2_fixed_tile_size_vary_fov/test2_comparison_and_projection.png`
- `experiments/test2_fixed_tile_size_vary_fov/test2_summary.csv`
- `experiments/test2_fixed_tile_size_vary_fov/test2_full_frame_projection.csv`
- `experiments/test2_fixed_tile_size_vary_fov/summary.md`
- `experiments/test2_fixed_tile_size_vary_fov/on_axis_capture.png`
- Per-run stitched objects and metric plots are in `experiments/test2_fixed_tile_size_vary_fov/test2_*`

Interpretation:
- Under the current settings, scaling is already superlinear in crop size over the measured range.
- Full-frame reconstruction at roughly `2300` pixels is projected to take hours if the current regime holds.

## Test 3: Fixed reconstruction problem, vary tile batch size

Status: in progress

Question:
- For a fixed reconstruction problem, is runtime dominated by tile count alone, or can larger `tile_batch_size` materially improve throughput?

Current setup:
- `CROP_SIZE=512`
- `ROI_SIZE=128`
- `N_CAPTURES=61`
- `UPSAMPLE_RATIO=4`
- `EPOCHS=1000`
- `device=mps`

Planned sweep:
- `tile_batch_size in {1, 2, 4, 8}`

Notes:
- The `tile_batch_size=4` point already exists from Test 2 with the same reconstruction geometry and can be reused.
- This is the next key scaling-law test because a full-frame `2304 x 2304` crop at `ROI_SIZE=128` implies `324` tiles, so batching efficiency directly affects viability.

Success criteria:
- Record total runtime, runtime per epoch, and final loss for each batch size.
- Generate a comparison figure and summary under `experiments/`.
- Keep only `stitched_object.png` and `reconstruction_metrics.png` in per-run folders after aggregation.

## Test 5: Fixed geometry, vary upsample ratio

Status: pending

Question:
- Does increasing `UPSAMPLE_RATIO` produce meaningfully better reconstructions, and what is the runtime cost?

Purpose:
- Separate "larger latent reconstruction grid" from "better physically useful detail".
- Quantify whether higher upsampling buys real reconstruction quality or mainly increases compute and output image size.

Proposed fixed settings:
- Use one representative geometry first, likely `CROP_SIZE=256`
- Keep `ROI_SIZE=128`
- Keep `N_CAPTURES=61`
- Keep `EPOCHS=1000` initially
- Keep `device=mps`
- Use the best `tile_batch_size` identified by Test 3

Sweep:
- `UPSAMPLE_RATIO in {4, 8, 16}`
- If `16` does not fit in memory or is prohibitively slow, stop at `8`

Measurements:
- Total runtime
- Runtime per epoch
- Final loss
- Peak memory usage if feasible

Quality evaluation:
- Compare stitched-object crops at matched field of view
- Compare radial Fourier magnitude / spectrum of the reconstruction
- Compare edge sharpness or line-profile contrast on the same ROI
- If a suitable in-frame feature exists, compare contrast on that feature across upsample ratios

Interpretation rule:
- A larger `stitched_object.png` is not sufficient evidence of better reconstruction quality.
- Higher `UPSAMPLE_RATIO` is only a win if it improves either loss-quality tradeoff or the frequency/detail metrics in a meaningful way.

Expected scaling:
- With `ROI_SIZE` fixed, latent object size is `N = ROI_SIZE * UPSAMPLE_RATIO`
- The forward model operates on `[T, B, N, N]` tensors, so runtime and memory can rise steeply with upsample ratio
- Roughly, `4 -> 8` is expected to cost several times more than `4`
- `4 -> 16` may be an order of magnitude more expensive and may not be practical at larger crop sizes

## Test 4: Higher learning rate, fewer epochs

Status: pending

Question:
- Can a more aggressive optimizer setup recover similar reconstruction quality at a lower epoch budget?

Planned approach:
- Keep the reconstruction geometry fixed.
- Reduce `EPOCHS`.
- Increase learning rates in `src/ptych/core/inverse.py`.
- Keep separate learning rates for object and pupil parameters.
- Compare final loss, runtime, and qualitative reconstruction quality.

Suggested initial sweep:
- Object learning rate in `{1e-3, 3e-3, 1e-2}`
- Pupil learning rate about `10x` lower

Why it matters:
- If Test 3 improves batching efficiency and Test 5 identifies a worthwhile `UPSAMPLE_RATIO`, Test 4 is the next lever for reducing runtime.
- `epochs` is a direct linear cost multiplier, so this is likely one of the strongest remaining speed levers.

## Remaining Scaling Questions

- How much throughput improvement is available from larger `tile_batch_size` before memory pressure dominates?
- Does the `tile_batch_size` optimum change as `CROP_SIZE` increases beyond `512`?
- How much of the projected full-frame cost can be reduced by lowering `EPOCHS` without unacceptable quality loss?
- Is `ROI_SIZE=128` still the right operating point once batching and lower-epoch solves are both optimized?
- Does a higher `UPSAMPLE_RATIO` recover materially better detail, or mostly increase compute and output size?

## Practical Next Steps

1. Finish Test 3 and identify the best feasible `tile_batch_size`.
2. Update the full-frame projection using the best Test 3 throughput point.
3. Run Test 5 on one representative geometry to evaluate the quality-vs-cost tradeoff of higher `UPSAMPLE_RATIO`.
4. Run Test 4 on one representative geometry after choosing the target upsample regime.
5. Re-estimate full-frame runtime under the improved batch size, epoch budget, and any chosen upsample ratio.
