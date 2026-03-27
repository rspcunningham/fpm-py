# Reconstruction Scaling Plan

This file now tracks experiment status and the remaining scaling questions for `reconstruction_demo.py`.

## Status

- Test 1: complete
- Test 2: complete
- Test 3: complete
- Test 5: complete
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

Status: complete

Question:
- For a fixed reconstruction problem, is runtime dominated by tile count alone, or can larger `tile_batch_size` materially improve throughput?

Measured setup:
- `CROP_SIZE=512`
- `ROI_SIZE=128`
- `N_CAPTURES=61`
- `UPSAMPLE_RATIO=4`
- `EPOCHS=1000`
- `device=mps`

Measured sweep:
- `tile_batch_size in {1, 2, 4, 8}`

Measured results:
- `tile_batch_size=1`: `1290.24s`, final loss `0.005520`
- `tile_batch_size=2`: `783.55s`, final loss `0.006150`
- `tile_batch_size=4`: `970.81s`, final loss `0.007514`
- `tile_batch_size=8`: `624.79s`, final loss `0.007843`

Main result:
- Fastest total runtime: `tile_batch_size=8`
- Best final loss: `tile_batch_size=1`
- Throughput and reconstruction quality did not share the same optimum in this sweep

Artifacts:
- `experiments/test3_fixed_problem_vary_tile_batch_size/test3_comparison.png`
- `experiments/test3_fixed_problem_vary_tile_batch_size/test3_summary.csv`
- `experiments/test3_fixed_problem_vary_tile_batch_size/summary.md`
- Per-run stitched objects and metric plots are in `experiments/test3_fixed_problem_vary_tile_batch_size/test3_*`

Notes:
- The `tile_batch_size=4` point was reused from Test 2 because it had identical geometry.
- This test confirmed that batch-count overhead matters materially at larger tile grids.

## Test 5: Fixed geometry, vary upsample ratio

Status: complete

Question:
- Does increasing `UPSAMPLE_RATIO` produce meaningfully better reconstructions, and what is the runtime cost?

Purpose:
- Separate "larger latent reconstruction grid" from "better physically useful detail".
- Quantify whether higher upsampling buys real reconstruction quality or mainly increases compute and output image size.

Measured setup:
- `CROP_SIZE=128`
- `ROI_SIZE=128`
- `tile_batch_size=1`
- `N_CAPTURES=61`
- `EPOCHS=1000`
- `device=mps`

Measured sweep:
- `UPSAMPLE_RATIO in {4, 8, 16}`

Measured results:
- `UPSAMPLE_RATIO=4`: `56.09s`, final loss `0.004387`
- `UPSAMPLE_RATIO=8`: `219.99s`, final loss `0.004409`
- `UPSAMPLE_RATIO=16`: `1077.48s`, final loss `0.004385`

Main result:
- Fastest runtime: `UPSAMPLE_RATIO=4`
- Best final loss: `UPSAMPLE_RATIO=16`
- `16x` was `19.21x` slower than `4x` while improving final loss by only about `0.04%`

Artifacts:
- `experiments/test5_fixed_geometry_vary_upsample_ratio/test5_comparison.png`
- `experiments/test5_fixed_geometry_vary_upsample_ratio/test5_summary.csv`
- `experiments/test5_fixed_geometry_vary_upsample_ratio/summary.md`
- Per-run stitched objects and metric plots are in `experiments/test5_fixed_geometry_vary_upsample_ratio/test5_*`

Interpretation:
- Runtime rose extremely steeply with `UPSAMPLE_RATIO`.
- At this single-tile geometry, larger upsample ratios did not buy a meaningful loss improvement relative to their runtime cost.
- This test isolated upsample effects cleanly by avoiding tile-batching confounds.

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
3. Run Test 4 on one representative geometry after choosing the target upsample regime.
4. Re-estimate full-frame runtime under the improved batch size, epoch budget, and chosen upsample ratio.
