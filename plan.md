# Reconstruction Scaling Plan

This file now tracks experiment status and the remaining scaling questions for `reconstruction_demo.py`.

## Status

- Test 1: complete
- Test 2: complete
- Test 3: complete
- Test 5: complete
- Test 4: complete

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
- During an active run, `mactop` showed low CPU usage, negligible disk activity, zero swap, and GPU near 99% busy. That is consistent with a GPU/backend-limited regime rather than a CPU- or I/O-bound regime.

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
- `experiments/test5_fixed_geometry_vary_object_to_capture_ratio/test5_comparison.png`
- `experiments/test5_fixed_geometry_vary_object_to_capture_ratio/test5_summary.csv`
- `experiments/test5_fixed_geometry_vary_object_to_capture_ratio/summary.md`
- Per-run stitched objects and metric plots are in `experiments/test5_fixed_geometry_vary_object_to_capture_ratio/test5_*`

Interpretation:
- Runtime rose extremely steeply with `UPSAMPLE_RATIO`.
- At this single-tile geometry, larger upsample ratios did not buy a meaningful loss improvement relative to their runtime cost.
- This test isolated upsample effects cleanly by avoiding tile-batching confounds.

## Test 4: Higher learning rate, fewer epochs

Status: complete

Question:
- Can a more aggressive optimizer setup recover similar reconstruction quality at a lower epoch budget?

Measured setup:
- `CROP_SIZE=256`
- `ROI_SIZE=128`
- `UPSAMPLE_RATIO=4`
- `tile_batch_size=4`
- `N_CAPTURES=61`
- `device=mps`

Measured sweep:
- `250`-epoch runs:
  - constant LR with object/pupil pairs `{1e-3,1e-4}`, `{3e-3,3e-4}`, `{1e-2,1e-3}`
  - cosine LR with object/pupil pairs `{3e-3,3e-4}`, `{1e-2,1e-3}`
- focused `150`-epoch follow-up:
  - constant LR with object/pupil pair `{1e-2,1e-3}`
  - cosine LR with object/pupil pair `{1e-2,1e-3}`

Measured results:
- historical reference, `1000` epochs, old default constant LR: `164.68s`, final loss `0.008135`
- `250` epochs, constant `1e-3 / 1e-4`: `40.53s`, final loss `0.036665`
- `250` epochs, constant `3e-3 / 3e-4`: `40.61s`, final loss `0.009446`
- `250` epochs, constant `1e-2 / 1e-3`: `41.15s`, final loss `0.007293`
- `250` epochs, cosine `3e-3 / 3e-4`: `41.31s`, final loss `0.018312`
- `250` epochs, cosine `1e-2 / 1e-3`: `41.49s`, final loss `0.007657`
- `150` epochs, constant `1e-2 / 1e-3`: `24.65s`, final loss `0.007570`
- `150` epochs, cosine `1e-2 / 1e-3`: `24.71s`, final loss `0.008903`

Main result:
- Best objective result overall: `250` epochs with constant `1e-2 / 1e-3`
- Best objective result at the more aggressive budget: `150` epochs with constant `1e-2 / 1e-3`
- Relative to the old `1000`-epoch default, the best `150`-epoch setting was `6.68x` faster while also improving final loss by about `6.9%`

Artifacts:
- `experiments/test4_higher_lr_fewer_epochs/test4_comparison.png`
- `experiments/test4_higher_lr_fewer_epochs/test4_summary.csv`
- `experiments/test4_higher_lr_fewer_epochs/summary.md`
- Per-run stitched objects and metric plots are in `experiments/test4_higher_lr_fewer_epochs/test4_*`

Interpretation:
- Higher learning rates are necessary when the epoch budget is cut aggressively.
- Cosine decay did not improve objective loss in this short-run regime.
- Subjectively, the `1e-2` constant and cosine runs at `150` epochs looked nearly the same, so the constant schedule is the cleaner default because it keeps the lower loss at the same runtime.
- For further scaling tests, the strongest short-solve default is constant object/intensity LR `1e-2`, pupil/k-vector LR `1e-3`, with `150` epochs if speed is the priority and `250` epochs if more optimization margin is desired.

## Remaining Scaling Questions

- How much throughput improvement is available from larger `tile_batch_size` before memory pressure dominates?
- Does the `tile_batch_size` optimum change as `CROP_SIZE` increases beyond `512`?
- How well does the new short-solve optimizer setting transfer to larger fields of view such as `CROP_SIZE=512` and beyond?
- Is `ROI_SIZE=128` still the right operating point once batching and lower-epoch solves are both optimized?
- Does a higher `UPSAMPLE_RATIO` recover materially better detail, or mostly increase compute and output size?

## Practical Next Steps

1. Re-estimate the full-frame projection using the improved short-solve optimizer setting from Test 4.
2. Validate that Test 4 setting on a larger crop, such as `CROP_SIZE=512`, to confirm the speedup carries over.
3. Revisit `ROI_SIZE` and `tile_batch_size` only after confirming the new optimizer regime at the larger field of view.
